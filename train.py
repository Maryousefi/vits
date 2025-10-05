# train.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text import get_symbols

torch.backends.cudnn.benchmark = True


def main():
    """Main single-GPU training loop for VITS (Colab version)."""
    assert torch.cuda.is_available(), "CUDA GPU required — please enable GPU in Colab."

    # Load hyperparameters
    hps = utils.get_hparams()

    # Verify filelists
    tf = getattr(hps.train, "training_files", None) or getattr(hps.data, "training_files", None)
    vf = getattr(hps.train, "validation_files", None) or getattr(hps.data, "validation_files", None)
    if not tf or not os.path.exists(tf):
        raise FileNotFoundError(f"Training filelist not found: {tf}")
    if not vf or not os.path.exists(vf):
        raise FileNotFoundError(f"Validation filelist not found: {vf}")

    # Create dataset and dataloaders
    train_dataset = TextAudioLoader(tf, hps.data)
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_dataset = TextAudioLoader(vf, hps.data)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # Initialize model
    symbols = get_symbols(hps.data.text_cleaners)
    n_symbols = len(symbols)

    net_g = SynthesizerTrn(
        n_symbols,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda()
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda()

    optim_g = torch.optim.AdamW(
        net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)
    writer = SummaryWriter(log_dir=os.path.join(hps.model_dir, "runs"))
    logger = utils.get_logger(hps.model_dir)

    # Load checkpoint if available
    global_step = 0
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
        logger.info(f"Resumed from step {global_step}")
    except Exception:
        epoch_str = 1
        global_step = 0
        logger.info("Starting training from scratch")

    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    # Training loop
    for epoch in range(epoch_str, hps.train.epochs + 1):
        net_g.train()
        net_d.train()

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()

            # === Train Discriminator ===
            with autocast(enabled=hps.train.fp16_run):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                    x, x_lengths, spec, spec_lengths
                )

                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )

                y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            torch.nn.utils.clip_grad_norm_(net_d.parameters(), 5.0)
            scaler.step(optim_d)

            # === Train Generator ===
            with autocast(enabled=hps.train.fp16_run):
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            torch.nn.utils.clip_grad_norm_(net_g.parameters(), 5.0)
            scaler.step(optim_g)
            scaler.update()

            global_step += 1

            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    f"Epoch [{epoch}] Step [{global_step}] G_loss={loss_gen_all.item():.4f} "
                    f"D_loss={loss_disc.item():.4f} lr={lr:.6f}"
                )

                writer.add_scalar("Loss/Generator", loss_gen_all.item(), global_step)
                writer.add_scalar("Loss/Discriminator", loss_disc.item(), global_step)
                writer.add_scalar("LR", lr, global_step)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, val_loader, writer, global_step)
                utils.save_checkpoint(
                    net_g, optim_g, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"G_{global_step}.pth")
                )
                utils.save_checkpoint(
                    net_d, optim_d, hps.train.learning_rate, epoch,
                    os.path.join(hps.model_dir, f"D_{global_step}.pth")
                )

        scheduler_g.step()
        scheduler_d.step()

    logger.info("Training complete!")


def evaluate(hps, generator, val_loader, writer, global_step):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(val_loader):
            x, x_lengths = x.cuda(), x_lengths.cuda()
            spec, spec_lengths = spec.cuda(), spec_lengths.cuda()
            y, y_lengths = y.cuda(), y_lengths.cuda()
            break

        y_hat, attn, mask, *_ = generator.infer(x, x_lengths, max_len=1000)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )

    writer.add_image("Eval/Mel_Gen", utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()), global_step)
    writer.add_audio("Eval/Audio_Gen", y_hat[0, :, :].cpu(), global_step, sample_rate=hps.data.sampling_rate)
    generator.train()


if __name__ == "__main__":
    main()

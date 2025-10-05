import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# Get symbol helper
from text import get_symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Single-node multi-GPU or single-GPU training (Colab friendly)."""
    assert torch.cuda.is_available(), "❌ CUDA GPU not found. Please enable GPU runtime."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    hps = utils.get_hparams()

    # -----------------------
    # Fix filelist loading
    # -----------------------
    tf = None
    vf = None
    # prefer hps.data, but also allow hps.train for compatibility
    if hasattr(hps, "data") and hasattr(hps.data, "training_files"):
        tf = hps.data.training_files
    elif hasattr(hps, "train") and hasattr(hps.train, "training_files"):
        tf = hps.train.training_files

    if hasattr(hps, "data") and hasattr(hps.data, "validation_files"):
        vf = hps.data.validation_files
    elif hasattr(hps, "train") and hasattr(hps.train, "validation_files"):
        vf = hps.train.validation_files

    if tf is None:
        raise FileNotFoundError("training_files not found in config (checked hps.data and hps.train).")
    if vf is None:
        raise FileNotFoundError("validation_files not found in config (checked hps.data and hps.train).")

    tf = os.path.expanduser(tf)
    vf = os.path.expanduser(vf)
    if not os.path.isabs(tf):
        tf = os.path.join(os.getcwd(), tf)
    if not os.path.isabs(vf):
        vf = os.path.join(os.getcwd(), vf)

    if not os.path.exists(tf):
        raise FileNotFoundError(f"Training filelist not found at: {tf}")
    if not os.path.exists(vf):
        raise FileNotFoundError(f"Validation filelist not found at: {vf}")

    # Write back to hps.data
    if not hasattr(hps, "data"):
        hps.data = type("X", (), {})()
    hps.data.training_files = tf
    hps.data.validation_files = vf

    # Add safe mel defaults
    if not hasattr(hps.data, "mel_fmin"):
        hps.data.mel_fmin = 0.0
    if not hasattr(hps.data, "mel_fmax"):
        hps.data.mel_fmax = None

    # -----------------------
    # Single vs Multi GPU
    # -----------------------
    if n_gpus == 1:
        print("🚀 Using single GPU mode.")
        run(rank=0, n_gpus=1, hps=hps)
    else:
        print(f"🚀 Using multi-GPU mode ({n_gpus} GPUs).")
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    else:
        logger = writer = writer_eval = None

    dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # -----------------------
    # Data loading
    # -----------------------
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset, num_workers=4, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler
    )

    if rank == 0:
        eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset, num_workers=2, shuffle=False, batch_size=hps.train.batch_size,
            pin_memory=True, drop_last=False, collate_fn=collate_fn
        )
    else:
        eval_loader = None

    # -----------------------
    # Model setup
    # -----------------------
    symbols = get_symbols(hps.data.text_cleaners)
    n_symbols = len(symbols)

    net_g = SynthesizerTrn(
        n_symbols,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
                           [scheduler_g, scheduler_d], scaler,
                           [train_loader, eval_loader], logger, [writer, writer_eval])
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers:
        writer, writer_eval = writers

    # handle distributed sampler epoch
    if hasattr(train_loader.batch_sampler, "set_epoch"):
        train_loader.batch_sampler.set_epoch(epoch)

    global global_step
    net_g.train()
    net_d.train()

    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                x, x_lengths, spec, spec_lengths
            )

            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax
            )
            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax
            )
            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # Discriminator pass
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        # Generator pass
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel)
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0 and global_step % hps.train.log_interval == 0:
            lr = optim_g.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] step={global_step} lr={lr:.6f}")
            writer.add_scalar("train/loss_g", loss_gen_all.item(), global_step)
            writer.add_scalar("train/loss_d", loss_disc.item(), global_step)
            writer.add_scalar("train/lr", lr, global_step)

        if rank == 0 and global_step % hps.train.eval_interval == 0 and eval_loader is not None:
            evaluate(hps, net_g, eval_loader, writer_eval)
            utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, f"G_{global_step}.pth"))
            utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, f"D_{global_step}.pth"))
        global_step += 1

    if rank == 0:
        logger.info(f"====> Epoch {epoch} complete")


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for x, x_lengths, spec, spec_lengths, y, y_lengths in eval_loader:
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            break

        y_hat, attn, mask, *_ = generator.module.infer(x[:1], x_lengths[:1], max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
        y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(),
                                          hps.data.filter_length, hps.data.n_mel_channels,
                                          hps.data.sampling_rate, hps.data.hop_length,
                                          hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)

    image_dict = {"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())}
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}

    utils.summarize(writer=writer_eval, global_step=global_step,
                    images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)
    generator.train()


if __name__ == "__main__":
    main()

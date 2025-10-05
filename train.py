import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["NUMBA_DISABLE_JIT"] = "1"  # suppress numba spam

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, DistributedBucketSampler
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text import get_symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Single-node, single-GPU or multi-GPU training entrypoint."""
    assert torch.cuda.is_available(), " GPU not available — please enable GPU in Colab!"

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    hps = utils.get_hparams()

    tf = getattr(hps.train, "training_files", None) or getattr(hps.data, "training_files", None)
    vf = getattr(hps.train, "validation_files", None) or getattr(hps.data, "validation_files", None)
    if not tf or not os.path.exists(tf):
        raise FileNotFoundError(f"Training filelist not found: {tf}")
    if not vf or not os.path.exists(vf):
        raise FileNotFoundError(f"Validation filelist not found: {vf}")

    print(f" Found training filelist: {tf}")
    print(f" Found validation filelist: {vf}")
    print(f" Using {n_gpus} GPU(s)")

    # Run single-GPU training
    run(rank=0, n_gpus=n_gpus, hps=hps)


def run(rank, n_gpus, hps):
    global global_step
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # -------------------------------
    # Dataset and Dataloader setup
    # -------------------------------
    print(" Loading datasets ...")
    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioCollate()

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=hps.train.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    symbols = get_symbols(hps.data.text_cleaners)
    n_symbols = len(symbols)
    print(f" Loaded {n_symbols} symbols")

    # -------------------------------
    # Model Setup
    # -------------------------------
    net_g = SynthesizerTrn(
        n_symbols,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    print(" Training started!")

    for epoch in range(1, hps.train.epochs + 1):
        train_one_epoch(
            rank, epoch, hps, net_g, net_d, optim_g, optim_d, scaler, train_loader, writer
        )
        evaluate(hps, net_g, eval_loader, writer_eval)

        scheduler_g.step()
        scheduler_d.step()
        utils.save_checkpoint(
            net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"G_{epoch}.pth")
        )
        utils.save_checkpoint(
            net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, f"D_{epoch}.pth")
        )
        print(f" Epoch {epoch} finished — model saved!\n")

    print(" Training complete!")


def train_one_epoch(rank, epoch, hps, net_g, net_d, optim_g, optim_d, scaler, loader, writer):
    global global_step
    net_g.train()
    net_d.train()

    progress = tqdm(loader, desc=f" Epoch {epoch}", dynamic_ncols=True)
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(progress):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

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
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel)
            loss_kl_ = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl_

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        scaler.step(optim_g)
        scaler.update()

        if batch_idx % 10 == 0:
            progress.set_postfix(
                {"G_loss": f"{loss_gen_all.item():.4f}", "D_loss": f"{loss_disc.item():.4f}"}
            )

        global_step += 1


@torch.no_grad()
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    batch = next(iter(eval_loader))
    x, x_lengths, spec, spec_lengths, y, y_lengths = [b.cuda(0) for b in batch]

    y_hat, attn, mask, *_ = generator.infer(x[:1], x_lengths[:1], max_len=1000)
    mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
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

    image_dict = {"gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())}
    audio_dict = {"gen/audio": y_hat[0, :, :]}
    utils.summarize(writer=writer_eval, global_step=global_step, images=image_dict, audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)


if __name__ == "__main__":
    main()

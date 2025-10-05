import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import plot_spectrogram_to_numpy, save_checkpoint, load_checkpoint
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss
from commons import slice_segments
import logging
import json
import argparse

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        hps = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hps["train"]["seed"])

    # Data
    train_dataset = TextAudioLoader(hps["train"]["training_files"], hps)
    train_loader = DataLoader(
        train_dataset,
        num_workers=2,
        shuffle=True,
        batch_size=hps["train"]["batch_size"],
        pin_memory=True,
        drop_last=True,
        collate_fn=TextAudioCollate()
    )
    val_dataset = TextAudioLoader(hps["train"]["validation_files"], hps)
    val_loader = DataLoader(
        val_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        collate_fn=TextAudioCollate()
    )

    # Model — FIXED HERE
    n_mel_channels = hps["data"]["n_mel_channels"]

    net_g = SynthesizerTrn(
        len(train_dataset.get_text_cleaner_symbols()),
        n_mel_channels,  # FIXED: was filter_length // 2 + 1
        hps["model"]["inter_channels"],
        hps["model"]["hidden_channels"],
        hps["model"]["filter_channels"],
        hps["model"]["n_heads"],
        hps["model"]["n_layers"],
        hps["model"]["kernel_size"],
        hps["model"]["p_dropout"],
        hps["model"]["resblock"],
        hps["model"]["resblock_kernel_sizes"],
        hps["model"]["resblock_dilation_sizes"],
        hps["model"]["upsample_rates"],
        hps["model"]["upsample_initial_channel"],
        hps["model"]["upsample_kernel_sizes"],
        hps["model"]["use_spectral_norm"],
        hps["model"]["gin_channels"]
    ).to(device)

    net_d = MultiPeriodDiscriminator().to(device)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps["train"]["learning_rate"],
        betas=hps["train"]["betas"],
        eps=hps["train"]["eps"]
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps["train"]["learning_rate"],
        betas=hps["train"]["betas"],
        eps=hps["train"]["eps"]
    )

    epoch = 0
    os.makedirs("logs", exist_ok=True)

    # Training loop
    for epoch in range(epoch, hps["train"]["epochs"]):
        net_g.train()
        net_d.train()
        start = time.time()

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)

            y_hat, l_length, attn, ids_slice, x_mask, z_mask, _ = net_g(
                x, x_lengths, spec, spec_lengths
            )

            # Losses
            loss_mel = F.l1_loss(spec, spec)
            y_slice = slice_segments(y, ids_slice, hps["train"]["segment_size"])
            y_hat_slice = slice_segments(y_hat, ids_slice, hps["train"]["segment_size"])

            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_slice, y_hat_slice)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_g = loss_gen + loss_mel * 45

            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_slice, y_hat_slice.detach())
            loss_disc, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            loss_disc.backward()
            optim_d.step()

            if batch_idx % hps["train"]["log_interval"] == 0:
                elapsed = time.time() - start
                logging.info(
                    f"Epoch {epoch} | Batch {batch_idx} | G: {loss_g.item():.4f} | D: {loss_disc.item():.4f} | Time: {elapsed:.2f}s"
                )
                start = time.time()

        if (epoch + 1) % 5 == 0:
            save_checkpoint(net_g, net_d, optim_g, optim_d, epoch + 1, f"logs/{args.model}_G.pth")

    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from utils import plot_spectrogram_to_numpy, save_checkpoint, load_checkpoint
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss
from commons import slice_segments
import logging

logging.basicConfig(level=logging.INFO)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()

    # Load config
    import json
    with open(args.config, "r", encoding="utf-8") as f:
        hps = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hps["train"]["seed"])

    # Data
    train_dataset = TextAudioLoader(hps["train"]["training_files"], hps)
    train_loader = DataLoader(train_dataset, num_workers=2, shuffle=True,
                              batch_size=hps["train"]["batch_size"],
                              pin_memory=True, drop_last=True,
                              collate_fn=TextAudioCollate())
    val_dataset = TextAudioLoader(hps["train"]["validation_files"], hps)
    val_loader = DataLoader(val_dataset, num_workers=1, shuffle=False,
                            batch_size=1, pin_memory=False,
                            collate_fn=TextAudioCollate())

    # Models
    net_g = SynthesizerTrn(
        len(train_dataset.get_text_cleaner_symbols()),
        hps["data"]["filter_length"] // 2 + 1,
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

    optim_g = torch.optim.AdamW(net_g.parameters(),
                                hps["train"]["learning_rate"],
                                betas=hps["train"]["betas"],
                                eps=hps["train"]["eps"])
    optim_d = torch.optim.AdamW(net_d.parameters(),
                                hps["train"]["learning_rate"],
                                betas=hps["train"]["betas"],
                                eps=hps["train"]["eps"])

    try:
        ckpt_g, ckpt_d, epoch = load_checkpoint(f"logs/{args.model}_G.pth", net_g, net_d, optim_g, optim_d)
        logging.info(f"Loaded checkpoint from epoch {epoch}")
    except Exception:
        logging.info("Starting from scratch.")
        epoch = 0

    net_g = DataParallel(net_g)
    net_d = DataParallel(net_d)

    # Training loop
    total_epochs = hps["train"]["epochs"]
    for epoch in range(epoch, total_epochs):
        net_g.train()
        net_d.train()
        start = time.time()

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)

            # Forward pass generator
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(
                x, x_lengths, spec, spec_lengths
            )

            # Slice target
            y_mel = spec  # Use linear spectrogram (513 channels)
            y_hat_mel = spec  # Same for prediction

            # Generator loss
            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            y = slice_segments(y, ids_slice, hps["train"]["segment_size"])
            y_hat = slice_segments(y_hat, ids_slice, hps["train"]["segment_size"])
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)

            loss_g = loss_gen + loss_mel * 45

            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            # Discriminator loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad()
            loss_disc.backward()
            optim_d.step()

            if batch_idx % hps["train"]["log_interval"] == 0:
                elapsed = time.time() - start
                logging.info(
                    f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}] "
                    f"G Loss: {loss_g.item():.4f} | D Loss: {loss_disc.item():.4f} | Time: {elapsed:.2f}s"
                )
                start = time.time()

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint(net_g, net_d, optim_g, optim_d, epoch + 1,
                            f"logs/{args.model}_G.pth")

    logging.info("Training completed successfully.")

if __name__ == "__main__":
    main()

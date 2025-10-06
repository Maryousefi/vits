import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint
from data_utils import TextAudioLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from losses import generator_loss, discriminator_loss
from commons import slice_segments, mel_spectrogram_torch
import logging
import json
import argparse

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    args = parser.parse_args()

    # -------------------
    # Load Config
    # -------------------
    with open(args.config, "r", encoding="utf-8") as f:
        hps = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hps["train"]["seed"])

    # -------------------
    # Data Loading
    # -------------------
    train_dataset = TextAudioLoader(hps["train"]["training_files"], hps)
    val_dataset = TextAudioLoader(hps["train"]["validation_files"], hps)

    collate_fn = TextAudioCollate()

    train_loader = DataLoader(
        train_dataset,
        num_workers=2,
        shuffle=True,
        batch_size=hps["train"]["batch_size"],
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # -------------------
    # Model Setup
    # -------------------
    n_mel_channels = hps["data"]["n_mel_channels"]

    # Fixed: no len() here
    n_vocab = getattr(train_dataset, "n_symbols", 300)

    net_g = SynthesizerTrn(
        n_vocab,  # number of phoneme or text symbols
        n_mel_channels,
        hps["train"]["segment_size"],
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
        n_speakers=hps["data"]["n_speakers"],
        gin_channels=hps["model"]["gin_channels"],
        use_sdp=True
    ).to(device)

    net_d = MultiPeriodDiscriminator(
        use_spectral_norm=hps["model"]["use_spectral_norm"]
    ).to(device)

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

    os.makedirs("logs", exist_ok=True)

    # Training Loop
    for epoch in range(hps["train"]["epochs"]):
        net_g.train()
        net_d.train()
        start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Get both mel spectrograms and audio waveforms
            x, x_lengths, y_mel, y_lengths, y_audio, audio_lengths = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            
            x, x_lengths = x.to(device), x_lengths.to(device)
            y_mel, y_lengths = y_mel.to(device), y_lengths.to(device)
            y_audio = y_audio.to(device)

            # Add channel dimension to audio for slicing (from 2D to 3D)
            y_audio_3d = y_audio.unsqueeze(1)  # Shape: (batch, 1, audio_length)

            # Generator forward - use mel spectrograms for training
            y_hat, l_length, attn, ids_slice, x_mask, z_mask, _ = net_g(
                x, x_lengths, y_mel, y_lengths
            )

            # Calculate mel frame length from audio segment size
            segment_size = hps["train"]["segment_size"]
            mel_segment_size = segment_size // hps["data"]["hop_length"]
            
            # Generator loss - slice all tensors to the same segment size
            y_mel_slice = slice_segments(y_mel, ids_slice, mel_segment_size)
            y_audio_slice = slice_segments(y_audio_3d, ids_slice, segment_size)
            y_hat_slice = slice_segments(y_hat, ids_slice, segment_size)
            
            # Convert generator output to mel spectrogram for mel loss
            y_hat_mel = mel_spectrogram_torch(
                y_hat_slice.squeeze(1),
                hps["data"]["filter_length"],
                hps["data"]["n_mel_channels"],
                hps["data"]["sampling_rate"],
                hps["data"]["hop_length"],
                hps["data"]["win_length"],
                hps["data"].get("mel_fmin", 0.0),
                hps["data"].get("mel_fmax", 8000.0),
                center=False
            )
            
            # Slice the generated mel to match the ground truth mel segment size
            y_hat_mel_slice = slice_segments(y_hat_mel, torch.zeros_like(ids_slice), mel_segment_size)
            
            # Prepare audio for discriminator
            y_audio_slice_disc = y_audio_slice  # Shape: (batch, 1, segment_size)
            y_hat_slice_disc = y_hat_slice      # Shape: (batch, 1, segment_size)
            
            # FIXED: Use audio waveforms for discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_audio_slice_disc, y_hat_slice_disc)
            loss_gen, _ = generator_loss(y_d_hat_g)
            
            # FIXED: Compare mel spectrograms with the same dimensions
            loss_mel = F.l1_loss(y_hat_mel_slice, y_mel_slice)
            loss_g = loss_gen + loss_mel * 45.0

            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()

            # Discriminator loss
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y_audio_slice_disc, y_hat_slice_disc.detach())
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

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                net_g, net_d, optim_g, optim_d, epoch + 1, f"logs/{args.model}_G.pth"
            )

    logging.info("Training completed successfully.")


if __name__ == "__main__":
    main()

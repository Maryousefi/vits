import os
import random
import torch
import torch.utils.data
import numpy as np
import librosa
from commons import spectrogram_torch
from utils import load_wav_to_torch, get_hparams_from_file

# ===============================================================
# Text-Audio Dataset & Collation Utilities for Single-Speaker VITS
# ===============================================================

class TextAudioLoader(torch.utils.data.Dataset):
    """
    Loads (text, audio) pairs for training.
    Each line in the filelist must have: <path>|<text>
    """

    def __init__(self, filelist_path, hps):
        self.filepaths_and_text = self._load_filepaths_and_text(filelist_path)
        self.max_wav_value = hps["data"]["max_wav_value"]
        self.sampling_rate = hps["data"]["sampling_rate"]
        self.filter_length = hps["data"]["filter_length"]
        self.hop_length = hps["data"]["hop_length"]
        self.win_length = hps["data"]["win_length"]
        self.n_mel_channels = hps["data"]["n_mel_channels"]  # should be 80
        self.text_cleaners = hps["data"]["text_cleaners"]
        self.add_blank = hps["data"].get("add_blank", True)

    def _load_filepaths_and_text(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            filepaths_and_text = [line.strip().split("|") for line in f.readlines()]
        return filepaths_and_text

    def get_text(self, text):
        """
        You can adapt this for Persian text processing.
        For now it converts text to a list of character IDs.
        """
        # Simple placeholder cleaner; replace with your Persian cleaner if needed
        text = text.strip().lower()
        symbols = list(text)
        text_norm = [ord(s) % 256 for s in symbols]  # crude numeric mapping
        return torch.LongTensor(text_norm)

    def get_mel(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            audio = librosa.resample(
                audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate
            )
            audio = torch.from_numpy(audio)

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        # Compute Mel spectrogram - FIXED PARAMETER NAMES
        mel = spectrogram_torch(
            audio_norm,
            self.filter_length,  # n_fft
            self.sampling_rate,
            self.hop_length,     # hop_size
            self.win_length,     # win_size
            self.n_mel_channels,
            center=False
        )
        mel = torch.squeeze(mel, 0)
        return mel, audio_norm

    def __getitem__(self, index):
        filepath, text = self.filepaths_and_text[index]
        text = self.get_text(text)
        mel, audio = self.get_mel(filepath)
        return (text, mel, audio)

    def __len__(self):
        return len(self.filepaths_and_text)

    def get_text_cleaner_symbols(self):
        # Rough estimate of text vocabulary size
        return 256


class TextAudioCollate:
    """Collates training batches from TextAudioLoader."""

    def __call__(self, batch):
        # Sort batch by descending text length for packing efficiency
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        text_lengths = torch.LongTensor([len(x[0]) for x in batch])
        mel_lengths = torch.LongTensor([x[1].size(1) for x in batch])

        # Pad text sequences
        max_text_len = text_lengths.max().item()
        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        for i, (text, _, _) in enumerate(batch):
            text_padded[i, :text.size(0)] = text

        # Pad Mel spectrograms
        max_mel_len = mel_lengths.max().item()
        n_mel_channels = batch[0][1].size(0)
        mel_padded = torch.zeros(len(batch), n_mel_channels, max_mel_len)
        for i, (_, mel, _) in enumerate(batch):
            mel_padded[i, :, :mel.size(1)] = mel

        # Pad audio waveforms
        max_audio_len = max([x[2].size(1) for x in batch])
        audio_padded = torch.zeros(len(batch), 1, max_audio_len)
        for i, (_, _, audio) in enumerate(batch):
            audio_padded[i, :, :audio.size(1)] = audio

        return text_padded, text_lengths, mel_padded, mel_lengths, audio_padded, torch.LongTensor(
            [x[2].size(1) for x in batch]
        )


# ===============================================================
# Utility Functions
# ===============================================================

def create_dataloader(hps, is_val=False):
    filelist = (
        hps["train"]["validation_files"]
        if is_val
        else hps["train"]["training_files"]
    )
    dataset = TextAudioLoader(filelist, hps)
    collate_fn = TextAudioCollate()

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1 if is_val else hps["train"]["batch_size"],
        num_workers=2,
        shuffle=not is_val,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=not is_val,
    )
    return loader, dataset


if __name__ == "__main__":
    # Debug loader
    hps = get_hparams_from_file("configs/fa_single_speaker.json")
    train_loader, _ = create_dataloader(hps)
    for batch in train_loader:
        print("Loaded batch shapes:")
        for tensor in batch:
            print(tensor.shape)
        break

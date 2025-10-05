# data_utils.py
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from text import text_to_sequence


class TextAudioLoader(Dataset):
    """
    Loads (text, mel, wav) pairs from filelist for single-GPU training.
    Each line of filelist: wav_path|text
    """

    def __init__(self, filelist_path, data_hparams):
        assert os.path.isfile(filelist_path), f"Filelist not found: {filelist_path}"
        self.audiopaths_and_text = self._load_filelist(filelist_path)
        self.data_hparams = data_hparams

        self.sampling_rate = getattr(data_hparams, "sampling_rate", 22050)
        self.n_mel_channels = getattr(data_hparams, "n_mel_channels", 80)
        self.filter_length = getattr(data_hparams, "filter_length", 1024)
        self.hop_length = getattr(data_hparams, "hop_length", 256)
        self.win_length = getattr(data_hparams, "win_length", 1024)
        self.mel_fmin = getattr(data_hparams, "mel_fmin", 0.0)
        self.mel_fmax = getattr(data_hparams, "mel_fmax", None)
        self.text_cleaners = getattr(data_hparams, "text_cleaners", ["persian_cleaners"])

    def _load_filelist(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        pairs = []
        for line in lines:
            parts = line.split("|", 1)
            if len(parts) < 2:
                continue
            wav, text = parts
            pairs.append((wav, text))
        return pairs

    def __len__(self):
        return len(self.audiopaths_and_text)

    def _load_audio(self, path):
        wav, _ = librosa.load(path, sr=self.sampling_rate)
        wav = wav.astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
        )
        mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return torch.FloatTensor(wav), torch.FloatTensor(mel)

    def __getitem__(self, index):
        wav_path, text = self.audiopaths_and_text[index]
        wav, mel = self._load_audio(wav_path)
        text_seq = torch.LongTensor(text_to_sequence(text, self.text_cleaners))
        return text_seq, mel, wav


class TextAudioCollate:
    """Pads text, mel, and waveform sequences."""

    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        # Sort descending by mel length
        batch.sort(key=lambda x: x[1].shape[1], reverse=True)

        # Pad text
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        max_input_len = input_lengths.max().item()
        text_padded = torch.full((len(batch), max_input_len), self.pad_value, dtype=torch.long)
        for i, x in enumerate(batch):
            text_padded[i, :len(x[0])] = x[0]

        # Pad mel
        spec_lengths = torch.LongTensor([x[1].shape[1] for x in batch])
        n_mel = batch[0][1].shape[0]
        max_spec_len = spec_lengths.max().item()
        spec_padded = torch.zeros((len(batch), n_mel, max_spec_len))
        for i, x in enumerate(batch):
            spec_padded[i, :, :x[1].shape[1]] = x[1]

        # Pad audio
        wav_lengths = torch.LongTensor([len(x[2]) for x in batch])
        max_wav_len = wav_lengths.max().item()
        wav_padded = torch.zeros((len(batch), max_wav_len))
        for i, x in enumerate(batch):
            wav_padded[i, :len(x[2])] = x[2]

        return text_padded, input_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths

import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio

import commons  
from commons import spectrogram_torch
from text import text_to_sequence


class TextAudioLoader(torch.utils.data.Dataset):
    """
    Loads text and corresponding mel-spectrogram or waveform paths.
    Expected filelist format: <audio_path>|<text>
    """

    def __init__(self, filelist_path, hps):
        self.filelist = self.load_filelist(filelist_path)
        self.max_wav_value = hps["data"]["max_wav_value"]
        self.sampling_rate = hps["data"]["sampling_rate"]
        self.filter_length = hps["data"]["filter_length"]
        self.hop_length = hps["data"]["hop_length"]
        self.win_length = hps["data"]["win_length"]
        self.n_mel_channels = hps["data"]["n_mel_channels"]
        self.text_cleaners = hps["data"]["text_cleaners"]
        self.add_blank = hps["data"]["add_blank"]
        self.n_speakers = hps["data"]["n_speakers"]
        self._n_symbols = 300  # default fallback

    def load_filelist(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if len(line.strip()) > 0]

    def get_audio_text_pair(self, line):
        parts = line.strip().split("|")
        if len(parts) < 2:
            raise ValueError(f"Invalid line in filelist: {line}")

        audio_path, text = parts[0], parts[1]
        text_seq = self.get_text(text)
        mel = self.get_mel(audio_path)
        return text_seq, mel

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)  
        text_tensor = torch.LongTensor(text_norm)
        return text_tensor

    def get_mel(self, filename):
        # Load and normalize waveform
        audio, sr = torchaudio.load(filename)
        if sr != self.sampling_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)
        audio = audio / self.max_wav_value
        audio = audio.mean(dim=0, keepdim=True)  # mono
        audio = audio.clamp(-1, 1)

        # Compute spectrogram
        spec = spectrogram_torch(
            audio,
            self.filter_length,
            self.sampling_rate,
            self.hop_length,
            self.win_length,
            center=False
        )
        spec = torch.squeeze(spec, 0)
        return spec

    def __getitem__(self, index):
        line = self.filelist[index]
        text, mel = self.get_audio_text_pair(line)
        return text, mel

    def __len__(self):
        return len(self.filelist)

    # Property for train.py compatibility
    @property
    def n_symbols(self):
        return self._n_symbols

    def get_text_cleaner_symbols(self):
        # Deprecated (kept for compatibility)
        return self._n_symbols


class TextAudioCollate:
    """Zero-pads model inputs and targets for batched training."""

    def __init__(self):
        pass

    def __call__(self, batch):
        # batch: list of (text, mel)
        _, mel = batch[0]
        n_mel_channels = mel.size(0)

        # Sort descending by text length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0,
            descending=True
        )

        max_input_len = input_lengths[0]
        max_output_len = max([x[1].size(1) for x in batch])

        # Prepare padded tensors
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        mel_padded = torch.FloatTensor(len(batch), n_mel_channels, max_output_len)
        mel_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        # Fill tensors
        for i in range(len(ids_sorted_decreasing)):
            text, mel = batch[ids_sorted_decreasing[i]]
            text_padded[i, :text.size(0)] = text
            mel_padded[i, :, :mel.size(1)] = mel
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, output_lengths

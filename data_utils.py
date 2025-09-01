import os
import random
import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read
import librosa
import soundfile as sf

from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn

import commons
import utils
from text import text_to_sequence, get_symbols


class TextAudioLoader(torch.utils.data.Dataset):
    """
    1) Loads audio, text pairs
    2) Normalizes text and converts them to sequences of integers
    3) Computes spectrograms from audio files
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = utils.load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        # Load symbols
        self.symbols = get_symbols(self.text_cleaners)
        print(f"[TextAudioLoader] Loaded {len(self.symbols)} symbols with cleaners={self.text_cleaners}")

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """Filter dataset by text length and check audio files."""
        audiopaths_and_text_new = []
        lengths = []
        skipped = 0

        for audiopath, text in self.audiopaths_and_text:
            text_len = len(text)
            if self.min_text_len <= text_len <= self.max_text_len:
                if os.path.exists(audiopath):
                    audiopaths_and_text_new.append([audiopath, text])
                    try:
                        file_size = os.path.getsize(audiopath)
                        estimated_length = file_size // (2 * self.hop_length)
                        lengths.append(max(estimated_length, 10))
                    except:
                        lengths.append(100)
                else:
                    print(f"[Warning] Missing audio: {audiopath}")
                    skipped += 1
            else:
                skipped += 1

        print(f"[Filter] Kept {len(audiopaths_and_text_new)} | Skipped {skipped}")
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        """Load and normalize audio with multiple backends."""
        audio, sampling_rate = None, None
        try:
            audio, sampling_rate = sf.read(filename)
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        except Exception as e1:
            try:
                audio, sampling_rate = librosa.load(filename, sr=None, mono=True)
                audio = audio.astype(np.float32)
            except Exception as e2:
                try:
                    sampling_rate, audio = read(filename)
                    audio = audio.astype(np.float32)
                    if audio.dtype == np.int16:
                        audio /= 32768.0
                    elif audio.dtype == np.int32:
                        audio /= 2147483648.0
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed {filename}. sf: {e1}, librosa: {e2}, scipy: {e3}"
                    )

        audio = torch.FloatTensor(audio)

        # Resample if mismatch
        if sampling_rate != self.sampling_rate:
            print(f"[Resample] {filename}: {sampling_rate} -> {self.sampling_rate}")
            audio = torch.FloatTensor(
                librosa.resample(audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate)
            )

        # Normalize
        if audio.abs().max() > 1.0:
            audio /= audio.abs().max()

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        # Spectrogram caching
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, map_location="cpu")
            except:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
                try:
                    torch.save(spec, spec_filename)
                except:
                    pass
        else:
            spec = spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                center=False,
            )
            try:
                torch.save(spec, spec_filename)
            except:
                pass

        return torch.squeeze(spec, 0), audio_norm

    def get_text(self, text):
        """Convert text string into tensor of IDs."""
        try:
            text_norm = text_to_sequence(text, self.text_cleaners)
            if self.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            return torch.LongTensor(text_norm)
        except Exception as e:
            print(f"[Error] text '{text}' -> {e}")
            return torch.LongTensor([0])

    def __getitem__(self, index):
        try:
            return self.get_audio_text_pair(self.audiopaths_and_text[index])
        except Exception as e:
            print(f"[Error] sample {index}: {e}")
            return (
                torch.LongTensor([0]),
                torch.zeros(513, 10),
                torch.zeros(1, 1000),
            )

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """Collate function to zero-pad batch samples."""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        batch = [b for b in batch if b is not None and len(b) == 3]
        if len(batch) == 0:
            raise ValueError("Empty batch")

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0,
            descending=True,
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        spec_padded = torch.zeros(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.zeros(len(batch), 1, max_wav_len)

        text_lengths, spec_lengths, wav_lengths = [], [], []

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths.append(text.size(0))

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths.append(spec.size(1))

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths.append(wav.size(1))

        text_lengths = torch.LongTensor(text_lengths)
        spec_lengths = torch.LongTensor(spec_lengths)
        wav_lengths = torch.LongTensor(wav_lengths)

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                ids_sorted_decreasing,
            )
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """Torch STFT wrapper compatible with PyTorch 2.2.2"""
    global hann_window
    dtype_device = f"{y.dtype}_{y.device}"
    wnsize_dtype_device = f"{win_size}_{dtype_device}"
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)

    try:
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.abs(spec)
    except:
        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    return spec


hann_window = {}

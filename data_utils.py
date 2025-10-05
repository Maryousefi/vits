# data_utils.py
import os
import math
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.sampler import Sampler

import librosa

# Import text utilities (your text package)
# text_to_sequence should accept (text, cleaner_names)
try:
    from text import text_to_sequence
except Exception:
    # defensive fallback
    def text_to_sequence(text, cleaners):
        return [ord(c) % 256 for c in text]


class TextAudioLoader(Dataset):
    """
    Dataset that yields (text_seq_tensor, text_len, mel_spec, mel_len, waveform, wav_len)

    filelist format: /abs/or/rel/path/to/wav.wav|transcription
    data_hparams is the hps.data object (or a dict-like object) that contains:
      - sampling_rate
      - n_mel_channels
      - filter_length (n_fft)
      - hop_length
      - win_length
      - mel_fmin (optional)
      - mel_fmax (optional)
      - text_cleaners (list or string)
    """

    def __init__(self, filelist_path: str, data_hparams):
        assert os.path.isfile(filelist_path), f"Filelist not found: {filelist_path}"
        self.audiopaths_and_text = self._load_filelist(filelist_path)

        # read mel/audio params
        self.sampling_rate = getattr(data_hparams, "sampling_rate", 22050)
        self.n_mel_channels = getattr(data_hparams, "n_mel_channels", 80)
        self.filter_length = getattr(data_hparams, "filter_length", 1024)
        self.hop_length = getattr(data_hparams, "hop_length", 256)
        self.win_length = getattr(data_hparams, "win_length", self.filter_length)
        self.mel_fmin = getattr(data_hparams, "mel_fmin", 0.0)
        self.mel_fmax = getattr(data_hparams, "mel_fmax", None)
        self.text_cleaners = getattr(data_hparams, "text_cleaners", ["persian_cleaners"])

    def _load_filelist(self, filelist_path: str) -> List[Tuple[str, str]]:
        pairs = []
        with open(filelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # split only on first '|' (text may contain '|')
                parts = line.split("|", 1)
                if len(parts) < 2:
                    continue
                wav_path = parts[0].strip()
                text = parts[1].strip()
                pairs.append((wav_path, text))
        return pairs

    def __len__(self):
        return len(self.audiopaths_and_text)

    def _load_wav(self, filename: str) -> Tuple[np.ndarray, int]:
        # Use librosa to load wave
        wav, sr = librosa.load(filename, sr=self.sampling_rate)
        # Guarantee 1-d float32 numpy array
        wav = wav.astype(np.float32)
        return wav, sr

    def _wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        # compute power spectrogram -> mel
        # librosa.feature.melspectrogram expects power by default (magnitude^2)
        S = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sampling_rate,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            power=2.0,
        )
        # Convert power to log scale (dB). Use top_db similar to librosa.power_to_db default
        S_db = librosa.power_to_db(S, ref=np.max)
        # Convert to float32
        return S_db.astype(np.float32)

    def __getitem__(self, index: int):
        """
        Returns:
            text_tensor (LongTensor) - shape (T_text,)
            text_length (LongTensor scalar)
            spec (FloatTensor) - shape (n_mel_channels, T_spec)
            spec_length (LongTensor scalar) - number of frames (T_spec)
            waveform (FloatTensor) - shape (N_samples,)
            waveform_length (LongTensor scalar)
        """
        wav_path, text = self.audiopaths_and_text[index]

        # load waveform
        wav, sr = self._load_wav(wav_path)
        wav_len = wav.shape[0]

        # compute mel-spectrogram
        mel = self._wav_to_mel(wav)  # shape (n_mel_channels, T)
        mel_len = mel.shape[1]

        # convert text -> sequence of symbol ids using text_to_sequence
        text_seq = text_to_sequence(text, self.text_cleaners)
        text_tensor = torch.LongTensor(text_seq)
        text_len = torch.LongTensor([text_tensor.size(0)]).long().squeeze(0)

        # wrap waveform and mel into torch tensors
        wav_tensor = torch.FloatTensor(wav)
        mel_tensor = torch.FloatTensor(mel)

        # lengths as plain ints (collate will convert to tensors)
        return text_tensor, int(text_tensor.size(0)), mel_tensor, int(mel_len), wav_tensor, int(wav_len)


class TextAudioCollate:
    """
    Collate that pads:
      - text sequences (pad id = 0)
      - mel spectrograms (pad 0.0)
      - waveforms (pad 0.0)

    Returns tensors in the order expected by train.py:
      (x_padded, x_lengths, spec_padded, spec_lengths, y_padded, y_lengths)
    """

    def __init__(self, pad_value_text: int = 0):
        self.pad_value_text = pad_value_text

    def __call__(self, batch):
        # batch is list of tuples returned by dataset.__getitem__
        # each element: (text_tensor, text_len, mel_tensor, mel_len, wav_tensor, wav_len)
        # sort by text length or mel length (descending) for efficiency
        batch.sort(key=lambda x: x[1], reverse=True)  # sort by text_len

        # texts
        input_lengths = torch.LongTensor([b[1] for b in batch])
        max_input_len = int(input_lengths.max().item())
        text_padded = torch.full((len(batch), max_input_len), fill_value=self.pad_value_text, dtype=torch.long)
        for i, b in enumerate(batch):
            t = b[0]
            text_padded[i, : t.size(0)] = t

        # mels
        spec_lengths = torch.LongTensor([b[3] for b in batch])
        max_spec_len = int(spec_lengths.max().item())
        n_mel = batch[0][2].size(0)
        spec_padded = torch.zeros((len(batch), n_mel, max_spec_len), dtype=torch.float)
        for i, b in enumerate(batch):
            mel = b[2]
            spec_padded[i, :, : mel.size(1)] = mel

        # waveforms
        wav_lengths = torch.LongTensor([b[5] for b in batch])
        max_wav_len = int(wav_lengths.max().item())
        wav_padded = torch.zeros((len(batch), max_wav_len), dtype=torch.float)
        for i, b in enumerate(batch):
            wav = b[4]
            wav_padded[i, : wav.size(0)] = wav

        return text_padded, input_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


# -----------------------------
# DistributedBucketSampler (fixed)
# -----------------------------
class DistributedBucketSampler(Sampler):
    """
    Sampler that groups samples into buckets by length, pads each bucket to multiple
    of (batch_size * world_size) and yields LISTS OF INDICES (batches) when iterated.

    This is intended to be used as `batch_sampler=` in DataLoader.

    Args:
        dataset: dataset to sample from (expected to yield tuples where one of the elements is length)
                 This implementation expects dataset[i] to be (text_tensor, text_len, mel_tensor, mel_len, wav_tensor, wav_len)
        batch_size: number of items per batch
        boundaries: length boundaries for bucketing (list of ints)
        num_replicas: world size (if None will use dist.get_world_size())
        rank: current rank (if None will use dist.get_rank())
        shuffle: shuffle each epoch
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package required to auto-detect world size")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Distributed package required to auto-detect rank")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.boundaries = list(boundaries)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = shuffle

        # Build buckets based on mel length (index 3 in dataset return)
        self.buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for idx in range(len(dataset)):
            item = dataset[idx]
            # item expected (text_tensor, text_len, mel_tensor, mel_len, wav_tensor, wav_len)
            if isinstance(item, tuple) and len(item) >= 4:
                length = int(item[3])
            else:
                # fallback: try dataset.get length as len(dataset[idx][2].shape[1]) etc.
                try:
                    length = int(item[2].shape[1])
                except Exception:
                    length = 0
            bucket_id = self._get_bucket(length)
            self.buckets[bucket_id].append(idx)

        # compute total size (padded)
        self.total_size = 0
        for bucket in self.buckets:
            bucket_len = len(bucket)
            if bucket_len == 0:
                continue
            # each bucket must be padded to multiple of (batch_size * num_replicas)
            per_rank = self.batch_size * self.num_replicas
            bucket_size = int(math.ceil(bucket_len / per_rank)) * per_rank
            self.total_size += bucket_size

    def _get_bucket(self, length: int) -> int:
        for i, b in enumerate(self.boundaries):
            if length <= b:
                return i
        return len(self.boundaries)

    def __iter__(self):
        # aggregate indices per bucket
        all_indices = []
        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(torch.randint(0, 10**9, (1,)).item())
                perm = torch.randperm(len(bucket), generator=g).tolist()
                indices_bucket = [bucket[i] for i in perm]
            else:
                indices_bucket = list(bucket)

            # pad bucket to multiple of (batch_size * num_replicas)
            per_rank = self.batch_size * self.num_replicas
            if len(indices_bucket) % per_rank != 0:
                needed = per_rank - (len(indices_bucket) % per_rank)
                # repeat from start to pad
                indices_bucket += indices_bucket[:needed]

            all_indices.extend(indices_bucket)

        # now take only indices for this rank
        indices_rank = all_indices[self.rank::self.num_replicas]

        # group into batches (lists) of size batch_size
        batches = [indices_rank[i: i + self.batch_size] for i in range(0, len(indices_rank), self.batch_size)]

        return iter(batches)

    def __len__(self):
        # number of batches for this rank per epoch
        # total_size is padded total samples (across all ranks), so per-rank samples:
        per_rank_samples = self.total_size // self.num_replicas
        if per_rank_samples == 0:
            return 0
        return per_rank_samples // self.batch_size


# Export names
__all__ = [
    "TextAudioLoader",
    "TextAudioCollate",
    "DistributedBucketSampler",
]

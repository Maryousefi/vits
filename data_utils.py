import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa

from mel_processing import spectrogram_torch
from text import text_to_sequence


class TextAudioLoader(Dataset):
    """
    Loads text and audio pairs from filelists.
    Each line in filelist: wav_path|text
    """

    def __init__(self, filelist_path, data_config):
        self.audiopaths_and_text = self._load_filelist(filelist_path)
        self.text_cleaners = data_config.text_cleaners
        self.sampling_rate = data_config.sampling_rate
        self.filter_length = data_config.filter_length
        self.hop_length = data_config.hop_length
        self.win_length = data_config.win_length

    def _load_filelist(self, filelist_path):
        with open(filelist_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        audiopaths_and_text = []
        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            audiopaths_and_text.append((parts[0], parts[1]))
        return audiopaths_and_text

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text

        # Load audio
        audio, sr = librosa.load(audiopath, sr=self.sampling_rate)
        audio = torch.FloatTensor(audio)

        # Compute spectrogram (513 channels for 1024 FFT)
        spec = spectrogram_torch(
            audio.unsqueeze(0),
            n_fft=self.filter_length,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_length,
            win_size=self.win_length,
            center=False,
        ).squeeze(0)

        # Process text
        text_seq = text_to_sequence(text, self.text_cleaners)
        text_tensor = torch.LongTensor(text_seq)

        return text_tensor, spec, audio

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:
    """
    Collate function to batch text, spectrogram, and audio samples with padding.
    """

    def __call__(self, batch):
        # Sort by spectrogram length (descending)
        batch.sort(key=lambda x: x[1].shape[1], reverse=True)

        # Text
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        max_input_len = input_lengths.max().item()
        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i in range(len(batch)):
            text_padded[i, :input_lengths[i]] = batch[i][0]

        # Spectrogram
        spec_lengths = torch.LongTensor([x[1].shape[1] for x in batch])
        n_mel_channels = batch[0][1].shape[0]
        max_spec_len = spec_lengths.max().item()
        spec_padded = torch.zeros(len(batch), n_mel_channels, max_spec_len)
        for i in range(len(batch)):
            spec_padded[i, :, :spec_lengths[i]] = batch[i][1]

        # Audio
        wav_lengths = torch.LongTensor([x[2].size(0) for x in batch])
        max_wav_len = wav_lengths.max().item()
        wav_padded = torch.zeros(len(batch), max_wav_len)
        for i in range(len(batch)):
            wav_padded[i, :wav_lengths[i]] = batch[i][2]

        return (
            text_padded,
            input_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
        )


# -----------------------------
# DistributedBucketSampler
# -----------------------------
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class DistributedBucketSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset for distributed training.
    It groups sequences into buckets based on their length, then samples within each bucket.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        # Assign each sample to a bucket based on spectrogram length
        self.buckets = [[] for _ in range(len(boundaries) + 1)]
        for i in range(len(dataset)):
            _, spec, _ = dataset[i]
            length = spec.shape[1]
            idx_bucket = self._get_bucket(length)
            self.buckets[idx_bucket].append(i)

        self.total_size = 0
        for bucket in self.buckets:
            bucket_size = int(math.ceil(len(bucket) / (self.batch_size * self.num_replicas))) \
                          * self.batch_size * self.num_replicas
            self.total_size += bucket_size

    def _get_bucket(self, length):
        for i, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return i
        return len(self.boundaries)

    def __iter__(self):
        indices = []
        for bucket in self.buckets:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(torch.randint(0, 1e9, (1,)).item())
                indices_bucket = torch.randperm(len(bucket), generator=g).tolist()
            else:
                indices_bucket = list(range(len(bucket)))

            while len(indices_bucket) % (self.batch_size * self.num_replicas) != 0:
                indices_bucket += indices_bucket[: (self.batch_size * self.num_replicas) - len(indices_bucket)]

            indices.extend([bucket[i] for i in indices_bucket])

        indices_rank = indices[self.rank::self.num_replicas]
        return iter(indices_rank)

    def __len__(self):
        return self.total_size // (self.num_replicas * self.batch_size)

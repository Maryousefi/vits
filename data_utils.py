import os
import random
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import librosa


# ==============================================================
# Text-Audio Dataset
# ==============================================================

class TextAudioLoader(Dataset):
    """
    Loads text and audio pairs from filelists.
    Each line of the filelist should be in format:
        wav_path|text
    """

    def __init__(self, filelist_path, hps_data):
        """
        Args:
            filelist_path (str): path to filelist (e.g. filelists/fa_single/train.txt)
            hps_data: hyperparameter data config (should include text_cleaners, sampling_rate, etc.)
        """
        self.audiopaths_and_text = self._load_filelist(filelist_path)
        self.text_cleaners = getattr(hps_data, "text_cleaners", ["persian_cleaners"])
        self.sampling_rate = getattr(hps_data, "sampling_rate", 22050)

    def _load_filelist(self, filelist_path):
        with open(filelist_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        pairs = []
        for line in lines:
            parts = line.split("|")
            if len(parts) < 2:
                continue
            wav_path, text = parts[0], parts[1]
            # Expand relative paths to absolute
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(os.getcwd(), wav_path)
            pairs.append((wav_path, text))
        return pairs

    def get_audio_text_pair(self, audiopath_and_text):
        audiopath, text = audiopath_and_text
        audio = self.get_audio(audiopath)
        from text import text_to_sequence
        text_seq = text_to_sequence(text, self.text_cleaners)
        return torch.LongTensor(text_seq), audio, len(audio)

    def get_audio(self, filename):
        audio, sr = librosa.load(filename, sr=self.sampling_rate)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = torch.FloatTensor(audio)
        return audio

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


# ==============================================================
# Collate Function
# ==============================================================

class TextAudioCollate:
    """Collate function to batch text and audio samples with padding."""

    def __call__(self, batch):
        # Sort by audio length (descending)
        batch.sort(key=lambda x: x[2], reverse=True)

        # --- Text ---
        input_lengths = torch.LongTensor([len(x[0]) for x in batch])
        max_input_len = input_lengths.max().item()
        text_padded = torch.zeros(len(batch), max_input_len, dtype=torch.long)
        for i in range(len(batch)):
            text_padded[i, :input_lengths[i]] = batch[i][0]

        # --- Audio ---
        audio_lengths = torch.LongTensor([len(x[1]) for x in batch])
        max_target_len = audio_lengths.max().item()
        audio_padded = torch.zeros(len(batch), max_target_len)
        for i in range(len(batch)):
            audio_padded[i, :audio_lengths[i]] = batch[i][1]

        return text_padded, input_lengths, audio_padded, audio_lengths


# ==============================================================
# Distributed Bucket Sampler
# ==============================================================

class DistributedBucketSampler(torch.utils.data.Sampler):
    """
    Sampler for distributed training with length-based bucketing.

    Groups samples into buckets by sequence length, shuffles within each bucket,
    and ensures each replica (GPU) gets the same number of samples.

    Args:
        dataset: dataset to sample from
        batch_size: number of samples per batch per GPU
        boundaries: list of length boundaries
        num_replicas: number of GPUs (world size)
        rank: current GPU id
        shuffle: whether to shuffle samples each epoch
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

        # Precompute bucket membership
        self.buckets = [[] for _ in range(len(boundaries) + 1)]
        for idx, (_, _, length) in enumerate(dataset):
            bucket_idx = self._get_bucket(length)
            self.buckets[bucket_idx].append(idx)

        # Compute total size (padding to make divisible)
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

    def set_epoch(self, epoch):
        """Sets the epoch for deterministic shuffling."""
        self.epoch = int(epoch)

    def __iter__(self):
        indices = []
        g = torch.Generator()
        g.manual_seed(self.epoch + 1234)

        for bucket in self.buckets:
            if len(bucket) == 0:
                continue
            if self.shuffle:
                idxs = torch.randperm(len(bucket), generator=g).tolist()
            else:
                idxs = list(range(len(bucket)))

            # pad to make it evenly divisible
            while len(idxs) % (self.batch_size * self.num_replicas) != 0:
                idxs += idxs[: (self.batch_size * self.num_replicas) - len(idxs)]

            indices.extend([bucket[i] for i in idxs])

        # Subsample for this rank
        indices_rank = indices[self.rank::self.num_replicas]
        return iter(indices_rank)

    def __len__(self):
        return self.total_size // (self.num_replicas * self.batch_size)

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
    1) loads audio, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = utils.load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        # Initialize symbols for this dataset
        self.symbols = get_symbols(self.text_cleaners)
        print(f"TextAudioLoader initialized with {len(self.symbols)} symbols")
        print(f"Text cleaners: {self.text_cleaners}")

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        audiopaths_and_text_new = []
        lengths = []
        skipped = 0
        
        for audiopath, text in self.audiopaths_and_text:
            text_len = len(text)
            if self.min_text_len <= text_len <= self.max_text_len:
                # Check if audio file exists
                if os.path.exists(audiopath):
                    audiopaths_and_text_new.append([audiopath, text])
                    try:
                        # Estimate length from file size
                        file_size = os.path.getsize(audiopath)
                        estimated_length = file_size // (2 * self.hop_length)
                        lengths.append(max(estimated_length, 10))  # Minimum length
                    except:
                        lengths.append(100)  # Default fallback
                else:
                    print(f"Warning: Audio file not found: {audiopath}")
                    skipped += 1
            else:
                skipped += 1
                
        print(f"Filtered dataset: {len(audiopaths_and_text_new)} samples kept, {skipped} skipped")
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        """Load and process audio file with multiple fallbacks."""
        audio = None
        sampling_rate = None
        
        # Try multiple loading methods
        try:
            # Method 1: soundfile (best for quality and format support)
            audio, sampling_rate = sf.read(filename)
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:  # Convert stereo to mono
                audio = audio.mean(axis=1)
        except Exception as e1:
            try:
                # Method 2: librosa (good fallback)
                audio, sampling_rate = librosa.load(filename, sr=None, mono=True)
                audio = audio.astype(np.float32)
            except Exception as e2:
                try:
                    # Method 3: scipy (for basic WAV files)
                    sampling_rate, audio = read(filename)
                    audio = audio.astype(np.float32)
                    # Handle integer audio
                    if audio.dtype == np.int16:
                        audio = audio / 32768.0
                    elif audio.dtype == np.int32:
                        audio = audio / 2147483648.0
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                except Exception as e3:
                    raise RuntimeError(f"Failed to load {filename}. Tried sf.read: {e1}, librosa.load: {e2}, scipy.read: {e3}")

        # Convert to torch tensor
        audio = torch.FloatTensor(audio)
        
        # Resample if necessary
        if sampling_rate != self.sampling_rate:
            print(f"Resampling {filename} from {sampling_rate} to {self.sampling_rate}")
            audio = torch.FloatTensor(librosa.resample(
                audio.numpy(), orig_sr=sampling_rate, target_sr=self.sampling_rate
            ))

        # Normalize audio amplitude
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
            
        # Apply max_wav_value normalization
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)  # Add channel dimension
        
        # Handle spectrogram caching
        spec_filename = filename.replace(".wav", ".spec.pt")
        
        if os.path.exists(spec_filename):
            try:
                spec = torch.load(spec_filename, map_location='cpu', weights_only=True)
            except:
                # Recalculate if loading fails
                spec = spectrogram_torch(audio_norm, 
                    self.filter_length, self.sampling_rate, 
                    self.hop_length, self.win_length,
                    center=False)
                try:
                    torch.save(spec, spec_filename)
                except:
                    pass  # Ignore save errors
        else:
            spec = spectrogram_torch(audio_norm, 
                self.filter_length, self.sampling_rate, 
                self.hop_length, self.win_length,
                center=False)
            try:
                torch.save(spec, spec_filename)
            except:
                pass  # Ignore save errors
            
        spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def get_text(self, text):
        """Convert text to sequence."""
        try:
            if self.cleaned_text:
                text_norm = text_to_sequence(text, self.text_cleaners)
            else:
                text_norm = text_to_sequence(text, self.text_cleaners)
                
            if self.add_blank:
                text_norm = commons.intersperse(text_norm, 0)
            text_norm = torch.LongTensor(text_norm)
            return text_norm
        except Exception as e:
            print(f"Error processing text '{text}': {e}")
            # Return minimal sequence if processing fails
            return torch.LongTensor([0])

    def __getitem__(self, index):
        try:
            return self.get_audio_text_pair(self.audiopaths_and_text[index])
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            # Return a dummy sample to prevent training crash
            dummy_text = torch.LongTensor([0])
            dummy_spec = torch.zeros(513, 10)  # Minimal spec
            dummy_wav = torch.zeros(1, 1000)   # Minimal wav
            return dummy_text, dummy_spec, dummy_wav

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and audio"""
        # Filter out None values and ensure we have valid samples
        batch = [item for item in batch if item is not None and len(item) == 3]
        
        if len(batch) == 0:
            raise ValueError("Empty batch after filtering")
            
        # Right zero-pad all sequences to max length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
  
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
  
        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
  
            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[ids_bucket[j*self.batch_size + k]] for k in range(self.batch_size)]
                batches.append(batch)
  
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
  
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
  
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """
    Create spectrogram from audio - Compatible with PyTorch 2.2.2
    """
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # PyTorch 2.2.2 compatible STFT
    try:
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                         window=hann_window[wnsize_dtype_device],
                         center=center, pad_mode='reflect', normalized=False, 
                         onesided=True, return_complex=True)
        spec = torch.abs(spec)
    except:
        # Fallback to older format
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                         window=hann_window[wnsize_dtype_device],
                         center=center, pad_mode='reflect', normalized=False, 
                         onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        
    return spec

# Global variable for hann window caching
hann_window = {}

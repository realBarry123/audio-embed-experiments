import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import mirdata
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import resample
from datasets import load_dataset, Audio
import io
import soundfile as sf

import time


class MIRDataset(Dataset):
    """
    Adapted from https://mirdata.readthedocs.io/en/stable/source/tutorial.html 06-09-2026
    """
    def __init__(self, dataset_name: str, chunk_duration=5.0, include_melody=False):
        self.dataset_name = dataset_name
        self.include_melody = include_melody
        self.loader = mirdata.initialize(dataset_name)
        self.loader.download()
        self.loader.validate()
        self.EMPTY_NOTE_FREQ = 2.0 ** (-25/12) * 440

        self.chunk_duration = chunk_duration
        self.chunk_index = []

        for tid in self.loader.track_ids:
            audio_signal, fs = self.loader.track(tid).audio_stereo
            chunk_size = int(round(self.chunk_duration * fs))
            n_chunks = int(np.floor(len(audio_signal[0]) / chunk_size)) # discard last chunk
            for i in range(n_chunks):
                start = i * chunk_size
                self.chunk_index.append((tid, start, chunk_size))

    def __len__(self) -> int:
        return len(self.chunk_index)

    def __getitem__(self, item: int) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        track_id, start, chunk_size = self.chunk_index[item]
        track = self.loader.track(track_id)

        audio_signal, fs = track.audio_stereo
        audio_chunk = audio_signal[:, start: start+chunk_size]

        if self.include_melody:
            melody = track.melody
            assert fs == 44100
            assert melody.frequency_unit in ["hz", "Hz"]
            assert melody.time_unit == "s"
            # diffs = np.diff(np.insert(melody.times, 0, 0.0)) * fs
            # melody_signal = track.melody.frequencies.repeat(diffs.astype(np.int32)+1)
            hop_samples = int(round((melody.times[1] - melody.times[0]) * fs))
            melody_signal = melody.frequencies.repeat(hop_samples)
            melody_chunk = melody_signal[start: start+chunk_size]

            mask = (np.abs(melody_chunk) <= 1e-3) | np.isnan(melody_chunk)
            safe = np.where(mask, 1, melody_chunk)
            melody_chunk = np.where(mask, 0, (np.log2(safe/440) * 12).astype(np.int32) + 13) 
            melody_chunk = np.clip(melody_chunk, 0, 25) # band-aid solution for edge cases
            # print(np.min(melody_chunk), np.max(melody_chunk))
            
            melody_one_hot = np.zeros((melody_chunk.size, 26))
            melody_one_hot[np.arange(melody_chunk.size), melody_chunk] = 1
            return audio_chunk.astype(np.float32), melody_one_hot
        
        return audio_chunk.astype(np.float32) # shape: (C=2, T=chunk_size)

    def get_loaders(self, valid_split, batch_size, seed=0) -> tuple[DataLoader, DataLoader]:
        """
        Adapted from https://stackoverflow.com/a/50544887 2026-06-10
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_split * dataset_size))

        np.random.seed(seed)
        np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(indices[split:])
        valid_sampler = SubsetRandomSampler(indices[:split])
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(self, batch_size=batch_size, sampler=valid_sampler)

        return train_loader, valid_loader
    

class AudioSetDataset(Dataset):
    """
    Dataset adapter for AudioSet dataset
    """
    def __init__(self, chunk_duration=5.0):
        self.dataset = load_dataset("agkphysics/AudioSet", "balanced", streaming=False, split="train")
        self.dataset = self.dataset.cast_column("audio", Audio(decode=False))
        self.chunk_duration = chunk_duration
        self.fs = 44100

        self.chunk_index = []
        
        # Pre-compute chunk indices
        for idx, sample in enumerate(self.dataset):
            try:
                audio_bytes = sample["audio"]["bytes"]
                info = sf.info(io.BytesIO(audio_bytes))
                n_frames_resampled = int(info.frames * self.fs / info.samplerate)
                chunk_size = int(round(self.chunk_duration * self.fs))
                n_chunks = n_frames_resampled // chunk_size
                for i in range(n_chunks):
                    self.chunk_index.append((idx, i * chunk_size, chunk_size))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

    def __len__(self) -> int:
        return len(self.chunk_index)

    def __getitem__(self, item: int) -> torch.Tensor:
        dataset_idx, start, chunk_size = self.chunk_index[item]
        sample = self.dataset[dataset_idx]
        
        audio_bytes = sample["audio"]["bytes"]
        audio_array, fs = sf.read(io.BytesIO(audio_bytes))
        audio_array = torch.tensor(audio_array.astype(np.float32)).to("cuda")
        if fs != self.fs:
            audio_array = resample(audio_array, orig_freq=fs, new_freq=self.fs)
    
        chunk = audio_array[start:start + chunk_size]
        if len(list(chunk.shape)) == 1: # mono
            chunk = repeat(chunk, "t -> 2 t")
        elif chunk.shape[1] == 1: # sneaky mono
            chunk = repeat(chunk.squeeze(1), "t -> 2 t")
        else: 
            chunk = rearrange(chunk[:, :2], "t c -> c t")

        return chunk

    def get_loaders(self, valid_split: float = 0.2, batch_size: int = 32, seed: int = 0) -> tuple[DataLoader, DataLoader]:
        """Create train/validation data loaders"""
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_split * dataset_size))
        
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_sampler = SubsetRandomSampler(indices[split:])
        valid_sampler = SubsetRandomSampler(indices[:split])
        
        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        valid_loader = DataLoader(self, batch_size=batch_size, sampler=valid_sampler)
        
        return train_loader, valid_loader

if __name__ == "__main__":

    dataset = MIRDataset("orchset", include_melody=True)
    train_loader, valid_loader = dataset.get_loaders(
        valid_split=0.2, 
        batch_size=1
    )
    total_note_range = [0, 0]
    for x, y in tqdm(train_loader):
        # x = rearrange(x, "1 channels frames -> frames channels")
        note_range = (y.min(), y.max())
        if note_range[0].item() < total_note_range[0]:
            total_note_range[0] = note_range[0].item()
        elif note_range[1].item() > total_note_range[1]:
            total_note_range[1] = note_range[1].item()
        # sf.write(f"test_{dataset.dataset_name}.wav", x, 44100)
    print("Orchset note range:", total_note_range)
    exit()

    dataset = AudioSetDataset()
    train_loader, valid_loader = dataset.get_loaders(valid_split=0.2, batch_size=32)
    for x in valid_loader:
        x = rearrange(x, "1 t c -> t c")
        sf.write(f"test_audioset.wav", x, 44100)
        break
    
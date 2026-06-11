import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import mirdata
from torch.utils.data import Dataset, DataLoader


class MIRDataset(Dataset):
    """
    Adapted from https://mirdata.readthedocs.io/en/stable/source/tutorial.html 06-09-2026
    """
    def __init__(self, dataset_name: str, chunk_duration=5.0):
        self.loader = mirdata.initialize(dataset_name)
        self.loader.download()
        self.loader.validate()

        self.chunk_duration = chunk_duration
        self.chunk_index = []

        for tid in self.loader.track_ids:
            audio_signal, fs = self.loader.track(tid).audio_mono
            chunk_size = int(round(self.chunk_duration * fs))
            n_chunks = int(np.floor(len(audio_signal) / chunk_size)) # discard last chunk
            for i in range(n_chunks):
                start = i * chunk_size
                self.chunk_index.append((tid, start, chunk_size))

    @staticmethod
    def pad(to_pad: np.ndarray, pad_size: int) -> np.ndarray:
        """Right-pads a 1D array to `pad_size`"""
        return np.pad(
            to_pad, 
            (0, pad_size - len(to_pad)), 
            mode="constant", 
            constant_values=0.0
        )

    def __len__(self):
        return len(self.loader.track_ids)

    def __getitem__(self, item: int) -> np.ndarray:
        track_id, start, chunk_size = self.chunk_index[item]
        track = self.loader.track(track_id)
        audio_signal, fs = track.audio_mono

        chunk = audio_signal[start:start + chunk_size]

        return chunk.astype(np.float32)

    def get_loaders(self, valid_split, batch_size, seed=0):
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

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import mirdata
from torch.utils.data import Dataset, DataLoader


class MIRDataset(Dataset):
    """
    Adapted from https://mirdata.readthedocs.io/en/stable/source/tutorial.html 06-09-2026
    """
    def __init__(self, dataset_name: str):

        # Initialize the loader, download if required, and validate
        self.loader = mirdata.initialize(dataset_name)
        self.loader.download()
        self.loader.validate()

        # Used for padding tensors
        self.longest_track = max(
            [len(self.loader.track(tid).audio_mono[0]) for tid in self.loader.track_ids]
        )

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

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        track_id = self.loader.track_ids[item]
        track = self.loader.track(track_id)

        audio_signal, fs = track.audio_mono

        audio_signal_padded = self.pad(audio_signal, self.longest_track)

        return audio_signal_padded.astype(np.float32)

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

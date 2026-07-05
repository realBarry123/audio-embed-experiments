import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import mirdata
from torch.utils.data import Dataset, DataLoader
import librosa
from datasets import load_dataset, Audio
import io
import soundfile as sf


class MIRDataset(Dataset):
    """
    Adapted from https://mirdata.readthedocs.io/en/stable/source/tutorial.html 06-09-2026
    """
    def __init__(self, dataset_name: str, chunk_duration=5.0):
        self.dataset_name = dataset_name
        self.loader = mirdata.initialize(dataset_name)
        self.loader.download()
        self.loader.validate()

        self.chunk_duration = chunk_duration
        self.chunk_index = []

        for tid in self.loader.track_ids:
            audio_signal, fs = self.loader.track(tid).audio_stereo
            chunk_size = int(round(self.chunk_duration * fs))
            n_chunks = int(np.floor(len(audio_signal[0]) / chunk_size)) # discard last chunk
            for i in range(n_chunks):
                start = i * chunk_size
                self.chunk_index.append((tid, start, chunk_size))

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, item: int) -> np.ndarray:
        track_id, start, chunk_size = self.chunk_index[item]
        track = self.loader.track(track_id)
        audio_signal, fs = track.audio_stereo

        chunk = audio_signal[:, start:start + chunk_size]

        return chunk.astype(np.float32) # shape: (C=2, T=chunk_size)

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
    

class AudioSetDataset(Dataset):
    """
    Dataset adapter for Hugging Face WavCaps dataset
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
                #print("sample:", sample)
                #audio_array = sample['audio']['array']
                #fs = sample['audio']['sampling_rate']
                audio_bytes = sample["audio"]["bytes"]
                audio_array, fs = sf.read(io.BytesIO(audio_bytes))
                
                if fs != self.fs:
                    audio_array = librosa.resample(audio_array, orig_sr=fs, target_sr=self.fs)
                
                chunk_size = int(round(self.chunk_duration * self.fs))
                n_chunks = int(np.floor(len(audio_array) / chunk_size))
                
                for i in range(n_chunks):
                    start = i * chunk_size
                    self.chunk_index.append((idx, start, chunk_size))
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, item: int) -> dict:
        dataset_idx, start, chunk_size = self.chunk_index[item]
        sample = self.dataset[dataset_idx]
        
        #audio_array = sample['audio']['array']
        #fs = sample['audio']['sampling_rate']
        audio_bytes = sample["audio"]["bytes"]
        audio_array, fs = sf.read(io.BytesIO(audio_bytes))

        if fs != self.fs:
            audio_array = librosa.resample(audio_array, orig_sr=fs, target_sr=self.fs)
    
        chunk = audio_array[start:start + chunk_size]
        
        return torch.tensor(chunk.astype(np.float32))

    def get_loaders(self, valid_split: float = 0.2, batch_size: int = 32, seed: int = 0):
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
    import soundfile as sf
    from einops import rearrange

    dataset = MIRDataset("orchset")
    train_loader, valid_loader = dataset.get_loaders(
        valid_split=0.2, 
        batch_size=1
    )
    for x in valid_loader:
        x = rearrange(x, "1 channels frames -> frames channels")
        sf.write(f"test_{dataset.dataset_name}.wav", x, 44100)
        break

    dataset = AudioSetDataset()
    train_loader, valid_loader = dataset.get_loaders(valid_split=0.2, batch_size=32)
    for x in valid_loader:
        x = rearrange(x, "1 channels frames -> frames channels")
        sf.write(f"test_wavcaps.wav", x, 44100)
        break
    
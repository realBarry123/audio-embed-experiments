import os
import torch
from torch import nn
from tqdm import tqdm
import torchaudio.transforms as T
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from einops import rearrange
import wandb

from audembed import datasets, models

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

def train_probe(
        model, 
        loader, 
        optim, 
        y_fn,
        encode_fn=None, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True
    ):
    iterator = tqdm(loader, desc=f"E{epoch} Train") if do_tqdm else loader
    total_loss = 0
    total = 0

    for x in iterator:
        x = x.to(device)
        if encode_fn is not None: 
            x = encode_fn(x).detach()

        y = y_fn(x, target_frames=x.shape[1])
            
        model.train()
        y_hat = model(x)
        # print("y (spectrogram):", y.shape, "   y_hat (latent):", y_hat.shape)
        loss = nn.functional.mse_loss(y, y_hat)
        total_loss += loss.item()
        total += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss / total

def valid_probe(
        model, 
        loader, 
        y_fn,
        encode_fn=None, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True
    ):
    iterator = tqdm(loader, desc=f"E{epoch} Valid") if do_tqdm else loader
    total_loss = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x in iterator:
            x = x.to(device)
            if encode_fn is not None: 
                x = encode_fn(x).detach()

            y = y_fn(x, target_frames=x.shape[1])
                
            model.train()
            y_hat = model(x)
            loss = nn.functional.mse_loss(y, y_hat)
            total_loss += loss.item()
            total += 1

    return total_loss / total

state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

def encode_fn(x):
    with diffusion_model.trace("_"):
        latent = diffusion_model.vae.encoder(x).save() # [batch, param*channel, frame]

    latent = rearrange(latent, "b (p c) f -> b f c p", c=64, p=2)[..., 1] # (B, F, C=64)
    with torch.no_grad():
        features = sae.encode(latent).detach()
    return features

spectrogram = T.Spectrogram(n_fft=254, hop_length=127).to(DEVICE)

def to_spectrogram(x, target_frames):
    """
    GPT-5 mini 2026-06-21
    """
    if x.dim() == 3:
        waveform = x.mean(dim=1)  # (batch, time)
    else:
        waveform = x  # (batch, time)

    B, T = waveform.shape
    n_fft = 2048
    # choose hop so that STFT produces ~target_frames frames:
    if target_frames <= 1:
        raise ValueError("target_frames must be at least 1")
    else:
        hop_length = max(1, (T - n_fft) // (target_frames - 1))

    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        return_complex=True,
        center=True,
    )
    mag = rearrange(spec.abs(), "barch freq frames -> batch frames freq") # .contiguous()
    return mag

DO_WANDB = True

train_configs = {
    "batch_size": 1, 
    "lr": 0.001,
    "dataset_name": "orchset"
}

EPOCHS = 16
SEED = 123

try: 
    state_dict, configs, start_epoch = torch.load("experiments/sae/models/probe.pt")
    probe = models.FeatureProbe(**configs).to(DEVICE)
    probe.load_state_dict(state_dict)
except FileNotFoundError:
    start_epoch = 0
    probe = models.FeatureProbe(2048, 128).to(DEVICE)

if DO_WANDB:
    run = wandb.init(
        entity="barry-and-only-barry",
        project="audio-embed-experiments",
        config=dict(probe.configs, **train_configs),
    )

dataset = datasets.MIRDataset(train_configs["dataset_name"])
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.2, 
    batch_size=train_configs["batch_size"]
)

optim = torch.optim.Adam(params=probe.parameters(), lr=train_configs["lr"])

diffusion_model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)

for epoch in range(start_epoch, start_epoch + EPOCHS):
    train_loss = train_probe(
        probe, 
        train_loader, 
        optim, 
        y_fn=to_spectrogram,
        encode_fn=encode_fn, 
        epoch=epoch, 
        device=DEVICE
    )
    valid_loss = valid_probe(
        probe,
        valid_loader,
        y_fn=to_spectrogram, 
        encode_fn=encode_fn, 
        epoch=epoch, 
        device=DEVICE
    )
    if DO_WANDB: 
        run.log({
            "train_loss": train_loss, 
            "valid_loss": valid_loss,
        })
    torch.save([probe.state_dict(), probe.configs, epoch], "experiments/sae/models/probe.pt")
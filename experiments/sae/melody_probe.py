import os
import torch
import wandb
from torch import nn
from tqdm import tqdm
from dotenv import load_dotenv
from einops import rearrange

from audembed import audio_datasets, models

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
        encode_fn, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True
    ):
    iterator = tqdm(loader, desc=f"E{epoch} Train") if do_tqdm else loader
    total_loss = 0
    total = 0

    model.train()

    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        x_feat = encode_fn(x).detach()

        resample_factor = x.shape[2] // x_feat.shape[1]
        y = y[:, resample_factor//2::resample_factor, :][:, :x_feat.shape[1], :]
        
        y_hat = model(x_feat)
        loss = nn.functional.cross_entropy(y, y_hat)
        total_loss += loss.item()
        total += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss / total

def valid_probe(
        model, 
        loader, 
        encode_fn, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True
    ):
    iterator = tqdm(loader, desc=f"E{epoch} Valid") if do_tqdm else loader
    total_loss = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            x_feat = encode_fn(x).detach()

            print(x.shape[1], x_feat.shape[1])

            resample_factor = x.shape[2] // x_feat.shape[1]
            y = y[:, resample_factor//2::resample_factor, :][:, :x_feat.shape[1], :]
            # print(y.shape)

            y_hat = model(x_feat)
            loss = nn.functional.cross_entropy(y, y_hat)
            total_loss += loss.item()
            total += 1

    return total_loss / total

state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
sae = models.SAE(**configs).to(DEVICE)
sae.load_state_dict(state_dict)

vae = models.VAEWrapper(CACHE_PATH, DEVICE)

def encode_fn(x):
    latent = vae.encode(x)
    # latent = rearrange(latent, "b f (p c) -> b f c p", c=64, p=2)[..., 1] # (B, F, C=64)
    with torch.no_grad():
        features = sae.encode(latent).detach()
    return features

DO_WANDB = True

train_configs = {
    "batch_size": 1, 
    "lr": 0.001,
    "dataset_name": "orchset"
}

EPOCHS = 32

try: 
    state_dict, configs, start_epoch = torch.load("experiments/sae/models/melody_probe.pt")
    probe = models.FeatureProbe(**configs).to(DEVICE)
    probe.load_state_dict(state_dict)
except FileNotFoundError:
    start_epoch = 0
    probe = models.FeatureProbe(2048, 26, bias=False).to(DEVICE)

dataset = audio_datasets.MIRDataset(train_configs["dataset_name"], chunk_duration=2.0, include_melody=True)
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.2, 
    batch_size=train_configs["batch_size"],
    seed=0
)

optim = torch.optim.Adam(params=probe.parameters(), lr=train_configs["lr"])

if DO_WANDB:
    run = wandb.init(
        entity="barry-and-only-barry",
        project="audio-embed-experiments",
        config=dict(probe.configs, **train_configs),
    )

if start_epoch == 0: 
    valid_loss = valid_probe(
        probe,
        valid_loader,
        encode_fn=encode_fn, 
        epoch=-1,
        device=DEVICE
    )
    print("Initial validation loss:", valid_loss)


for epoch in range(start_epoch, start_epoch + EPOCHS):
    train_loss = train_probe(
        probe, 
        train_loader, 
        optim, 
        encode_fn=encode_fn, 
        epoch=epoch, 
        device=DEVICE
    )
    valid_loss = valid_probe(
        probe,
        valid_loader,
        encode_fn=encode_fn, 
        epoch=epoch, 
        device=DEVICE
    )
    if DO_WANDB: 
        run.log({
            "train_loss": train_loss, 
            "valid_loss": valid_loss,
        })
    torch.save([probe.state_dict(), probe.configs, epoch+1], "experiments/sae/models/melody_probe.pt")
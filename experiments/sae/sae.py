import os
import torch
from torch import nn
from tqdm import tqdm
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from einops import rearrange
import wandb
from itertools import islice

from audembed import audio_datasets, models

def train_sae(
        model, 
        loader, 
        optim, 
        lamb, 
        encode_fn=None, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True,
        epoch_size=None
    ):
    if epoch_size is not None:
        iterator = islice(loader, epoch_size)
    else: 
        iterator = loader
    iterator = tqdm(iterator, desc=f"E{epoch} Train") if do_tqdm else iterator
    total_loss = 0
    total_sparsity_loss = 0
    total_l0 = 0
    total = 0

    for x in iterator: # x: (batch, channel=2, sample)
        x = x.to(device)
        if encode_fn is not None: 
            x = encode_fn(x).detach()
        
        # x: (batch, frame, latent_dim=64)
        model.train()
        x_hat, h = model(x)
        sparsity_loss = lamb * torch.linalg.vector_norm(h, ord=1, dim=-1).mean()
        loss = nn.functional.mse_loss(x, x_hat) + sparsity_loss
        total_loss += loss.item()
        total_sparsity_loss += sparsity_loss.item()

        l0 = (h.abs() > 1e-8).float().sum(dim=-1).mean()
        total_l0 += l0.item()

        total += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss / total, total_sparsity_loss / total, total_l0 / total

def valid_sae(
        model, 
        loader, 
        lamb, 
        encode_fn=None, 
        epoch=0, 
        device="cuda", 
        do_tqdm=True,
        epoch_size=None
    ):
    if epoch_size is not None:
        iterator = islice(loader, epoch_size)
    else: 
        iterator = loader
    iterator = tqdm(iterator, desc=f"E{epoch} Valid") if do_tqdm else iterator
    total_loss = 0
    total_sparsity_loss = 0
    total_l0 = 0
    total = 0

    for x in iterator:
        x = x.to(device)

        if encode_fn is not None: 
            x = encode_fn(x).detach()

        model.eval()
        with torch.no_grad():
            x_hat, h = model(x)
            sparsity_loss = lamb * torch.linalg.vector_norm(h, ord=1, dim=-1).mean()
            loss = nn.functional.mse_loss(x, x_hat) + sparsity_loss
            total_loss += loss.item()
            total_sparsity_loss += sparsity_loss.item()

            l0 = (h.abs() > 1e-8).float().sum(dim=-1).mean()
            total_l0 += l0.item()

            total += 1
    
    return total_loss / total, total_sparsity_loss / total, total_l0 / total


DO_WANDB = True

train_configs = {
    "batch_size": 1, 
    "lr": 0.001,
    "lambda": 1e-4,
    "dataset_name": "audioset",
    "epoch_size": 64,
    "wandb_id": None
}

EPOCHS = 16
SEED = 123

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

try: 
    state_dict, configs, start_epoch = torch.load("experiments/sae/models/sae.pt")
    model = models.SAE(**configs).to(DEVICE)
    model.load_state_dict(state_dict)
except FileNotFoundError:
    start_epoch = 0
    model = models.SAE(latent_dim=64, feature_dim=8192).to(DEVICE)

if train_configs["dataset_name"] == "audioset":
    dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0)
else: 
    dataset = audio_datasets.MIRDataset(train_configs["dataset_name"])

train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.2, 
    batch_size=train_configs["batch_size"]
)

optim = torch.optim.Adam(params=model.parameters(), lr=train_configs["lr"])

diffusion_model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH,
    device_map=DEVICE
)

def vae_encode(audio):
    with diffusion_model.trace("_"):
        latent = diffusion_model.vae.encode(audio).latent_dist.mean.save()
    return rearrange(latent, "b c f -> b f c")

if DO_WANDB:
    if train_configs["run_id"] is not None:
        run = wandb.init(
            entity="barry-and-only-barry",
            project="audio-embed-experiments",
            config=dict(model.configs, **train_configs),
            resume="allow",
            id=train_configs["run_id"]
        )
    else:
        run = wandb.init(
            entity="barry-and-only-barry",
            project="audio-embed-experiments",
            config=dict(model.configs, **train_configs),
            resume="allow"
        )
        train_configs["run_id"] = run.id

for epoch in range(start_epoch, start_epoch + EPOCHS):
    train_loss, train_sparsity_loss, train_l0 = train_sae(
        model, 
        train_loader, 
        optim, 
        lamb=train_configs["lambda"], 
        encode_fn=vae_encode, 
        epoch=epoch, 
        device=DEVICE,
        epoch_size=train_configs["epoch_size"]
    )
    valid_loss, valid_sparsity_loss, valid_l0 = valid_sae(
        model,
        valid_loader,
        lamb=train_configs["lambda"], 
        encode_fn=vae_encode, 
        epoch=epoch, 
        device=DEVICE,
        epoch_size=int(train_configs["epoch_size"]*0.2)
    )
    if DO_WANDB: 
        run.log({
            "train_loss": train_loss, 
            "train_sparsity_loss": train_sparsity_loss, 
            "train_l0": train_l0,
            "valid_loss": valid_loss,
            "valid_sparsity_loss": valid_sparsity_loss,
            "valid_l0": valid_l0
        })
    torch.save([model.state_dict(), model.configs, epoch], "experiments/sae/models/sae.pt")
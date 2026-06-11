import torch
from torch import nn
from tqdm import tqdm

from audembed import datasets

class SAE(nn.Module):
    def __init__(self, latent_dim, feature_dim, do_relu=True, do_norm=True):
        super(self).__init__()
        self.configs = {
            "latent_dim": latent_dim,
            "feature_dim": feature_dim,
            "do_relu": do_relu,
            "do_norm": do_norm
        }
        self.encoder_linear = nn.Linear(latent_dim, feature_dim, bias=True)
        self.decoder_linear = nn.Linear(feature_dim, latent_dim, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.modules.normalization.RMSNorm()
    
    def encode(self, x):
        x = self.encoder_linear(x)
        x = self.relu(x)
        x = self.norm(x)
        return x
    
    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    

def train_sae(
        model, 
        loader, 
        optim, 
        lamb, 
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
            x = encode_fn(x)

        model.train()
        x_hat, h = model(x)

        loss = nn.functional.mse_loss(x, x_hat) + lamb * torch.linalg.vector_norm(h, dim=-1).mean()
        total_loss += loss.item()
        total += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

    return total_loss / total

def valid_sae(
        model, 
        loader, 
        lamb, 
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
            x = encode_fn(x)

        model.eval()
        with torch.no_grad():
            x_hat, h = model(x)
            loss = nn.functional.mse_loss(x, x_hat) + lamb * torch.linalg.vector_norm(h, dim=-1).mean()
            total_loss += loss.item()
            total += 1
    
    return total_loss / total

if __name__ == "__main__":
    import os
    from nnsight.modeling.diffusion import DiffusionModel
    from dotenv import load_dotenv
    import wandb

    DO_WANDB = True

    EPOCHS = 16
    BATCH_SIZE = 32
    SEED = 123

    try: 
        state_dict, configs, epoch = torch.load("models/sae.pt")
        model = SAE(**configs)
        model.load_state_dict(state_dict)
    except:
        model = SAE(latent_dim=64, feature_dim=2048)
    
    if DO_WANDB:
        run = wandb.init(
            entity="barry-and-only-barry",
            project="audio-embed-experiments",
            config=configs,
        )
    
    dataset = datasets.MIRDataset("orchset")
    train_loader, valid_loader = dataset.get_loaders(valid_split=0.2, batch_size=32)

    optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)

    if not load_dotenv():
        raise SystemExit("No .env file found, please make one in the root directory")

    CACHE_PATH = os.getenv("CACHE_PATH")
    DEVICE = os.getenv("DEVICE")

    if not CACHE_PATH or not DEVICE:
        raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")
    
    diffusion_model = DiffusionModel(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
        cache_dir=CACHE_PATH,
        device_map=DEVICE
    )

    def encode_fn(x):
        with diffusion_model.trace("_"):
            latent = diffusion_model.vae.encoder(x).save() # [batch, param*channel, frame]
        return latent

    for epoch in range(epoch, epoch + EPOCHS):
        train_loss = train_sae(
            model, 
            train_loader, 
            optim, 
            lamb=0.01, 
            encode_fn=encode_fn, 
            epoch=epoch, 
            device=DEVICE
        )
        valid_loss = valid_sae(
            model,
            valid_loader,
            lamb=0.01, 
            encode_fn=encode_fn, 
            epoch=epoch, 
            device=DEVICE
        )
        if DO_WANDB: 
            run.log({"train_loss": train_loss, "valid_loss": valid_loss})
        epochs += 1
        torch.save([model.state_dict(), model.configs, epoch], "models/sae.pt")
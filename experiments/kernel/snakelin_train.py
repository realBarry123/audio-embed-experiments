import os
import torch
from torch import nn
import nnsight
import wandb
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv

from audembed import snake, audio_datasets

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")
CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")
if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

vae = AutoencoderOobleck.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="vae",
    torch_dtype=torch.float32,
    cache_dir=CACHE_PATH
).to(DEVICE)

encoder = nnsight.NNsight(vae.encoder)

EPOCHS = 8
DO_WANDB = False
MODEL_NAME = "snake1lin"

train_configs = {
    "batch_size": 1, 
    "lr": 0.01,
    "epoch_size": 128
}

dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0, device=DEVICE)
train_loader, valid_loader = dataset.get_loaders(
    subset=150,
    valid_split=0.2, 
    batch_size=train_configs["batch_size"]
)
#train_loader = torch.utils.data.Subset(train_loader, range(80))
#valid_loader = torch.utils.data.Subset(valid_loader, range(20))

try: 
    state_dict, features, start_epoch = torch.load(f"experiments/kernel/models/{MODEL_NAME}.pt")
    model = snake.SnakeLinearized(features).to(DEVICE)
    model.load_state_dict(state_dict)
    print(f"loaded up old run, starting epoch {start_epoch}")
except FileNotFoundError:
    model = snake.SnakeLinearized(
        features=encoder.block[0].res_unit1.conv1.weight.shape[0], # out_features
        module=encoder.block[0].res_unit1.snake1
    ).to(DEVICE)
    start_epoch = 0
    print("initialized new model")

if DO_WANDB:
    run = wandb.init(
        entity="barry-and-only-barry",
        project="audio-embed-experiments",
        config=train_configs,
    )

optim = torch.optim.Adam(params=model.parameters(), lr=train_configs["lr"])

for epoch in range(start_epoch, start_epoch + EPOCHS):
    total_loss = 0
    total = 0

    model.train()

    for x in tqdm(train_loader, desc=f"E{epoch} Train"):
        x = x.to(DEVICE)
        with encoder.trace(x):
            x = encoder.block[0].res_unit1.conv1.output.save()
        y = model.original(x)
        y_hat = model(x)
        loss = nn.functional.mse_loss(y, y_hat)
        total_loss += loss.item()
        total += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

    if DO_WANDB: run.log({"train_loss": total_loss / total})
    else: print(f"E{epoch} train_loss: {total_loss / total}")

    torch.save([model.state_dict(), model.features, epoch + 1], f"experiments/kernel/models/{MODEL_NAME}.pt")

    total_loss = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for x in tqdm(valid_loader, desc=f"E{epoch} Valid"):
            x = x.to(DEVICE)
            with encoder.trace(x):
                x = encoder.block[0].res_unit1.conv1.output.save()
            y = model.original(x)
            y_hat = model(x)
            loss = nn.functional.mse_loss(y, y_hat)
            total_loss += loss.item()
            total += 1
            
    if DO_WANDB: run.log({"valid_loss": total_loss / total})
    else: print(f"E{epoch} valid_loss: {total_loss / 20}")
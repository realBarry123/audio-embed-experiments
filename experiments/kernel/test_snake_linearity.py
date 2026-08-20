import os
import torch
import nnsight
from tqdm import tqdm
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv
from audembed import audio_datasets

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

dataset = audio_datasets.AudioSetDataset(chunk_duration=2.0, device=DEVICE)
train_loader, valid_loader = dataset.get_loaders(
    valid_split=0.01, # comes down to 915 clips
    batch_size=1
)
encoder = nnsight.NNsight(vae.encoder)

snake_layers = {}

for block in range(5):
    snake_layers.update({
        f"block{block}.res1.snake1": encoder.block[block].res_unit1.snake1,
        f"block{block}.res1.snake2": encoder.block[block].res_unit1.snake2,
        f"block{block}.res2.snake1": encoder.block[block].res_unit2.snake1,
        f"block{block}.res2.snake2": encoder.block[block].res_unit2.snake2,
        f"block{block}.res3.snake1": encoder.block[block].res_unit3.snake1,
        f"block{block}.res3.snake2": encoder.block[block].res_unit3.snake2,
        f"block{block}.final_snake1": encoder.block[block].snake1
    })
snake_layers.update({"final_snake1": encoder.snake1})

snake_losses = snake_layers.fromkeys(snake_layers, 0)

#def loss_fn(x, y):
#    cos_sim = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
#    return (cos_sim + 1) / 2

def mre(x, y):
    return (x - y).abs().mean().item() / x.abs().mean().item()

i = 0
for x in tqdm(valid_loader):
    x = x.to(DEVICE)
    with encoder.trace(x):
        for name, module in snake_layers.items():
            loss = mre(module.input.save(), module.output.save())
            snake_losses[name] += loss
    i += 1

snake_losses.update((name, loss/i) for name, loss in snake_losses.items())

print(snake_losses)
import pickle 
with open('experiments/kernel/results/snake_losses.pkl', 'wb') as f:
    pickle.dump(snake_losses, f)
import os
import torch
from diffusers import AutoencoderOobleck
from dotenv import load_dotenv

if not load_dotenv(): 
    raise SystemExit("No .env file found, please make one in the root directory")
CACHE_PATH = os.getenv("CACHE_PATH")
if not CACHE_PATH: 
    raise ValueError("Missing required environment variables: CACHE_PATH")

vae = AutoencoderOobleck.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    subfolder="vae",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH
).to("cpu")

def get_snake_params(snake: torch.nn.Module):
    alpha = snake.alpha if not snake.logscale else torch.exp(snake.alpha)
    beta = snake.beta if not snake.logscale else torch.exp(snake.beta)
    return alpha.squeeze((0, 2)).detach().clone().cpu(), beta.squeeze((0, 2)).detach().clone().cpu()

def print_snake_params(snake: torch.nn.Module, label: str):
    alpha, beta = get_snake_params(snake)
    norms = torch.stack((alpha, beta), dim=1).norm(dim=1)
    diffs = alpha-beta
    print(f"\n{label}:")
    print(f"\talpha: {round(alpha.min().item(), 3)}–{round(alpha.max().item(), 3)}")
    print(f"\tbeta: {round(beta.min().item(), 3)}–{round(beta.max().item(), 3)}")
    print(f"\tnorm: {round(norms.min().item(), 3)}–{round(norms.max().item(), 3)}")
    print(f"\tdiff: {round(diffs.min().item(), 3)}–{round(diffs.max().item(), 3)}")


print_snake_params(vae.encoder.block[0].res_unit1.snake1, "0.res_unit1.snake1")
print_snake_params(vae.encoder.block[0].res_unit1.snake2, "0.res_unit1.snake2")
print_snake_params(vae.encoder.block[0].res_unit2.snake1, "0.res_unit2.snake1")
print_snake_params(vae.encoder.block[0].res_unit2.snake2, "0.res_unit2.snake2")
print_snake_params(vae.encoder.block[0].res_unit3.snake1, "0.res_unit3.snake1")
print_snake_params(vae.encoder.block[0].res_unit3.snake2, "0.res_unit3.snake2")
print_snake_params(vae.encoder.block[0].snake1, "0.snake1")
import os
import sys
import datetime

import torch
import soundfile as sf
from nnsight.modeling.diffusion import DiffusionModel
from dotenv import load_dotenv
from tqdm import tqdm

if len(sys.argv) != 2:
    raise SystemExit(f"Incorrect number of command line arguments")
prompt = sys.argv[1]

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")

# Create folder for results
timestamp = str(datetime.datetime.now().strftime("%y-%m-%dT%H:%M:%S"))

if not os.path.exists(timestamp):
    os.makedirs(f"experiments/lens/results/{timestamp}")
else:
    raise ZeroDivisionError("Time is moving backwards!")

model = DiffusionModel(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16,
    cache_dir=CACHE_PATH,
    #dispatch=True,
    device_map=DEVICE
)

for layer in tqdm(range(-1, model.text_encoder.config.num_hidden_layers)):
    with torch.no_grad():
        with model.generate(
            prompt,
            num_inference_steps=200, # 200
            audio_end_in_s=4.0,
            seed=0
        ):
            if layer == -1:
                #TODO
                pass
            else:
                print(f"\n\nlayer: {layer}")
                hidden_state = model.text_encoder.encoder.block[layer].output[0]
                print("range of hidden state:", hidden_state.min().item(), hidden_state.max().item())
                model.text_encoder.encoder.final_layer_norm.input = hidden_state
                audio = model.output.audios[0].save()
                audio = audio.T.float().cpu().numpy()
                sf.write(f"experiments/lens/results/{timestamp}/layer{layer}.wav", audio, model.vae.sampling_rate)

with open("experiments/lens/results/report.txt", "a") as f:
    f.write(f"{timestamp}: \'{prompt}\'")

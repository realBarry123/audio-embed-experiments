import os
import sys
from dotenv import load_dotenv

from stable_audio_3 import StableAudioModel
import soundfile as sf

#if len(sys.argv) != 2:
#    raise SystemExit(f"Incorrect number of command line arguments")
#prompt = sys.argv[1]

if not load_dotenv():
    raise SystemExit("No .env file found, please make one in the root directory")

CACHE_PATH = os.getenv("CACHE_PATH")
DEVICE = os.getenv("DEVICE")

if not CACHE_PATH or not DEVICE:
    raise ValueError("Missing required environment variables: CACHE_PATH, DEVICE")


model = StableAudioModel.from_pretrained(
    model_name="medium",
    device=DEVICE
)
with open("stable-audio-3.txt", "w") as f:
    f.write(str(model))
exit()
audio = model.generate(
    prompt=(
        "House music that encapsulates the feeling of being at a festival "
        "in the sunny weather with all your friends 124 BPM"
    ),
    duration=180
)

audio = audio[0].T.float().cpu().numpy()
sf.write(f"experiments/lens/stable3.wav", audio, model.vae.sampling_rate)

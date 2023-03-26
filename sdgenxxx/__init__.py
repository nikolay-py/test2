import sdkit
from sdkit.models import download_model, load_model, resolve_downloaded_model_path
import sys
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
sys.stdout = None

load_dotenv()

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")
DEVICE = os.getenv("DEVICE")

#Initizlize Context, set device and load models
context = sdkit.Context()
context.model_paths["stable-diffusion"] = WEIGHTS_PATH + 'uberRealisticPornMerge_urpmv13.safetensors'
context.device = DEVICE
load_model(context, "stable-diffusion")
context.model_paths["nsfw_checker"] = WEIGHTS_PATH + 'nsfw_checker'
load_model(context, "nsfw_checker")
context.model_paths["gfpgan"] = WEIGHTS_PATH + 'GFPGANv1.4.pth'
load_model(context, "gfpgan")

def get_context():
    return context
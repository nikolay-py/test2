import sdkit
from sdkit.generate import generate_images
from sdkit.models import download_model, load_model
from sdkit.filter import apply_filters
from sdkit.utils import log, save_images
from PIL import Image
import random
import json
import base64
from io import BytesIO
import sys
import warnings
import argparse
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
sys.stdout = None

load_dotenv()

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH")

def parse_cmd_line() -> str:
    parser = argparse.ArgumentParser(description='Stable Diffusion')
    parser.add_argument(
        '--prompt_json', action='store', required=True,
        help='json file with prompt and parameters')
    
    parser.add_argument(
        '--device', action='store', required=False,
        help='device cpu or cuda index')

    args = parser.parse_args()

    return args.prompt_json, args.device


def text2image():
    """Process prompt with SD"""
    PROMPT_FILE, DEVICE = parse_cmd_line()
    
    # load prompt from file
    with open(PROMPT_FILE, "r") as read_file:
        config = json.load(read_file)
    
    if (DEVICE != '') and (DEVICE != ' ') and (DEVICE is not None):
        config['device'] = DEVICE
    
    # initialize context object, load SD model - UPRM
    context = sdkit.Context()
    context.model_paths["stable-diffusion"] = WEIGHTS_PATH+'uberRealisticPornMerge_urpmv13.safetensors'
    context.device = config['device']
    load_model(context, "stable-diffusion")
    
    #generate images
    seed = random.randint(1, 10000000)
    images = generate_images(context, 
                             prompt=config['prompt'],
                             negative_prompt=config['negative_prompt'],
                             sampler_name=config['sampler'],
                             seed=seed,
                             width=config['width'],
                             height=config['height'],
                             num_outputs=config['n_images'],
                             guidance_scale=config['guidance_scale'],
                             num_inference_steps=config['num_inference_steps'])
    
    #if sfw - check generated inages
    if not config['nsfw']:
        context.model_paths["gfpgan"] = WEIGHTS_PATH + 'nsfw_checker'
        load_model(context, "nsfw_checker")
        images = apply_filters(context, "nsfw_checker", images)
    
    #make face restoration with gfpgun
    if config['gfpgan']:
        context.model_paths["gfpgan"] = WEIGHTS_PATH + "GFPGANv1.4.pth"
        load_model(context, "gfpgan")
        images = apply_filters(context, "gfpgan", images)
        
    #print generated images as json
    json_result = []
    for i, image in enumerate(images):
        buffer = BytesIO()
        image.save(buffer,format="JPEG")
        myimage = buffer.getvalue()
        enc_image_file = base64.b64encode(myimage)
        send_image_file = enc_image_file.decode('utf8')
        json_result.append({'num': i, 'pict': send_image_file})

    sys.stdout = sys.__stdout__
    print(json.dumps(json_result))
          
if __name__ == "__main__":
          text2image()
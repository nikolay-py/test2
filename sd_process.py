import warnings
import os
import random
import json
import base64
from io import BytesIO
import io
import sys

from dotenv import load_dotenv
import sdkit
from sdkit.models import download_model, load_model
from sdkit.generate import generate_images
from sdkit.filter import apply_filters
from sdkit.utils import log
from PIL import Image

#Block warnings
warnings.filterwarnings("ignore")
sys.stdout = None

#Get env parameters
load_dotenv()
PATH = os.getenv("WEIGHTS_PATH")
DEVICE = os.getenv("DEVICE")
NSFW = os.getenv("NSFW")

if not PATH.endswith('/'):
    PATH = PATH + '/'

#Initizlize Context, set device and load models
context = sdkit.Context()
if NSFW:
    context.model_paths["stable-diffusion"] = PATH+'weights/uberRealisticPornMerge_urpmv13.safetensors'
else:
    context.model_paths["stable-diffusion"] = PATH+'weights/v1-5-pruned-emaonly.safetensors'
    context.model_paths["nsfw_checker"] = PATH + 'weights/nsfw_checker'
    load_model(context, "nsfw_checker")
    
context.device = DEVICE
load_model(context, "stable-diffusion")

context.model_paths["gfpgan"] = PATH + "weights/GFPGANv1.4.pth"
load_model(context, "gfpgan")

context.model_paths["realesrgan"] = PATH +  "weights/RealESRGAN_x4plus.pth"
load_model(context, "realesrgan")

def process_prompt(config):
    """Process prompt with SD"""
    
    # Decode image if img2img
    if config['image'] == '':
        config['image'] = None
    else:
        img = base64.b64decode(config['image'])
        buf = io.BytesIO(img)
        config['image'] = Image.open(buf)
        
    #Generate images or rescale
    if config['scale'] > 1:
        images = apply_filters(context, "realesrgan", config['image'], scale=config['scale'])
    else:
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
                                 num_inference_steps=config['num_inference_steps'],
                                 init_image=config['image'])
    
        #If sfw - check generated inages
        if not NSFW:
            images = apply_filters(context, "nsfw_checker", images)

        #Make face restoration with gfpgun
        if config['gfpgan']:
            images = apply_filters(context, "gfpgan", images)
      
    #Encode results
    result = []
    for i, image in enumerate(images):
        buffer = BytesIO()
        image.save(buffer,format="JPEG")
        myimage = buffer.getvalue()
        enc_image_file = base64.b64encode(myimage)
        send_image_file = enc_image_file.decode('utf8')
        result.append({'num': i, 'pict': send_image_file})

    #sys.stdout = sys.__stdout__
    return result 
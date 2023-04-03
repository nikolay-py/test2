import base64
import warnings
import os
import random
import base64
from io import BytesIO
import io
import sys

import sdkit
from sdkit.models import load_model
from sdkit.generate import generate_images
from sdkit.filter import apply_filters
from sdkit.utils import log
from PIL import Image

from config import Config
import contextlib

# # #Block warnings
# warnings.filterwarnings("ignore")
# sys.stdout = None



#Get env parameters
path = Config.WEIGHTS_PATH
device = Config.DEVICE
nsfw = Config.NSFW

def get_context():
    context = sdkit.Context()
    if nsfw:
        context.model_paths["stable-diffusion"] = os.path.join(path,'weights/uberRealisticPornMerge_urpmv13.safetensors')
    else:
        context.model_paths["stable-diffusion"] = os.path.join(path,'weights/v1-5-pruned-emaonly.safetensors')
        context.model_paths["nsfw_checker"] = os.path.join(path,'weights/nsfw_checker')
        load_model(context, "nsfw_checker")

    context.device = device
    load_model(context, "stable-diffusion")

    context.model_paths["gfpgan"] = os.path.join(path,'weights/GFPGANv1.4.pth')
    load_model(context, "gfpgan")

    context.model_paths["realesrgan"] = os.path.join(path,'weights/RealESRGAN_x4plus.pth')
    load_model(context, "realesrgan")

with contextlib.redirect_stdout(None):
    warnings.filterwarnings("ignore")
    context = get_context()

def process_prompt(request_data, context=context):
    """Process prompt with SD"""
    
    # Decode image if img2img
    if request_data['image'] == '':
        request_data['image'] = None
    else:
        img = base64.b64decode(request_data['image'])
        buf = io.BytesIO(img)
        request_data['image'] = Image.open(buf)
        
    #Generate images or rescale
    if request_data['scale'] > 1:
        images = apply_filters(context, "realesrgan", request_data['image'], scale=request_data['scale'])
    else:
        seed = random.randint(1, 10000000)
        images = generate_images(context, 
                                prompt=request_data['prompt'],
                                negative_prompt=request_data['negative_prompt'],
                                sampler_name=request_data['sampler'],
                                seed=seed,
                                width=request_data['width'],
                                height=request_data['height'],
                                num_outputs=request_data['n_images'],
                                guidance_scale=request_data['guidance_scale'],
                                num_inference_steps=request_data['num_inference_steps'],
                                init_image=request_data['image'])
    
        #If sfw - check generated inages
        if not nsfw:
            images = apply_filters(context, "nsfw_checker", images)

        #Make face restoration with gfpgun
        if request_data['gfpgan']:
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

if __name__ == "__main__":
    pass
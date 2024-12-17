import torch
# from diffusers import FluxPipeline
from diffusers.utils import load_image
import base64
import io
from PIL import Image
import os
import requests
from urllib.parse import urlparse

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

class ImageGenerator:
    def __init__(self,device="cuda"):
        # Initialize the model and move it to GPU
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)

    def generate_image(self, prompt: str) -> str:
        # Generate the image
        image = self.pipe(
            prompt=prompt,
            output_type="pil"
        ).images[0]

        torch.cuda.empty_cache()

        # Convert image to base64 and return
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    
# class ImageGenerator:
#     def __init__(self):
#         self.pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#         self.pipe.to("cuda:0")

#     def generate_image(self, prompt: str) -> str:
#         # Fixed parameters
#         steps = 28
#         height = 512
#         width = 512
#         seed=42
#         image = self.pipe(
#             prompt=prompt,
#             output_type="pil",
#             num_inference_steps=steps,
#             height=height,
#             width=width,
#             generator=torch.Generator("cpu").manual_seed(seed)
#         ).images[0]

#         torch.cuda.empty_cache()
        
#         # Convert image to base64 and return
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
#         return img_str

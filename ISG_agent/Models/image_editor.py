import torch
from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image
import base64
import io
from typing import Union
from PIL import Image
import os
import requests
from urllib.parse import urlparse
class ImageEditor:
    def __init__(self):
        # Initialize the Stable Diffusion 3 pipeline
        self.pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            "BleachNick/SD3_UltraEdit_w_mask",
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str

    def base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL image."""
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return img

    def load_image_from_path_or_base64(self, image_input: Union[str, bytes]) -> Image.Image:
        """Load an image from either a file path or a base64 string."""
        print(image_input)
        if isinstance(image_input, str):
            if os.path.exists(image_input):
                # It's a file path
                
                return load_image(image_input).convert("RGB")
            else:
                # It's a base64 string
                return self.base64_to_image(image_input)
        else:
            raise ValueError("Unsupported image input type. Please pass a file path or base64 string.")

    def edit_image(self, prompt: str, image_input: str) -> str:
        """
        Edit the image based on the provided prompt, image, and mask (if any) and return the result in base64 format.

        Args:
            prompt (str): Text prompt for the image edit.
            image_input (str): Image in base64 format or as a file path.
            mask_input (str, optional): Mask in base64 format or as a file path.

        Returns:
            str: Edited image in base64 format.
        """
        img = self.load_image_from_path_or_base64(image_input).resize((512, 512))


        # Fixed parameters
        steps = 50
        image_guidance_scale = 1.5
        guidance_scale = 7.5

        # Perform the image editing using the SD3 pipeline
        edited_image = self.pipe(
            prompt=prompt,
            image=img,
            negative_prompt="",
            num_inference_steps=steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale
        ).images[0]
        torch.cuda.empty_cache()
        # Convert edited image to base64
        return self.image_to_base64(edited_image)

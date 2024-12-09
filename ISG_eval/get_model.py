from openai import OpenAI
import time
import torch
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    FluxPipeline
)
import re
import os
import json
import anthropic
import base64
import imghdr
import io
from PIL import Image
from .utils import local_image_to_data_url, modify_content
import google.generativeai as genai
import gc

# API Keys
openai_api = os.environ.get("OPENAI_API_KEY")
claude_api = os.environ.get("ANTHROPIC_API_KEY")
gemini_api = os.environ.get("GEMINI_API_KEY")


class LLM_SD:
    def __init__(self, text_generator, image_generator):
        self.text_generator = text_generator
        self.image_generator = image_generator
        
        # Initialize image generation pipeline
        if image_generator == "sd3":
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16
            ).to("cuda")
            
        elif image_generator == "sd2.1":
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16
            ).to("cuda")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
        elif image_generator == "flux":
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            ).to("cuda")

    def generate_image(self, prompt):
        if self.image_generator == "sd3":
            image = self.pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
            ).images[0]
            
        elif self.image_generator == "sd2.1":
            image = self.pipe(prompt).images[0]
            
        elif self.image_generator == "flux":
            image = self.pipe(
                prompt,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cuda").manual_seed(0)
            ).images[0]
        
        gc.collect()
        torch.cuda.empty_cache()
        return image

    def get_res(self, content, image:bool = True):   
        print(content)
        
        # GPT-4 Vision
        if self.text_generator in ["gpt-4o", "gpt4o-mini"]:
            return self._handle_gpt4_vision(content)
            
        # Claude
        elif self.text_generator == "claude-3.5-sonnet":
            return self._handle_claude(content, image)
            
        # Gemini
        elif self.text_generator == "gemini":
            return self._handle_gemini(content, image)

    def _handle_gpt4_vision(self, content):
        # Process images for GPT-4 Vision
        for item in content:
            if item["type"] == "image":
                item["image_url"] = {"url": local_image_to_data_url(item["content"])}
                item['type'] = 'image_url'
                item.pop("content")
        
        client = OpenAI(api_key=openai_api)
        success = False
        attempt = 0
        
        while not success and attempt < 3:
            try:
                chat_completion = client.chat.completions.create(
                    model='gpt-4o-latest',
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that can generate images based on the user's instructions. If the task requirement need you to provide the caption of the generated image, please provide it out of <image> and </image>. Notice: please use <image> and </image> to wrap the image caption for the images you want to generate. For example, if you want to generate an image of a cat, you should write <image>a cat</image> in your output. "
                        },
                        {"role": "user", "content": content}
                    ],
                    max_tokens=4096
                )
                success = True
                print(chat_completion)
                return chat_completion.choices[0].message.content.strip()
                
            except Exception as e:
                print(e)
                attempt += 1
                time.sleep(10)

    def _handle_claude(self, content, image):
        content_list = []
        
        for item in content:
            if item["type"] == "image":
                if image:
                    content_list.append(self._process_image_for_claude(item["content"]))
                else:
                    content_list.append({
                        "type": "text",
                        "text": "```Here is an image block, while I can't provide you the real image content. You can assume here is an image here.```"
                    })
            elif item["type"] == "text":
                content_list.append({
                    "type": "text",
                    "text": item["content"]
                })
        
        print(content_list)
        client = anthropic.Anthropic(api_key=claude_api)
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.9,
            system="You are a helpful assistant. Please follow my instructions carefully.",
            messages=[{"role": "user", "content": content_list}]
        )
        
        print(message.content[0].text)
        return message.content[0].text

    def _process_image_for_claude(self, image_path):
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            
        file_extension = imghdr.what(None, h=image_data)
        if file_extension not in ['jpeg', 'jpg', 'png', 'webp']:
            raise ValueError(f"Unsupported image format: {file_extension}")
            
        if file_extension == 'webp':
            img = Image.open(io.BytesIO(image_data))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            file_extension = 'png'
            
        mime_type = f'image/{file_extension}'
        img = Image.open(io.BytesIO(image_data))
        img = img.resize((256, 256), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format=file_extension.upper())
        image_data = buffer.getvalue()

        if len(image_data) > 5 * 1024 * 1024:  # 5MB limit
            raise ValueError("Image size exceeds 5MB limit")
            
        image_data_b64 = base64.b64encode(image_data).decode('utf-8')
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_data_b64,
            }
        }

    def _handle_gemini(self, content, image):
        genai.configure(api_key=gemini_api)
        content_list = []
        
        for item in content:
            if item["type"] == "image":
                if image:
                    image = genai.upload_file(path=item["content"])
                    content_list.append(image)
                else:
                    content_list.append("Here is an image block, while I can't provide you the real image content. You can assume here is an image here.")
            if item["type"] == "text":
                content_list.append(item["content"])
        
        print(content_list)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

        for attempt in range(3):
            try:
                response = model.generate_content(content_list)
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    print("All attempts failed. Skipping this sample.")
                    return None

    def get_mm_output(self, content, save_dir, id):
        text_output = self.get_res(content)
        return self._process_output(text_output, save_dir, id)

    def get_mm_output_wo_image(self, content, save_dir, id):
        text_output = self.get_res(content, image=False)
        return self._process_output(text_output, save_dir, id)
        
    def _process_output(self, text_output, save_dir, id):
        print(text_output)
        image_captions = re.findall(r'<image>(.*?)</image>', text_output)
        result = []
        text_parts = re.split(r'<image>.*?</image>', text_output)
        
        for i, text in enumerate(text_parts):
            if text.strip():
                result.append({"type": "text", "content": text.strip()})
            if i < len(image_captions):
                image_dict = {
                    "type": "image",
                    "caption": image_captions[i]
                }
                image = self.generate_image(image_captions[i])
                image_filename = f"{id}_g{i+1}.png"
                image_path = os.path.join(save_dir, image_filename)
                image.save(image_path)
                print(f"Image saved to {image_path}")
                image_dict["content"] = image_path
                result.append(image_dict)
                
        return result
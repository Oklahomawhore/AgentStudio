from openai import AzureOpenAI, OpenAI
import time
import anthropic
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import base64
import json

YOUR_API_KEY = os.environ.get('OPENAI_API_KEY')
class VQA_Model:
    def __init__(self, model_name):
        self.prompt = """You are a helpful and impartial visual assistant. Please follow user's instructions strictly."""
        self.model_name = model_name
    def generate_answer(self, content):
        pass
    
class VQA_GPT4O(VQA_Model):
    def __init__(self):
        super().__init__("gpt-4o")

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def generate_answer(self, content):
        client = AzureOpenAI(
            api_key='',  # replace with your actual API key
            api_version='',
            azure_endpoint=''
        )
        try:
            chat_completion = client.chat.completions.create(
                model='', # replace with your actual model name
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": content}],
                temperature=0,
                response_format={"type": "json_object"}
                )
            output = chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(e)
            output = None
                
        if output is not None:
            try:
                output = json.loads(output)
            except:
                print(output)
                output = None
        print(output)
        return output
    
# We do not use these model in our evaluation, but we keep these models for future use
    
# class VQA_Claude(VQA_Model):
#     def __init__(self):
#         super().__init__("claude-3.5-sonnet")

#     def encode_image(image_path):
#         with open(image_path, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
        
#     def generate_answer(self, content):
#         for dict in content:
#             if dict['type'] == 'image':
#                 image_base64 = self.encode_image(dict['image_path'])
#                 dict['source'] = {
#                     "type": "base64",
#                     "media_type": "image/png",
#                     "data": image_base64
#                 }
#                 dict.pop('image_path')
                
#         client = anthropic.Anthropic(
#                 # defaults to os.environ.get("ANTHROPIC_API_KEY")
#                 api_key="" # replace with your actual API key
#             )
#         success = False
#         attempt = 0
#         while not success and attempt < 3:
#             try:
#                 message = client.messages.create(
#                     model="claude-3-5-sonnet-20240620",
#                     max_tokens=1000,
#                     temperature=0.9,
#                     system="You are a helpful assistant.",
#                     messages=[{
#                             "role": "user",
#                             "content": content,
#                         }
#                     ]
#                 )
#                 success = True
#                 output = message.content[0].text
#             except Exception as e:
#                 print(e)
#                 attempt += 1
#                 time.sleep(10)
#         print(output)
#         return output
    
    
# class VQA_Local(VQA_Model):
#     def __init__(self, model_name, mode):
#         super().__init__(model_name, mode)
#         if model_name == "internvl2-8b":
#             path = "OpenGVLab/InternVL2-8B"
#             self.model = AutoModel.from_pretrained(
#                 path,
#                 torch_dtype=torch.bfloat16,
#                 low_cpu_mem_usage=True,
#                 use_flash_attn=True,
#                 trust_remote_code=True).eval().cuda()
#             self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
#         else:
#             raise ValueError(f"Model {model_name} not found")
            
#     def build_transform(self, input_size):
#         IMAGENET_MEAN = (0.485, 0.456, 0.406)
#         IMAGENET_STD = (0.229, 0.224, 0.225)
#         MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
#         transform = T.Compose([
#             T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#             T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
#             T.ToTensor(),
#             T.Normalize(mean=MEAN, std=STD)
#         ])
#         return transform

#     def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
#         best_ratio_diff = float('inf')
#         best_ratio = (1, 1)
#         area = width * height
#         for ratio in target_ratios:
#             target_aspect_ratio = ratio[0] / ratio[1]
#             ratio_diff = abs(aspect_ratio - target_aspect_ratio)
#             if ratio_diff < best_ratio_diff:
#                 best_ratio_diff = ratio_diff
#                 best_ratio = ratio
#             elif ratio_diff == best_ratio_diff:
#                 if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
#                     best_ratio = ratio
#         return best_ratio

#     def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
#         orig_width, orig_height = image.size
#         aspect_ratio = orig_width / orig_height

#         # calculate the existing image aspect ratio
#         target_ratios = set(
#             (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
#             i * j <= max_num and i * j >= min_num)
#         target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

#         # find the closest aspect ratio to the target
#         target_aspect_ratio = self.find_closest_aspect_ratio(
#             aspect_ratio, target_ratios, orig_width, orig_height, image_size)

#         # calculate the target width and height
#         target_width = image_size * target_aspect_ratio[0]
#         target_height = image_size * target_aspect_ratio[1]
#         blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

#         # resize the image
#         resized_img = image.resize((target_width, target_height))
#         processed_images = []
#         for i in range(blocks):
#             box = (
#                 (i % (target_width // image_size)) * image_size,
#                 (i // (target_width // image_size)) * image_size,
#                 ((i % (target_width // image_size)) + 1) * image_size,
#                 ((i // (target_width // image_size)) + 1) * image_size
#             )
#             # split the image
#             split_img = resized_img.crop(box)
#             processed_images.append(split_img)
#         assert len(processed_images) == blocks
#         if use_thumbnail and len(processed_images) != 1:
#             thumbnail_img = image.resize((image_size, image_size))
#             processed_images.append(thumbnail_img)
#         return processed_images

#     def load_image(self, image_file, input_size=448, max_num=12):
#         image = Image.open(image_file).convert('RGB')
#         transform = self.build_transform(input_size=input_size)
#         images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#         pixel_values = [transform(image) for image in images]
#         pixel_values = torch.stack(pixel_values)
#         return pixel_values
    
#     def generate_answer(self, content):
#         prompt = content['text']
#         image_path = content['image_path']
#         pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
#         generation_config = dict(max_new_tokens=1024, do_sample=True)
#         question = f'<image>\n{prompt}'
#         response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
#         print(response)
#         return response
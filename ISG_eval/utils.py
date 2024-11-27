import os
import json
import re
import base64
import time
from openai import AzureOpenAI, OpenAI
from mimetypes import guess_type

YOUR_API_KEY = os.environ.get('OPENAI_API_KEY')
def modify_content(dir):
    json_file = [f for f in os.listdir(dir) if f.endswith('.json')][0]
    with open(os.path.join(dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data.get('text', '')
    
    parts = re.split(r'(#image\d+#)', text)
    
    result = []
    for part in parts:
        if part.startswith('#image') and part.endswith('#'):
            result.append({
                "type": "image",
                "image_path": part.strip('#')
            })
        elif part.strip():
            result.append({
                "type": "text",
                "content": part.strip()
            })
    
    return result


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


def get_detailed_caption(image_path):
    content = [{"type": "text",
                "text": """Task: Generate a detailed caption for an image.
Input: An image.

Output: A detailed caption describe what is in this image. Focus on all important entities, their attributes and their relationships. Do not include any other information. Make sure the caption is clear, accurate and easy to understand.

Here is the images:"""},
               {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}]
    client = OpenAI(
                        api_key=YOUR_API_KEY
                    )
    success = False
    attempt = 0
    while not success and attempt < 3:
        try:
            response = client.chat.completions.create(
                        model="chatgpt-4o-latest",
                        messages=[
                        {"role": "system", "content": "You are a helpful assistant. Please output in JSON format. Do not write an introduction or summary. Do not output other irrelevant information. Do not output the example in the prompt. Focus on the input prompt and image."},
                        {"role": "user", "content": content}],
                    temperature=0.7,
                    )
            success = True
            output = response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            attempt += 1
            time.sleep(10)
    
    # print(output)
    return output

def get_caption(image_path):
    content = [{"type": "text",
                "text": """Task: Generate a caption for an image.
                            Input: An image.

                            Output: A short and accurate caption describe what is in this image. Focus on the main entities, their attributes and their relationships. Do not include any other information. Make sure the caption is clear, accurate and easy to understand.

                            Here is the images:"""},
               {"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}]

    client = OpenAI(
                        api_key=YOUR_API_KEY
                    )
    success = False
    attempt = 0
    while not success and attempt < 3:
        try:
            response = client.chat.completions.create(
                        model="chatgpt-4o-latest",
                        messages=[
                        {"role": "system", "content": "You are a helpful assistant. Please output in JSON format. Do not write an introduction or summary. Do not output other irrelevant information. Do not output the example in the prompt. Focus on the input prompt and image."},
                        {"role": "user", "content": content}],
                    temperature=0.7,
                    )
            success = True
            output = response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            attempt += 1
            time.sleep(10)
        return output

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


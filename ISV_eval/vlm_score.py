import os
from openai import OpenAI
import dotenv
from util import capture_screenshots
import base64
import dashscope
from http import HTTPStatus
import glob
from typing import List, Tuple, Dict, Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
import hashlib
import requests

from ISV_eval.util import inference_batch

model = None
processor = None

# Load the model and processor only once for local inference
def load_model_and_processor():
    global model, processor
    if model is None or processor is None:
        print("Loading Qwen2.5-VL model and processor...")
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
        print(f"Model and processor loaded successfully. Model on device {model.device}")
    return model, processor

from ISV_eval.util import prepare_message_for_vllm
dotenv.load_dotenv()

from moviepy import VideoFileClip

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")

def get_video_frames(video_path, num_frames=128, cache_dir='tmp'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)
    
    return video_file_path, frames, timestamps

def inference(model, processor, video_path=None, prompt: List[Dict] | str=None, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    if not model or not processor:
        model, processor = load_model_and_processor()
    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": f"file://{video_path}", "max_pixels": 8 * 28 * 28},
                ]
            },
        ]
    else:
        messages = prompt
        for message in messages:
            for content in message['content']:
                if isinstance(content, dict) and content['type'] == 'video':
                    content['max_pixels'] = 32 * 28 * 28
        
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    # print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs)
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    # print(f"shape: {len(generated_ids)} {generated_ids}")
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]
def inference_local(video_path, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": f"file://{video_path}"},
            ]
        },
    ]
    # use local
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"


    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        messages=messages
    )
    reply = chat_response
    return reply.choices[0].message.content


def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration  # Duration in seconds
    clip.close()
    return duration

def score_video(video_path, prompt=None):
    
    # client = OpenAI(
    #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # completion = client.chat.completions.create(
    #     model="qwen2.5-vl-72b-instruct",
    #     messages=[{"role": "user","content": [
    #         {"type": "video","video": [f"file://{path}" for path in paths]},
    #         {"type": "text","text": prompt if prompt else "Judge the quality of this video, first state your reasons and then give a score from 1-10"},
    #     ]}]
    # )
    # print(completion.choices[0].message.content)

    messages = [{
        "role": "user",
        "content": [
                    {"type" : "video", "video": video_path},
                    {"type" : "text" , "text": prompt if prompt else "Judge the quality of this video, first state your reasons and then give a score from 1-10"},
                 ]
                 }]
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model='qwen2.5-vl-32b-instruct',
        messages=messages,
    )
    if response.status_code != HTTPStatus.OK:
        print(response.message)
    try:
        print(response["output"]["choices"][0]["message"].content[0]["text"])
    except:
        print(response)
    # response = inference(video_path, prompt)
    print("response:", response)
    # return response
    return response["output"]["choices"][0]["message"].content[0]["text"]
    # return completion.choices[0].message.content

def score_video_batch(video_path: List[str], prompt=None):
    messages = []
    for video in video_path:
        message = []
        contents = [
            {"type" : "text", "text": prompt if prompt else "Judge the quality of this video, first state your reasons and then give a score from 1-10"}, 
            {"type" : "video", "video": video}
            ]
        message.append({"role":"system", "content": "You are a video caption expert."})
        message.append({"role": "user", "content": contents})
        messages.append(message)
    results = inference_batch(messages, "qwen-vl-max-latest", job_id=f"vlm_score_{video_path[0].split('/')[-1].split('.')[0]}")
    return results

def shorten_caption(scenes: List[str]):
    messages = []
    for scene in scenes:
        prompt = f"given this detailed scene description: {scene['caption']}, give me a short version with only necessary details."
        messages.append([{"role": "user", "content": prompt}])
    results = inference_batch(messages, "qwen-vl-max-latest", job_id=f"vlm_score_{scenes[0]['path'].split('/')[-1].split('.')[0]}")
    return results
if __name__ == '__main__':
    prompt= "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. Provide the result in json format with 'mm:ss.ff' format for time depiction."
    response = inference("/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning_run2/Task_0002/final_video_919c4168-d0c1-4404-a931-edfa1981ce1a.mp4", prompt=prompt) # 这里的时间戳是视频的时间戳
    print("response:", response)
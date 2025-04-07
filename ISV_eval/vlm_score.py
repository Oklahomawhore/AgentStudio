import os
from openai import OpenAI
import dotenv
from util import capture_screenshots
import base64
import dashscope
from http import HTTPStatus
import glob

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
import hashlib
import requests

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

from util import prepare_message_for_vllm
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

def inference(video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels, "resized_width" : 320, "resized_height" : 240},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration  # Duration in seconds
    clip.close()
    return duration


def score_video(video_path, prompt=None):
    # Get length of video
    video_length = get_video_length(video_path)
    print("Video length (s) :", video_length)
    
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



if __name__ == '__main__':
    prompt= "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. Provide the result in json format with 'mm:ss.ff' format for time depiction."
    score_video("/data/wangshu/wangshu_code/ISG/ISV_eval/“来感受一下4K 60帧的丝滑”.mp4", prompt=prompt) # 这里的时间戳是视频的时间戳
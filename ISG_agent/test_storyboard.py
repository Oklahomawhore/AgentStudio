
from production import gen_video, concat_video
from util import capture_screenshots, download_video_and_save_as_mp4
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from moviepy import *
import uuid
from openai import OpenAI
import requests
import base64
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import dashscope
import dotenv

dotenv.load_dotenv()
from http import HTTPStatus
# dashscope sdk >= 1.22.1
from dashscope import VideoSynthesis

def sample_call_i2v(prompt,img, save_dir):
    # call sync api, will return the result
    print('please wait...')
    rsp = VideoSynthesis.call(model='wanx2.1-i2v-turbo',
                              prompt=prompt,
                              img_url=img)
    print(rsp)
    if rsp.status_code == HTTPStatus.OK:
        print(rsp.output.video_url)
        output, _ = download_video_and_save_as_mp4(rsp.output.video_url, seconds_per_screenshot=1, save_directory=save_dir)
    else:
        print('Failed, status_code: %s, code: %s, message: %s' %
              (rsp.status_code, rsp.code, rsp.message))
        output = None
    return output

# Modify OpenAI's API key and API base to use vLLM's API server.
import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    if content_url.startswith('http'):
        with requests.get(content_url) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode('utf-8')
    else:#local
        with open(content_url, 'rb') as f:
            result = base64.b64encode(f.read()).decode('utf-8')
    return result


# Video input inference
def run_video(model, prompt, video_url) -> str:
    video_base64 = encode_base64_content_from_url(video_url)

    screenshots = capture_screenshots(video_url,0.2)

    ## Use base64 encoded video in the payload
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "video",
                    "video": [f"data:image/jpeg;base64,{base64_image}" for base64_image in screenshots],
                },
            ],
        }],
        model=model,
        max_completion_tokens=64,
    )

    result = chat_completion_from_base64.choices[0].message.content
    print("Chat completion output from base64 encoded image:", result)
    return result


def prompt_vllm(vid_file):
    prompt = "Look at this video, describe the whole progression of video so that it can be recreated based on text descriptions."


    resp = run_video("qwen-vl-max-2025-01-25", prompt, vid_file)
    print("Completion result:", resp)
    return resp


task_dir = "results_test_storyboard_2"
raw_video = "/data/wangshu/wangshu_code/ISG/ISG_agent/results_test_storyboard/低俗小说-经典片段.mp4"

screeshots = capture_screenshots(raw_video, 1, end_time=25.0)
# Save screenshots to task directory
for i, screenshot in enumerate(screeshots):
    if i >= 25:
        break

    output_path = os.path.join(task_dir, f"screenshot_{i}.jpg")
    # Decode base64 string to bytes
    img_bytes = base64.b64decode(screenshot)
    # Write bytes to file
    with open(output_path, 'wb') as f:
        f.write(img_bytes)



raw_clip = VideoFileClip(raw_video)
video_clips = [raw_clip.subclipped(i, i + 1) for i in range(25)]

video_task = []
prompts = []
# #用视频理解模型分段caption
for idx, (vid, img) in enumerate(zip(video_clips, screeshots)):
    output_path = f"{task_dir}/video_clip_{idx}.mp4"
    vid.write_videofile(output_path)
    prompts.append(output_path)
    
# Get VLLM captions

caps = []
imgs = []
for i in range(25):
    img = os.path.join(task_dir, f"screenshot_{i}.jpg")
    vid_file = f"{task_dir}/video_clip_{i}.mp4"
    if os.path.exists(f"{task_dir}/video_clip_{i}.txt"):
        with open(f"{task_dir}/video_clip_{i}.txt", "r") as f:
            cap = f.read()
    else:
        cap = prompt_vllm(vid_file)
        with open(f"{task_dir}/video_clip_{i}.txt", "w") as f:
            f.write(cap)
    caps.append(cap)
    imgs.append(img)

for cap, img in zip(caps,imgs):
    video_task.append((cap,img))

vid_results = [
    # "results_test_storyboard/72fa896f-f845-405f-a180-12ab78d3866c.mp4"
    # "results_test_storyboard/6f581b80-6444-43a3-911d-bf39e7ecba6f.mp4",
    # "results_test_storyboard/c83b27a9-0332-42eb-a954-23a5b8b4ac27.mp4",
    # "results_test_storyboard/c50583cd-d6a2-4c75-a5c5-4a0e9545761b.mp4",
    # "results_test_storyboard/eef47b99-c74b-4d83-9d35-5b02dff6a2d8.mp4"
]
for cap, img in video_task:
    # print(img.split('/')[-1].split('.')[0].split('_')[-1])
    # if img.split('/')[-1].split('.')[0].split('_')[-1] != '0':
    #     continue
    start = time.time()
    vid = sample_call_i2v(cap, img, "results_test_storyboard_2")
    vid_results.append(vid)
#     vid_results.insert(0, vid)

# Load video clips
video_clips = [VideoFileClip(video) for video in vid_results]

# Concatenate video clips
final_video = concatenate_videoclips(video_clips, method="compose")
output_path = f"{task_dir}/final_video_{uuid.uuid4()}.mp4"
final_video.write_videofile(output_path)
end = time.time()
# print(f"Time taken: {end-start:.6f}")






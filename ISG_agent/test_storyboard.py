
from production import gen_video, concat_video
from util import capture_screenshots
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

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
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
# def run_video(model, prompt, video_url) -> str:
#     video_base64 = encode_base64_content_from_url(video_url)

#     ## Use base64 encoded video in the payload
#     chat_completion_from_base64 = client.chat.completions.create(
#         messages=[{
#             "role":
#             "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": prompt
#                 },
#                 {
#                     "type": "video_url",
#                     "video_url": {
#                         "url": f"data:video/mp4;base64,{video_base64}"
#                     },
#                 },
#             ],
#         }],
#         model=model,
#         max_completion_tokens=64,
#     )

#     result = chat_completion_from_base64.choices[0].message.content
#     print("Chat completion output from base64 encoded image:", result)
#     return result

def run_video(model, prompt, vid_file):
    MODEL_PATH = model

    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )


    # For video input, you can pass following values instead:
    # "type": "video",
    # "video": "<video URL>",
    video_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video", 
                    "video": vid_file,
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28
                }
            ]
        },
    ]

    # Here we use video messages as a demonstration
    messages = video_messages

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,

        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)
    return generated_text


def prompt_vllm(vid_file):
    prompt = "Look at this video, describe the whole progression of video so that it can be recreated based on text descriptions."


    completion = run_video("Qwen/Qwen2.5-VL-7B-Instruct", prompt, vid_file)
    print("Completion result:", completion)
    return completion.choices[0].text

raw_video = "/data/wangshu/wangshu_code/ISG/ISG_agent/results_test_storyboard/低俗小说-经典片段.mp4"

screeshots = capture_screenshots(raw_video, 5)

task_dir = "results_test_storyboard"

raw_clip = VideoFileClip(raw_video)
video_clips = [raw_clip.subclipped(i * 5.0, (i + 1) * 5.0) for i in range(5)]

video_task = []
prompts = []
#用视频理解模型分段caption
for idx, (vid, img) in enumerate(zip(video_clips, screeshots)):
    output_path = f"{task_dir}/video_clip_{idx}.mp4"
    vid.write_videofile(output_path)
    p = prompt_vllm(output_path)
    video_task.append((p, img))


with ThreadPoolExecutor() as executor:
    start = time.time()
    vid_results = executor.map(gen_video, *zip(*video_task))
    
    # Load video clips
    video_clips = [VideoFileClip(video[0]) for video in list(vid_results)]

    # Concatenate video clips
    final_video = concatenate_videoclips(video_clips, method="compose")
    output_path = f"{task_dir}/final_video_{uuid.uuid4()}.mp4"
    final_video.write_videofile(output_path)
    end = time.time()
    print(f"Time taken: {end-start:.6f}")






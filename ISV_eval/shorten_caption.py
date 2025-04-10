import json
import os
from openai import OpenAI
import dotenv

from util import inference_batch

dotenv.load_dotenv()
with open("/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results/“来感受一下4K 60帧的丝滑”/scenes.json", "r") as f:
    scenes = json.load(f)

batch_messages = []
for scene in scenes:
    messages = [
        {"role": "user", "content": f"given this detailed scene description: {scene['caption']}, give me a short version with only necessary details."}
    ]
    batch_messages.append(messages)
results = inference_batch(batch_messages, "qwen2.5-vl-32b-instruct", job_id="short_caption")

assert len(results) == len(scenes), f"results length {len(results)} does not match scenes length {len(scenes)}"
for i, scene in enumerate(scenes):
    if results[i] != None:
        scene["short_caption"] = results[i]
with open("/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results/“来感受一下4K 60帧的丝滑”/scenes_short.json", "w") as f:
    json.dump(scenes, f, indent=4, ensure_ascii=False)

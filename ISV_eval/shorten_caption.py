import json
import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
with open("/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results/“来感受一下4K 60帧的丝滑”/scenes.json", "r") as f:
    scenes = json.load(f)
client = OpenAI(
   api_key=os.getenv("KLING_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

for scene in scenes:
    messages = [
        {"role": "user", "content": f"given this detailed scene description: {scene['caption']}, give me a short version with only necessary details."}
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
    )
    print(completion.choices[0].message.content)
    scene['short_caption'] = completion.choices[0].message.content

with open("/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results/“来感受一下4K 60帧的丝滑”/scenes_short.json", "w") as f:
    json.dump(scenes, f, indent=4, ensure_ascii=False)

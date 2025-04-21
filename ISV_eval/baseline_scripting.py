from openai import OpenAI
import json
import os
from dotenv import load_dotenv
load_dotenv()
prompt = "Given this short story, rewrite it into multiple scene content description with explict description for camera angle, style and characters and their movements, each spanning around five seconds, marking scene number clearly with 1,2,3.... {}"

output_dir = "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/scripts"
os.makedirs(output_dir, exist_ok=True)
with open("/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json", "r") as f:
    stories = json.load(f)

client = OpenAI(api_key=os.getenv("KLING_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

rewrited_stories = []
for story in stories:
    task_id = story["id"]
    story = story["Query"]
    for data in story:
        if data['type'] == "text":
            story_text = data['content']
    print(story_text)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.format(story_text)}
        ],
        temperature=0,
        max_tokens=1000,
        n=1,
        stop=None
    )
    print(response.choices[0].message.content)
    with open(f"{output_dir}/{task_id}.txt", "w") as f:
        f.write(response.choices[0].message.content)

from openai import OpenAI
import dotenv
import os
import random
import json

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("KLING_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))

entities_base = ["cat", "ball", "computer", "boy", "girl", "train", "seagull", "turtle", "teddy bear", "plane"]
stories = []

for i in range(10):
    number = random.sample(range(2, 4), 1)
    print(number)
    entities = random.sample(entities_base, number[0])

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Write an interesting story for children about {' and '.join(['a' + entity for entity in entities])}."},
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
    )

    story = response.choices[0].message.content
    print(story)
    stories.append({
    "id": f"{i:04d}",
    "Category": "Video Storytelling",
    "Query": [
      {
        "type": "text",
        "content": story
      }
    ],
    "NumberOfCharacters" : len(entities),
    "Entities": entities,
  },)

with open("/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/GPT-story/stories.json", "w") as f:
    json.dump(stories, f, indent=4)
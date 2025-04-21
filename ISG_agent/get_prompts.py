
import json
from util import replace_characters_in_content

for task in range(11):
    task_id = str(task).zfill(4)
    print(task_id)
    with open(f"/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning_run2/Task_{task_id}/characters.json", "r") as f:
        characters = json.load(f)

    with open(f"/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning_run2/Task_{task_id}/video_prompt.json", "r") as f:
        video_prompt = json.load(f)


    new_prompts  = []
    for scene, prompt in video_prompt.items():
        print(prompt)
        new_prompt, character_list = replace_characters_in_content(prompt, characters)
        print(f"scene {scene} : {character_list}")
        new_prompts.append(new_prompt)

    with open(f"/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning_run2/Task_{task_id}/prompts.txt", "w") as f:
        for prompt in new_prompts:
            f.write(prompt + "\n")
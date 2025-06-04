import os
import json
from datasets import Dataset
from logging import Logger

logger = Logger(__name__)

plan_template = json.load(open(os.path.join(os.path.dirname(__file__), "..", "ISG_agent/Prompt/plan_template_singleround.json"), 'r', encoding='utf-8'))
system_prompt = "You are a creative and storytelling-driven video generation agent. " \
"Your responsibility is to design the complete workflow for producing short-form video content — " \
"from concept planning, script writing, casting design, to detailed storyboarding."


def vgen_dataset(input_json, args) -> Dataset:
    # Load vgen dataset from a JSON file
    with open(input_json, 'r') as f:
        tasks = json.load(f)
    ds_dict = {"prompt": [], "task": []}
    for task in tasks:
        
        task_text = task["Query"][0]["content"]
        for step, (step_name, step_prompt) in enumerate(plan_template.items()):
            pass
        
        # Append the prompt to the dataset
        ds_dict["prompt"].append(f"{step_name}: {step_prompt} \n\n {task_text}")
        ds_dict["task"].append(task)
    # convert to dataset
    ds = Dataset.from_dict(ds_dict)
    return ds

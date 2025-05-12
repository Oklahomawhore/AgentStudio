import glob
import json
import os
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="The input directory containing the JSON files.",
)
args = parser.parse_args()
input_dir = args.input_dir

json_list = glob.glob(os.path.join(input_dir, "dry_run_step_*.json"))

task_best_prompt = defaultdict(list)


for json_file in json_list:
    with open(json_file, "r") as f:
        data = json.load(f)
        prompt_response = data["query_responses"]
        task_query_responses = defaultdict(list)
        for prompt in prompt_response:
            task = prompt["task_id"]
            task_query_responses[task].append({"query" : prompt["query"], "response": prompt["response"]})
        for task, reward in zip(data["tasks"], data["rewards"]):
            if task not in task_best_prompt:
                if reward  > 0:
                    for i in range(len(task_query_responses[task])):
                        task_best_prompt[task].append({
                            "instruction": task_query_responses[task][i]["query"],
                            "output": task_query_responses[task][i]["response"],
                            "best_reward": reward,
                        })
                    print(f"best reward for task {task} is {reward}")
            else:
                if reward > task_best_prompt[task][0]["best_reward"]:
                    task_best_prompt[task].clear()
                    for i in range(len(task_query_responses[task])):
                        task_best_prompt[task].append({
                            "instruction": task_query_responses[task][i]["query"],
                            "output": task_query_responses[task][i]["response"],
                            "best_reward": reward,
                        })
                    print(f"best reward for task {task} is {reward}")
alpaca_json = []
for task, prompt in task_best_prompt.items():
    for p in prompt:
        alpaca_json.append({
            "instruction": p["instruction"],
            "input" : "",
            "output": p["output"],
        })
with open(os.path.join(args.input_dir, "agent_alpaca.json"), "w") as f:
    json.dump(alpaca_json, f, indent=4, ensure_ascii=False)
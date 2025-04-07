file = "/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results/“来感受一下4K 60帧的丝滑”/agent_log/questions_asked.json"

import json
with open(file, "r") as f:
    questions = json.load(f)

for question, answers in questions.items():
    for answer in answers:
        for name, ans in answer.items():
            if isinstance(ans, dict):
                print(ans)
                ans = ans["choices"][0]["message"]["content"][0]["text"]
                answer[name] = ans

with open("/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results/“来感受一下4K 60帧的丝滑”/agent_log/questions_asked_fix.json", "w") as f:
    json.dump(questions, f, indent=4, ensure_ascii=False)


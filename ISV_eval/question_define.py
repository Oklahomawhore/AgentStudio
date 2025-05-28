from openai import OpenAI
import json
import os
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()

meta_prompt = """You are a professional director and screenwriter specializing in film and short videos. You are well-versed in original script adaptation and storyboard writing. Based on the following story, please come up with a series of questions that can only be correctly answered by someone who has perfectly recreated the video version of this story.

The question set should be 15 questions, 5 questions for each question type, and the questions should be diverse, covering various aspects of the story. The questions should be designed to assess the viewer's understanding of the story, including its plot, characters, and themes. The questions should also be designed to encourage critical thinking and analysis of the story.
The question types should include: 
fill-in-the-blank, yes/no, multiple choice.

IMPORTANT: Your questions can be infered easily from only looking at the video frames without access to the original story transcript.
examples for visually answerable questions: 
* what is the main character's intention? 
* where did the protagonist end up at last? 
* Does the setting include a beach or ocean?
* How many people are present in the courtroom scene?

Please organize the questions in a JSON object and provide the answers as well.

The JSON object should have the following structure:
{{
  "fill_in_the_blank": [
    {{
      "type": "fill-in-the-blank",
      "question": "The first protein cooked in the pot is __________.",
      "answer": "shrimp"
    }},
    {{
      "type": "fill-in-the-blank",
      "question": "The noodles are rinsed and drained after cooking in __________.",
      "answer": "water"
    }},
    {{
      "type": "fill-in-the-blank",
      "question": "The sauce includes peanut butter and __________ as sweeteners.",
      "answer": "sugar"
    }},
    {{
      "type": "fill-in-the-blank",
      "question": "The rice paper is dipped in __________ before assembling.",
      "answer": "water"
    }},
    {{
      "type": "fill-in-the-blank",
      "question": "The final roll contains vegetables, noodles, and __________.",
      "answer": "meat"
    }}
  ],
  "yes_no": [
    {{
      "type": "yes/no",
      "question": "Is beef used in this recipe?",
      "answer": "no"
    }},
    {{
      "type": "yes/no",
      "question": "Are the shrimp sliced before the pork?",
      "answer": "yes"
    }},
    {{
      "type": "yes/no",
      "question": "Is the sauce cooked in the same pot used for meats?",
      "answer": "no"
    }},
    {{
      "type": "yes/no",
      "question": "Are nuts used as a sauce garnish?",
      "answer": "yes"
    }},
    {{
      "type": "yes/no",
      "question": "Is the rice paper cooked over heat?",
      "answer": "no"
    }}
  ],
  "multiple_choice": [
    {{
      "type": "multiple-choice",
      "question": "How many different proteins are prepared?",
      "options": ["1", "2", "3", "4"],
      "answer": "2"
    }},
    {{
      "type": "multiple-choice",
      "question": "What is added to the pot first when making the sauce?",
      "options": ["Garlic", "Peanut butter", "Sugar", "Water"],
      "answer": "Garlic"
    }},
    {{
      "type": "multiple-choice",
      "question": "What is the final step in assembly?",
      "options": [
        "Dipping in sauce",
        "Adding vegetables",
        "Rolling tightly",
        "Placing rice paper"
      ],
      "answer": "Rolling tightly"
    }},
    {{
      "type": "multiple-choice",
      "question": "Which ingredient is NOT shown in the filling?",
      "options": ["Carrots", "Cucumber", "Lettuce", "Bell peppers"],
      "answer": "Bell peppers"
    }},
    {{
      "type": "multiple-choice",
      "question": "What color are the cooked shrimp?",
      "options": ["Blue", "Pink", "White", "Red"],
      "answer": "Pink"
    }}
  ]
}}

The story is {}

Some oracle video reviewer looked at the generated video and answered a few questions related to the story. \
Here are some questions the oracle answered correctly, you should avoid asking the same set of questions to raise your question answering difficulty.
The questions are:
{}
"""



OpenAIClient = OpenAI(
   api_key=os.getenv("OPENAI_MODEL_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

def extract_json_from_response(json_text):
    response_text = json_text.strip()
    start_index = response_text.find('{')
    if start_index == -1:
        print(response_text)
        raise ValueError("No valid JSON found in the response")

    # Find the position of the last closing bracket ']'
    open_brackets = 0
    end_index = start_index
    for idx in range(start_index, len(response_text)):
        if response_text[idx] == '{':
            open_brackets += 1
        elif response_text[idx] == '}':
            open_brackets -= 1
            if open_brackets == 0:
                end_index = idx + 1  # include the closing bracket
                break
    
    json_text = response_text[start_index:end_index]
    return json_text

def generate_task_questions(task, output_dir="/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/instance_questions_deepseek", prev_correct=None):
    os.makedirs(output_dir, exist_ok=True)
    task_id = task.get("id")
    for query in task.get("Query"):
        if query["type"] == "text":
            story = query["content"]
        if not os.path.exists(f"{output_dir}/{task_id}.json"):
            response = OpenAIClient.chat.completions.create(
                model="doubao-1-5-thinking-vision-pro-250428",
                messages=[
                    {"role": "user", "content": meta_prompt.format(story, prev_correct)}
                ],
                temperature=0.7,
                max_tokens=4096,
                response_format={"type" : "json_object"}
            )
            questions = response.choices[0].message.content
            questions = json.loads(extract_json_from_response(questions))
            with open(f"{output_dir}/{task_id}.json", "w") as f:
                json.dump(questions, f, indent=4, ensure_ascii=False)

def generate_dataset_questions(input_json, prev_correct=None, output_dir=None):
    with open(input_json, "r") as f:
        tasks = json.load(f)
    for i, task in enumerate(tasks):
        task_id = task.get("id")
        generate_task_questions(task, prev_correct=prev_correct[task_id] if prev_correct and task_id in prev_correct else None, output_dir=output_dir)

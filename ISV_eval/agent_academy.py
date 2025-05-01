import os
import copy
import asyncio
import json
from typing import Tuple, List, Dict

from tqdm import tqdm
from openai import OpenAI
import dashscope
import dotenv
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import glob
from torch.utils.data import Dataset

from vlm_score import score_video, score_video_batch, shorten_caption
from movie_review_commission import MovieReviewCommission
from agents import AudienceAgent, CulturalExpertAgent, CriticAgent
from eval import calc_final_score
from human_in_loop import HumanAnnotator
from ISV_eval.util import format_time_to_hhmmss, cut_video_into_scenes, QUESTION_TEMPLATE
from prompt_datasets import EnhancedVideoStorytellingDataset, QuestionType, StoryQuestions

plt.rcParams['font.family'] = 'SimHei'

dotenv.load_dotenv()

OpenAIClient = OpenAI(
   api_key=os.getenv("KLING_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

async def process_questions(commission, questions, video_path, scenes, batch=False):
    tasks = []  # Store all async tasks

    for part, value in questions.items():
        for part_name, part_questions in value.items():
            if part_name == "detailed_shot_assessment":
                for scene in scenes:
                    context = {
                        'video': video_path,
                        'text': f"scene number: {scene['scene_number']} ({scene['start_time']}-{scene['end_time']}) scene description: {scene['short_caption']}",
                        # 'image': glob.glob(scene['path'].replace(".mp4", "screenshot_*")),
                    }
                    for question_type, question_list in part_questions.items():
                        if question_type == "yes_no":
                            tasks.extend([
                                asyncio.create_task(
                                    commission.do_questionare(
                                        question=f"scene {scene['scene_number']}, answer this question with yes or no: {question['question']}",
                                        context=context,
                                        batch=batch
                                    )
                                ) for question in question_list
                            ])
                        elif question_type == "scoring_1_10":
                            tasks.extend([
                                asyncio.create_task(
                                    commission.do_questionare(
                                        question=f"scene {scene['scene_number']}, answer this question with a score of 1-10: {question['question']}",
                                        context=context,
                                        batch=batch
                                    )
                                ) for question in question_list
                            ])
            else:
                context = {'video': video_path}
                for question_type, question_list in part_questions.items():
                    if question_type == "yes_no":
                        tasks.extend([
                            asyncio.create_task(
                                commission.do_questionare(
                                    question=f"answer this question with yes or no: {question['question']}",
                                    context=context,
                                    batch=batch
                                )
                            ) for question in question_list
                        ])
                    elif question_type == "scoring_1_10":
                        tasks.extend([
                            asyncio.create_task(
                                commission.do_questionare(
                                    question=f"answer this question with a score of 1-10: {question['question']}",
                                    context=context,
                                    batch=batch
                                )
                            ) for question in question_list
                        ])
            
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

def process_question_v3(questions: StoryQuestions, video_path: str) -> List[Tuple]:
    context = {"video" : video_path}
    tasks = []
    for question in questions.fill_in_questions:
        tasks.append((context, QUESTION_TEMPLATE.FILL_IN_BLANK.format(question.to_dict()['question'])))
    for question in questions.yes_no_questions:
        tasks.append((context, QUESTION_TEMPLATE.YES_NO.format(question.to_dict()['question'])))
    for question in questions.multiple_choice_questions:
        tasks.append((context, QUESTION_TEMPLATE.MULTIPLE_CHOICE.format(question.to_dict()['question'], question.to_dict()['options'])))
    return tasks

async def get_score(video_path, 
                    processed_questions, 
                    method='agent', 
                    threshold=30.0, 
                    min_scene_length=15, 
                    output_dir='eval_results', 
                    task_id=None, 
                    batch=False, 
                    local_model=None, 
                    local_processor=None):
    """
    Get the score of the video

    Args:
        video_path: the path of the video
        processed_questions: questions to ask for the particular video
        method: the method to get the score
        threshold: threshold for scene change detection
        min_scene_length: minimum length of a scene in frames

    Returns:
        the score of the video
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if task_id:
        output_dir = os.path.join(output_dir, task_id)
    else:
        output_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0])
    # Cut video into scenes
    # scene_data_path = os.path.join(output_dir, "scenes_short.json")
    # os.makedirs(os.path.dirname(scene_data_path), exist_ok=True)
    # if os.path.exists(scene_data_path):
    #     with open(scene_data_path, 'r') as f:
    #         scenes = json.load(f)
    # else:
    #     # No scene captions found, generate them
    #     scenes = cut_video_into_scenes(video_path, threshold, min_scene_length, output_dir, plot_results=True)
    #     # Save scene caption
    #     with open(scene_data_path, 'w') as f:
    #         json.dump(scenes, f, indent=2)

    # with open('caption.txt', 'r') as f:
    #     prompt = f.read()
    #     # Get scene caption
    # captions = score_video_batch([scene['path'] for scene in scenes], prompt=prompt)
    # for i, scene in enumerate(scenes):
    #     if 'caption' in scene:
    #         continue
        
    #     scene['caption'] = captions[i]
    # results = shorten_caption(scenes)
    # for i, scene in enumerate(scenes):
    #     if results[i] is not None:
    #         scene['short_caption'] = results[i]
    # # Save scene caption
    # with open(scene_data_path, 'w') as f:
    #     json.dump(copy.deepcopy(scenes), f, indent=2, ensure_ascii=False)
    
    # 创建评审委员会成员
    if method == 'agent':
        critic = CriticAgent(
            specialty="艺术电影",
            critic_style="严苛",
            model="qwen-vl-max-latest",
            save_dir=os.path.join(output_dir, "agent_log")
        )

        cultural_expert = CulturalExpertAgent(
            cultural_background="中国文化",
            expertise_areas=["电影中的文化符号", "东西方文化比较"],
            model="qwen-vl-max-latest",
            save_dir=os.path.join(output_dir, "agent_log")
        )

        audience_rep = AudienceAgent(
            demographics={"age_group": "18-25岁", "occupation": "大学生", "education": "本科在读", "region": "二线城市"},
            model="qwen-vl-max-latest",
            save_dir=os.path.join(output_dir, "agent_log")
        )

        agents = [critic, cultural_expert, audience_rep]

        commission = MovieReviewCommission(agents=agents, video_path=video_path, save_path=os.path.join(output_dir, "agent_log"),local_model=local_model, local_processor=local_processor)
    
    elif method == 'human':
        commission = HumanAnnotator(name="wangshu", save_dir=os.path.join(output_dir, "human_annotations"), port=8001)
    # Load question template
    answer_path = os.path.join(output_dir, f"{commission}_{os.path.basename(video_path).split('.')[0]}.json")
    
    report_path = os.path.join(output_dir, f"{commission}_{os.path.basename(video_path).split('.')[0]}_report.json")
    
    
    # 处理问题
    tasks = []

    for context, question in processed_questions:
        tasks.append(asyncio.create_task(commission.do_questionare(question=question, context=context, batch=batch)))
    
    await asyncio.gather(*tasks)

    # await process_questions(commission, questions, video_path, scenes, batch=True)
    if batch:
        await commission.do_batch_questionare()
    # Save question results
    answer_list = commission.get_questionnaire_results()
    with open(answer_path, "w") as f:
        json.dump(answer_list, f, indent=2, ensure_ascii=False)
    # 计算最终分数
    # try:
    #     final_score = calc_final_score(question_path, answer_path)
    # except Exception as e:
    #     print(f"Error calculating final score: {e}")
    #     final_score = 5.0  # 默认中间分数
    
    # with open(report_path, "w") as f:
    #     json.dump(final_score, f, indent=2, ensure_ascii=False)
    # return final_score
    return answer_path


class PromptDataset:
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

async def get_score_for_task(results_dir, prompt_json, questions_dir, dataset_name='NovelConditionedVGen',model_name=None, method='agent', threshold=30.0, min_scene_length=15, output_dir='eval_results', batch=False):
    """
    Get the score of the video

    Args:
        video_path: the path of the video
        method: the method to get the score
        threshold: threshold for scene change detection
        min_scene_length: minimum length of a scene in frames

    Returns:
        the score of the video
    """
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, dataset_name, model_name or 'unknown_model')
    # Cut video into scenes
    # scene_data_path = os.path.join(output_dir, "scenes_short.json")
    # os.makedirs(os.path.dirname(scene_data_path), exist_ok=True)

    dataset = EnhancedVideoStorytellingDataset(
        json_path=prompt_json,
        questions_dir=questions_dir,
    )
    for i in range(len(dataset)):
        print("-*" * 20)
        print("*-" * 20)
        print(f"Processing task {i} for dataset {dataset_name}...")
        video_dir = os.path.join(results_dir, f"Task_{dataset[i]['id']}")
        video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
        questions = dataset.get_story_questions(i)
        processed_questions = process_question_v3(questions, video_paths[0])
        await get_score(video_paths[0], processed_questions, method=method, threshold=threshold, min_scene_length=min_scene_length, output_dir=output_dir, task_id=f"Task_{dataset[i]['id']}", batch=batch)
    return None

async def main():
    parser = ArgumentParser(description="Video Quality Assessment")
    parser.add_argument("--video_path", type=str, help="Path to the video file", default=None)
    parser.add_argument("--results-dir", type=str, help="Path to the results directory")
    parser.add_argument("--prompt-json", type=str, help="Path to the prompt JSON file")
    parser.add_argument("--questions-dir", type=str, help="Path to the questions directory")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset", default='NovelConditionedVGen')
    parser.add_argument("--model-name", type=str, help="Name of the model", default=None)
    parser.add_argument("--output-dir", type=str, help="Output directory for results", default='eval_results')
    parser.add_argument("--method", choices=['agent','human'], default='agent', help="Annotator type: agent or human")
    parser.add_argument("--batch", action='store_true', help="Use batch processing for questions")
    args = parser.parse_args()
    if args.video_path:
        score = await get_score(args.video_path, method=args.method, output_dir=args.output_dir,batch=args.batch)
        print(f"Video score: {score}")
    elif args.results_dir:
        score = await get_score_for_task(args, args.results_dir, args.prompt_json, args.questions_dir, dataset_name=args.dataset_name, model_name=args.model_name, method=args.method, output_dir=args.output_dir)
        print(f"Task score: {score}")
    else:
        print("Please provide either a video path or a results directory.")
        return
if __name__ == '__main__':
    asyncio.run(main())
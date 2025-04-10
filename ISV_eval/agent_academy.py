import os
import copy
import asyncio
import json

from tqdm import tqdm
from openai import OpenAI
import dashscope
import dotenv
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import glob

from vlm_score import score_video, score_video_batch, shorten_caption
from movie_review_commission import MovieReviewCommission
from agents import AudienceAgent, CulturalExpertAgent, CriticAgent
from eval import calc_final_score
from human_in_loop import HumanAnnotator
from util import format_time_to_hhmmss, cut_video_into_scenes

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

async def get_score(video_path, method='agent', threshold=30.0, min_scene_length=15, output_dir='eval_results'):
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
    output_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0])
    # Cut video into scenes
    scene_data_path = os.path.join(output_dir, "scenes_short.json")
    os.makedirs(os.path.dirname(scene_data_path), exist_ok=True)
    if os.path.exists(scene_data_path):
        with open(scene_data_path, 'r') as f:
            scenes = json.load(f)
    else:
        # No scene captions found, generate them
        scenes = cut_video_into_scenes(video_path, threshold, min_scene_length, output_dir, plot_results=True)
        # Save scene caption
        with open(scene_data_path, 'w') as f:
            json.dump(scenes, f, indent=2)

    with open('caption.txt', 'r') as f:
        prompt = f.read()
        # Get scene caption
    captions = score_video_batch([scene['path'] for scene in scenes], prompt=prompt)
    for i, scene in enumerate(scenes):
        if 'caption' in scene:
            continue
        
        scene['caption'] = captions[i]
    results = shorten_caption(scenes)
    for i, scene in enumerate(scenes):
        if results[i] is not None:
            scene['short_caption'] = results[i]
    # Save scene caption
    with open(scene_data_path, 'w') as f:
        json.dump(copy.deepcopy(scenes), f, indent=2, ensure_ascii=False)
    
    # 创建评审委员会成员
    if method == 'agent':
        critic = CriticAgent(
            specialty="艺术电影",
            critic_style="严苛",
            model="qwen-vl-max-latest"
        )

        cultural_expert = CulturalExpertAgent(
            cultural_background="中国文化",
            expertise_areas=["电影中的文化符号", "东西方文化比较"],
            model="qwen-vl-max-latest"
        )

        audience_rep = AudienceAgent(
            demographics={"age_group": "18-25岁", "occupation": "大学生", "education": "本科在读", "region": "二线城市"},
            model="qwen-vl-max-latest"
        )

        agents = [critic, cultural_expert, audience_rep]

        commission = MovieReviewCommission(agents=agents, video_path=video_path, save_path=os.path.join(output_dir, "agent_log"), scenes=scenes)
    elif method == 'human':
        commission = HumanAnnotator(name="wangshu", save_dir=os.path.join(output_dir, "human_annotations"), port=8001)
    # Load question template
    answer_path = os.path.join(output_dir, f"{commission}_{os.path.basename(video_path).split('.')[0]}.json")
    question_path = "/data/wangshu/wangshu_code/ISG/ISV_eval/questions.json"
    report_path = os.path.join(output_dir, f"{commission}_{os.path.basename(video_path).split('.')[0]}_report.json")
    with open(question_path, "r") as f:
        questions = json.load(f)
    
    
    await process_questions(commission, questions, video_path, scenes, batch=True)
    await commission.do_batch_questionare()
    # Save question results
    answer_list = commission.get_questionnaire_results()
    with open(answer_path, "w") as f:
        json.dump(answer_list, f, indent=2, ensure_ascii=False)
    # 计算最终分数
    try:
        final_score = calc_final_score(question_path, answer_path)
    except Exception as e:
        print(f"Error calculating final score: {e}")
        final_score = 5.0  # 默认中间分数
    
    with open(report_path, "w") as f:
        json.dump(final_score, f, indent=2, ensure_ascii=False)
    return final_score

async def main():
    parser = ArgumentParser(description="Video Quality Assessment")
    parser.add_argument("--video_path", type=str, help="Path to the video file", default='/data/wangshu/wangshu_code/ISG/ISV_eval/“来感受一下4K 60帧的丝滑”.mp4')
    parser.add_argument("--method", choices=['agent','human'], default='agent', help="Annotator type: agent or human")
    args = parser.parse_args()
    score = await get_score(args.video_path, method=args.method)
    print(f"Video score: {score}")
if __name__ == '__main__':
    asyncio.run(main())
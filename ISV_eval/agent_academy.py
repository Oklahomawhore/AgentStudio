import cv2
import numpy as np
import os
from tqdm import tqdm
from openai import OpenAI
import dashscope
import dotenv
from vlm_score import score_video
import matplotlib.pyplot as plt
import matplotlib
from movie_review_commission import MovieReviewCommission
from agents import AudienceAgent, CulturalExpertAgent, CriticAgent
import copy
import asyncio
import json

plt.rcParams['font.family'] = 'SimHei'

dotenv.load_dotenv()

OpenAIClient = OpenAI(
   api_key=os.getenv("KLING_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

def cut_video_into_scenes(video_path, threshold=10.0, min_scene_length=1, output_dir=None, plot_results=False):
    """
    Cut video into scenes based on pixel distance spikes between consecutive frames.
    
    Args:
        video_path (str): Path to the input video file
        threshold (float): Threshold for scene change detection (higher = fewer scenes)
        min_scene_length (int): Minimum number of frames for a scene
        output_dir (str, optional): Directory to save scene clips. If None, clips are not saved.
        plot_results (bool): Whether to generate and display a plot of frame differences
        
    Returns:
        list: A list of dictionaries containing scene information:
              - 'start_frame': starting frame number
              - 'end_frame': ending frame number
              - 'start_time': starting time in seconds
              - 'end_time': ending time in seconds
              - 'path': path to the saved clip (if output_dir is provided)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize variables for scene detection
    prev_frame = None
    frame_diffs = []
    avg_diffs = []
    thresholds = []
    frame_numbers = []
    scene_boundaries = [0]  # Start with the first frame
    
    # Process video frames to detect scene changes
    print(f"Analyzing video for scene changes: {video_path}")
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale and resize for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width//4, height//4))
        
        if prev_frame is not None:
            # Calculate mean absolute difference between frames
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
            frame_numbers.append(i+1)
            
            # Check for scene change
            if len(frame_diffs) > 1:
                # Use a moving average to smooth out noise
                avg_diff = sum(frame_diffs[-3:]) / min(len(frame_diffs), 3)
                avg_diffs.append(avg_diff)
                thresholds.append(threshold * avg_diff)
                
                # If the difference is above threshold and enough frames have passed since last scene
                if mean_diff > threshold and i - scene_boundaries[-1] >= min_scene_length:
                    scene_boundaries.append(i)
                    print(f"Scene change detected at frame {i} (time: {i/fps:.2f}s)")
        
        prev_frame = gray
    
    # Add the last frame as a scene boundary
    if scene_boundaries[-1] != frame_count - 1:
        scene_boundaries.append(frame_count - 1)
    
    # Plot results if requested
    if plot_results:
        plt.figure(figsize=(15, 8))
        
        # Plot frame differences
        plt.plot(frame_numbers, frame_diffs, 'b-', alpha=0.6, linewidth=1, label='帧间差异')
        
        # Plot moving average if we have enough frames
        if len(avg_diffs) > 0:
            # We need to align the x-axis for avg_diffs (starts at frame 3)
            avg_diff_frames = frame_numbers[1:len(avg_diffs)+2]
            plt.plot(avg_diff_frames, avg_diffs, 'g-', linewidth=2, label='滑动平均 (3帧)')
            plt.plot(avg_diff_frames, thresholds, 'r--', linewidth=2, label=f'阈值 (x{threshold})')
        
        # Mark scene boundaries
        for boundary in scene_boundaries:
            if boundary > 0 and boundary < frame_count - 1:  # Skip first and last
                plt.axvline(x=boundary, color='red', linestyle='-', alpha=0.5)
                plt.text(boundary, plt.ylim()[1]*0.9, f"{boundary}\n({boundary/fps:.1f}s)", 
                         horizontalalignment='center', color='red', fontsize=8)
        
        # Add labels and title
        plt.xlabel('帧编号')
        plt.ylabel('平均像素差异')
        plt.title(f'场景检测分析 - {os.path.basename(video_path)}\n检测到 {len(scene_boundaries)-1} 个场景')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        # Save plot if output directory exists
        if output_dir:
            plot_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + "_scene_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"分析图表已保存至: {plot_path}")
        
        plt.tight_layout()
        plt.savefig('scene_analysis.png')
    
    # Create scenes list
    scenes = []
    video_name = os.path.basename(video_path).split('.')[0]
    
    for i in range(len(scene_boundaries) - 1):
        start_frame = scene_boundaries[i]
        end_frame = scene_boundaries[i+1]
        
        scene_info = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_frame / fps,
            'end_time': end_frame / fps,
        }
        
        # Save scene clip if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            scene_path = os.path.join(output_dir, f"{video_name}_scene_{i+1:03d}.mp4")
            
            # Extract and save the scene
            extract_scene(video_path, scene_path, scene_info['start_time'], scene_info['end_time'])
            scene_info['path'] = scene_path
        else:
            scene_info['path'] = None
        scene_info['duration'] = scene_info['end_time'] - scene_info['start_time']
        scene_info['scene_number'] = i + 1
        
        scenes.append(scene_info)
    
    cap.release()
    print(f"检测到 {len(scenes)} 个场景")
    return scenes

def extract_scene(input_video, output_path, start_time, end_time):
    """
    Extract a scene from the video and save it to a file.
    
    Args:
        input_video (str): Path to the input video
        output_path (str): Path to save the output clip
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
    """
    duration = end_time - start_time
    os.system(f'ffmpeg -i "{input_video}" -ss {start_time} -t {duration} -c:v libx264 -c:a aac -strict experimental -b:a 128k "{output_path}" -y -loglevel error')
    return output_path

async def process_scene_questions(scene, questions, agents):
    """
    让所有智能体回答关于特定场景的问题并记录答案
    
    Args:
        scene: 包含场景信息和字幕的字典
        questions: 问卷问题字典
        agents: 智能体列表
    
    Returns:
        更新后的问题字典，包含所有智能体的回答
    """
    scene_caption = scene['caption']
    scene_number = scene['scene_number']
    
    # 创建问题副本以存储回答
    scene_questions = copy.deepcopy(questions)
    
    print(f"处理场景 {scene_number} 的问题...")
    
    # 为每个智能体处理问题
    for agent in agents:
        agent_name = agent.characteristics.name
        print(f"  智能体 {agent_name} 正在回答问题...")
        
        # 处理问卷的每个部分
        for part_key, part_value in questions.items():
            for section_key, section_value in part_value.items():
                for question_type, questions_list in section_value.items():
                    # 构建提示词
                    prompt = f"""
                    你将评估一个电影场景。以下是场景描述:
                    
                    === 场景 #{scene_number} ===
                    {scene_caption}
                    
                    请基于以上场景描述，回答以下问题:
                    """
                    
                    # 根据问题类型构建不同的提示
                    if question_type == "yes_no" or question_type == "global_impact":
                        prompt += "\n请回答以下问题，只需回答'是'或'否':\n"
                        
                        for i, question in enumerate(questions_list):
                            prompt += f"{i+1}. {question['question']}\n"
                            
                        prompt += "\n请以JSON数组格式返回答案，格式为: [\"是\", \"否\", ...]"
                        
                    elif question_type == "scoring_1_10":
                        prompt += "\n请为以下问题打分（1-10分，1分最低，10分最高）:\n"
                        
                        for i, question in enumerate(questions_list):
                            prompt += f"{i+1}. {question['question']}\n"
                            
                        prompt += "\n请以JSON数组格式返回分数，格式为: [5, 7, ...]"
                    
                    # 向智能体发送提示并获取回答
                    response = await agent._respond_to_user(prompt)
                    
                    try:
                        # 尝试解析JSON格式的回答
                        answers = json.loads(response)
                        
                        # 更新问题的回答
                        for i, answer in enumerate(answers):
                            if i < len(questions_list):
                                # 根据问题类型更新不同字段
                                if question_type == "yes_no" or question_type == "global_impact":
                                    if f"answer_{agent_name}" not in scene_questions[part_key][section_key][question_type][i]:
                                        scene_questions[part_key][section_key][question_type][i][f"answer_{agent_name}"] = answer
                                elif question_type == "scoring_1_10":
                                    if f"score_{agent_name}" not in scene_questions[part_key][section_key][question_type][i]:
                                        scene_questions[part_key][section_key][question_type][i][f"score_{agent_name}"] = answer
                    
                    except Exception as e:
                        print(f"      解析回答时出错: {e}")
                        # 记录原始响应
                        for i in range(len(questions_list)):
                            suffix = f"_{agent_name}"
                            if question_type == "yes_no" or question_type == "global_impact":
                                scene_questions[part_key][section_key][question_type][i][f"answer{suffix}_error"] = str(e)
                                scene_questions[part_key][section_key][question_type][i][f"answer{suffix}_raw"] = response
                            elif question_type == "scoring_1_10":
                                scene_questions[part_key][section_key][question_type][i][f"score{suffix}_error"] = str(e)
                                scene_questions[part_key][section_key][question_type][i][f"score{suffix}_raw"] = response
    
    return scene_questions

async def consolidate_answers(scene_questions, agents):
    """
    整合不同智能体的回答，形成最终的答案
    
    Args:
        scene_questions: 包含所有智能体回答的问题字典
        agents: 智能体列表
    
    Returns:
        更新后的问题字典，包含合并后的答案
    """
    final_questions = copy.deepcopy(scene_questions)
    
    print("整合智能体回答...")
    
    # 获取所有智能体名称
    agent_names = [agent.characteristics.name for agent in agents]
    
    # 处理问卷的每个部分
    for part_key, part_value in scene_questions.items():
        for section_key, section_value in part_value.items():
            for question_type, questions_list in section_value.items():
                for i, question in enumerate(questions_list):
                    # 根据问题类型处理不同字段
                    if question_type == "yes_no" or question_type == "global_impact":
                        # 收集所有智能体的回答
                        answers = []
                        for agent_name in agent_names:
                            answer_key = f"answer_{agent_name}"
                            if answer_key in question:
                                answers.append(question[answer_key])
                        
                        # 如果有足够的回答，进行投票
                        if answers:
                            # 计算"是"的票数
                            yes_votes = answers.count("是")
                            # 如果"是"的票数超过半数，结果为"是"
                            final_answer = "是" if yes_votes > len(answers) / 2 else "否"
                            final_questions[part_key][section_key][question_type][i]["answer"] = final_answer
                    
                    elif question_type == "scoring_1_10":
                        # 收集所有智能体的评分
                        scores = []
                        for agent_name in agent_names:
                            score_key = f"score_{agent_name}"
                            if score_key in question and question[score_key] is not None:
                                try:
                                    scores.append(int(question[score_key]))
                                except (ValueError, TypeError):
                                    # 跳过无效的分数
                                    pass
                        
                        # 如果有足够的评分，计算平均分
                        if scores:
                            avg_score = sum(scores) / len(scores)
                            final_questions[part_key][section_key][question_type][i]["score"] = round(avg_score, 1)
    
    return final_questions

async def get_score(video_path, method='', threshold=30.0, min_scene_length=15):
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
    # Cut video into scenes
    output_dir = os.path.join(os.path.dirname(video_path), "tmp")
    scene_data_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + "_scenes.json")
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
    for scene in tqdm(scenes, desc="captioning scenes"):
        if 'caption' in scene:
            continue
        caption = score_video(scene['path'], prompt=prompt)
        scene['caption'] = caption

        # Save scene caption
        with open(scene_data_path, 'w') as f:
            json.dump(copy.deepcopy(scenes), f, indent=2)
    
    # 创建评审委员会成员
    critic = CriticAgent(
        specialty="艺术电影",
        critic_style="严苛"
    )
    
    cultural_expert = CulturalExpertAgent(
        cultural_background="中国文化",
        expertise_areas=["电影中的文化符号", "东西方文化比较"]
    )
    
    audience_rep = AudienceAgent(
        demographics={"age_group": "18-25岁", "occupation": "大学生", "education": "本科在读", "region": "二线城市"}
    )

    agents = [critic, cultural_expert, audience_rep]

    # 加载问题
    with open("/data/wangshu/wangshu_code/ISG/ISV_eval/questions.json", "r") as f:
        questions = json.load(f)
    
    # 对每个场景进行问答
    scene_questions_map = {}
    for scene in scenes:
        scene_questions = await process_scene_questions(scene, questions, agents)
        scene_questions_map[scene['scene_number']] = scene_questions
    
    # 合并场景问题的回答
    all_scene_answers = {}
    for scene_num, scene_questions in scene_questions_map.items():
        consolidated_questions = await consolidate_answers(scene_questions, agents)
        all_scene_answers[scene_num] = consolidated_questions
    
    # 创建评审委员会
    commission = MovieReviewCommission(agents)
    
    # 进行场景讨论
    discussions = []
    print("开始场景讨论...")
    for scene in scenes:
        scene_num = scene['scene_number']
        scene_caption = scene['caption']
        scene_questions = all_scene_answers.get(scene_num, {})
        
        # 构建讨论话题
        topics = [
            f"场景 #{scene_num} 的叙事贡献",
            f"场景 #{scene_num} 的视觉美学",
            f"场景 #{scene_num} 的整体效果"
        ]
        
        # 进行讨论
        scene_discussions = await commission.conduct_discussion(f"场景 #{scene_num}", topics)
        discussions.append({
            "scene_number": scene_num,
            "discussions": scene_discussions
        })
    
    # 基于所有场景的回答和讨论进行投票
    print("开始最终评分...")
    voting_criteria = [
        "全局叙事连贯性",
        "视觉美学",
        "声音设计",
        "演员表演",
        "导演选择",
        "技术执行"
    ]
    
    votes = await commission.vote_on_film(os.path.basename(video_path), voting_criteria)
    
    # 生成最终报告
    film_info = {
        "title": os.path.basename(video_path),
        "scenes_count": len(scenes),
        "total_duration": sum(scene['duration'] for scene in scenes)
    }
    
    final_report = await commission.generate_final_report(os.path.basename(video_path), film_info)
    
    # 保存最终结果
    final_result = {
        "video_path": video_path,
        "scenes": scenes,
        "scene_answers": all_scene_answers,
        "discussions": discussions,
        "votes": votes,
        "final_report": final_report
    }
    
    result_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + "_result.json")
    with open(result_path, 'w') as f:
        json.dump(final_result, f, indent=2)
    
    # 计算最终分数
    try:
        final_score = final_report.get("final_rating", 0.0)
    except:
        final_score = 5.0  # 默认中间分数
    
    return final_score
async def main():
    video_path = '/data/wangshu/wangshu_code/ISG/ISV_eval/“来感受一下4K 60帧的丝滑”.mp4'
    score = await get_score(video_path)
    print(f"Video score: {score}")
if __name__ == '__main__':
    asyncio.run(main())
import json
import os
import re
from typing import Dict, List, Any, Union, Optional

from openai import OpenAI
import dotenv

dotenv.load_dotenv()

client = OpenAI(
   api_key=os.getenv("KLING_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

def calc_final_score(questions_json_path: str, answers_json_path: str) -> Dict:
    """
    计算电影评估的最终得分
    
    参数:
        questions_json_path: 问题模板文件路径
        answers_json_path: 回答文件路径
        
    返回:
        包含各个维度得分的字典
    """
    # 加载问题模板和回答
    with open(questions_json_path, 'r', encoding='utf-8') as f:
        questions_template = json.load(f)
    
    with open(answers_json_path, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    
    # 创建结果字典
    results = {
        "part_1": {"score": 0, "max_score": 0},
        "part_2": {"score": 0, "max_score": 0},
        "part_3": {"score": 0, "max_score": 0},
        "part_4": {"score": 0, "max_score": 0},
        "part_5": {"score": 0, "max_score": 0},
        "overall": {"score": 0, "max_score": 0}
    }
    
    # 填充答案并计算得分
    filled_template = fill_answers(questions_template, answers)
    
    # 计算每个部分的得分
    for part_key, part_value in filled_template.items():
        if part_key.startswith("part_"):
            part_score, part_max = calculate_part_score(part_value)
            results[part_key]["score"] = part_score
            results[part_key]["max_score"] = part_max
            results["overall"]["score"] += part_score
            results["overall"]["max_score"] += part_max
    
    # 计算最终百分比得分
    for key in results:
        if results[key]["max_score"] > 0:
            results[key]["percentage"] = (results[key]["score"] / results[key]["max_score"]) * 100
        else:
            results[key]["percentage"] = 0
    
    return results

def fill_answers(template: Dict, answers: Dict) -> Dict:
    """填充问题模板中的答案"""
    filled_template = template.copy()
    
    # 遍历所有部分和问题
    for part_key, part_value in template.items():
        if not part_key.startswith("part_"):
            continue
            
        for category_key, category_value in part_value.items():
            for question_type, questions in category_value.items():
                for i, question_obj in enumerate(questions):
                    question_text = question_obj["question"]
                    
                    # 处理多场景问题 (检查答案中是否有scene开头的键)
                    scene_keys = get_scene_keys(question_text, answers)
                    if scene_keys:
                        # 初始化场景答案字典
                        scene_answers = {}
                        
                        # 填充每个场景的答案
                        for scene_key in scene_keys:
                            scene_match = re.search(r"scene\s*(\d+)", scene_key.lower())
                            if scene_match:
                                scene_num = f"scene {scene_match.group(1)}"
                                if question_type == "yes_no":
                                    scene_answers[scene_num] = process_yes_no_answers(answers[scene_key])
                                elif question_type == "scoring_1_10":
                                    scene_answers[scene_num] = process_numeric_answers(answers[scene_key])
                        
                        # 将场景答案字典分配给问题对象
                        if scene_answers:
                            filled_template[part_key][category_key][question_type][i]["scenes"] = scene_answers
                    else:
                        # 处理常规问题
                        answer_key = get_matching_answer_key(question_text, answers)
                        if answer_key:
                            if question_type == "yes_no":
                                filled_template[part_key][category_key][question_type][i]["answer"] = process_yes_no_answers(answers[answer_key])
                            elif question_type == "scoring_1_10":
                                filled_template[part_key][category_key][question_type][i]["score"] = process_numeric_answers(answers[answer_key])
    
    return filled_template

def get_scene_keys(question_text: str, answers: Dict) -> List[str]:
    """查找与问题相关的场景答案键"""
    scene_keys = []
    for key in answers:
        possible_scene_num = key.split(",")[0]
        if "scene" in possible_scene_num.lower() and question_text in key:
            scene_keys.append(key)
    return scene_keys

def get_matching_answer_key(question_text: str, answers: Dict) -> Optional[str]:
    """查找与问题文本匹配的答案键"""
    # 首先尝试直接匹配
    for key in answers:
        if question_text in key:
            return key
    
    # 尝试忽略大小写和额外文本匹配
    question_lower = question_text.lower()
    for key in answers:
        if question_lower in key.lower():
            return key
        
        # 处理可能有前缀的问题 (如 "Look at the shot screenshot and the video, answer...")
        if "answer this question" in key.lower() and question_lower in key.lower():
            return key
    
    return None

def process_yes_no_answers(answers_list: List[Dict]) -> str:
    """处理是/否类型的回答，返回多数票结果"""
    yes_count = 0
    no_count = 0
    
    for answer_dict in answers_list:
        for _, answer in answer_dict.items():
            answer_text = answer.lower().strip()
            if answer_text.startswith("yes"):
                yes_count += 1
            elif answer_text.startswith("no"):
                no_count += 1
            else:
                # prompt LLM to decide
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": f"Given this answer from agent: {answer_text}, is it a yes or no answer? Please respond with 'yes' or 'no'. If undecidable, respond with 'undefined'."}
                    ]
                )
                res = completion.choices[0].message.content.strip().lower()
                if res.startswith("yes"):
                    yes_count += 1
                elif res.startswith("no"):
                    no_count += 1
                else:
                    continue
    
    if yes_count > no_count:
        return "yes"
    else:
        return "no"

def process_numeric_answers(answers_list: List[Dict]) -> float:
    """处理数值类型的回答，提取数字并计算平均值"""
    scores = []
    
    for answer_dict in answers_list:
        for _, answer in answer_dict.items():
            # 使用正则表达式提取数字
            match = re.search(r'(\d+)(?:[./]\d+)?', answer)
            if match:
                try:
                    score = float(match.group(1))
                    if 1 <= score <= 10:  # 确保分数在有效范围内
                        scores.append(score)
                except ValueError:
                    continue
            else:
                print(f"无法从回答中提取分数: {answer}")
    if scores:
        return sum(scores) / len(scores)
    return 0

def calculate_part_score(part_data: Dict) -> tuple:
    """计算部分得分和最大可能得分"""
    total_score = 0
    max_possible = 0
    
    for category_key, category_value in part_data.items():
        # 处理是/否问题
        if "yes_no" in category_value:
            for question in category_value["yes_no"]:
                # 处理多场景问题
                if "scenes" in question:
                    scene_score = 0
                    scene_count = len(question["scenes"])
                    
                    # 计算所有场景的总分
                    for scene_key, scene_answer in question["scenes"].items():
                        if scene_answer == "yes":
                            scene_score += 10
                    
                    # 标准化场景得分
                    normalized_score = scene_score / scene_count if scene_count > 0 else 0
                    total_score += normalized_score
                    max_possible += 10  # 每个问题最高10分
                # 处理常规是/否问题
                elif "answer" in question and question["answer"] is not None:
                    max_possible += 10
                    if question["answer"] == "yes":
                        total_score += 10
        
        # 处理1-10评分问题
        if "scoring_1_10" in category_value:
            for question in category_value["scoring_1_10"]:
                # 处理多场景评分问题
                if "scenes" in question:
                    scene_score = 0
                    scene_count = len(question["scenes"])
                    
                    # 计算所有场景的总分
                    for scene_key, scene_score_value in question["scenes"].items():
                        scene_score += scene_score_value
                    
                    # 标准化场景得分
                    normalized_score = scene_score / scene_count if scene_count > 0 else 0
                    total_score += normalized_score
                    max_possible += 10  # 每个问题最高10分
                # 处理常规评分问题
                elif "score" in question and question["score"] is not None:
                    max_possible += 10  # 最高10分
                    total_score += question["score"]
    
    return total_score, max_possible

def evaluate_all_movies(questions_path: str, results_dir: str, output_path: str) -> None:
    """评估所有电影并生成综合报告"""
    all_results = {}
    
    # 获取所有回答文件
    answer_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    for answer_file in answer_files:
        movie_name = os.path.splitext(answer_file)[0]
        answer_path = os.path.join(results_dir, answer_file)
        
        try:
            # 计算该电影的得分
            movie_results = calc_final_score(questions_path, answer_path)
            all_results[movie_name] = movie_results
            print(f"已评估: {movie_name}")
        except Exception as e:
            print(f"评估 {movie_name} 时出错: {str(e)}")
    
    # 保存综合报告
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"综合评估报告已保存至: {output_path}")

if __name__ == "__main__":
    # 示例使用
    questions_path = "./questions.json"
    results_dir = "./eval_results"
    output_path = "./evaluation_report.json"
    
    # 确保结果目录存在
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
    else:
        evaluate_all_movies(questions_path, results_dir, output_path)
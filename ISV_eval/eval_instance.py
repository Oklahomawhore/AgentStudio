import json
import os
import re
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import glob

from openai import OpenAI
from dotenv import load_dotenv

from util import QUESTION_TEMPLATE

load_dotenv()

client= OpenAI(api_key=os.getenv("KLING_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"))

class EvaluationMetrics:
    """用于计算评估指标的类"""
    
    @staticmethod
    def calculate_accuracy(correct: int, total: int) -> float:
        """计算准确率"""
        if total == 0:
            return 0.0
        return correct / total
    
    @staticmethod
    def calculate_aggregated_score(scores: Dict[str, float]) -> float:
        """计算多个指标的加权平均分数"""
        weights = {
            "fill_in_blank": 0.3,
            "yes_no": 0.3,
            "multiple_choice": 0.4
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric_type, score in scores.items():
            if metric_type in weights:
                weighted_sum += score * weights[metric_type]
                total_weight += weights[metric_type]
        
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight


class AnswerProcessor:
    """处理不同类型问题的答案"""
    
    @staticmethod
    def standardize_yes_no_answer(answer: str) -> str:
        """将是非题答案标准化为'Yes'或'No'"""
        answer = answer.strip('.').lower()
        if answer in ["yes", "y", "true", "correct", "是", "对", "正确"]:
            return "Yes"
        elif answer in ["no", "n", "false", "incorrect", "否", "不", "错", "不正确"]:
            return "No"
        return "Ambiguous"
    
    @staticmethod
    def standardize_multiple_choice_answer(answer: str) -> str:
        """从多选题答案中提取选项字母"""
        # 尝试直接匹配选项字母
        match = re.search(r'\b([A-D])\b', answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 查找包含"答案是"、"选择"等关键词后面的字母
        patterns = [
            r'(?:answer|选择|答案|选项)\s*(?:is|为|是)?\s*[：:]*\s*([A-D])',
            r'([A-D])\s*[.．。]',
            r'选\s*([A-D])'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return "Ambiguous"
    
    @staticmethod
    def standardize_fill_in_blank_answer(answer: str, ref_answer: str) -> str:
        """标准化填空题答案"""
        # 移除可能的标点符号和空白
        cleaned_answer = re.sub(r'[^\w\s]', '', answer).strip().lower()
        cleaned_ref = re.sub(r'[^\w\s]', '', ref_answer).strip().lower()
        
        # 如果答案完全匹配，直接返回
        if cleaned_answer == cleaned_ref:
            return ref_answer
        
        # 检查答案是否包含参考答案
        if cleaned_ref in cleaned_answer:
            return ref_answer
            
        return answer  # 返回原始答案以便后续处理


class ResponseEvaluator:
    """评估LLM响应的类"""
    
    def __init__(self, results_path: str, questions_dir: str, reprompt_llm=None):
        """
        初始化评估器
        
        Args:
            results_path: 结果JSON文件路径
            questions_dir: 问题JSON文件目录
            reprompt_llm: 用于重新提问的LLM函数(可选)
        """
        self.results_path = results_path
        self.questions_dir = questions_dir
        self.reprompt_llm = reprompt_llm
        
        # 加载结果和问题
        self.results = self._load_json(results_path)
        self.task_id = os.path.basename(os.path.dirname(results_path)).split('_')[-1]
        self.questions = self._load_questions()
        
        # 评估状态和结果
        self.evaluated_answers = {
            "fill_in_blank": [],
            "yes_no": [],
            "multiple_choice": []
        }
        self.scores = {
            "fill_in_blank": 0.0,
            "yes_no": 0.0,
            "multiple_choice": 0.0,
            "aggregate": 0.0
        }
    
    def _load_json(self, path: str) -> Dict:
        """加载JSON文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {path}: {e}")
            return {}
    
    def _load_questions(self) -> Optional[Dict]:
        """加载与结果相关的问题文件"""
        question_path = os.path.join(self.questions_dir, f"{self.task_id}.json")
        if os.path.exists(question_path):
            return self._load_json(question_path)
        else:
            print(f"Warning: No question file found for task {self.task_id}")
            return None
    
    def _get_agent_answers(self, question: str) -> List[str]:
        """从结果中获取所有代理对特定问题的回答"""
        if question not in self.results:
            return []
        
        answers = []
        for agent_response in self.results[question]:
            for agent, response in agent_response.items():
                answers.append(response)
        
        return answers
    
    def _evaluate_single_answer(self, answer: str, question_type: str, reference: Dict) -> Tuple[bool, str]:
        """评估单个答案"""
        if question_type == "yes_no":
            std_answer = AnswerProcessor.standardize_yes_no_answer(answer)
            if std_answer == "Ambiguous":
                if self.reprompt_llm:
                    # 重新询问LLM以消除歧义
                    reprompt = f"Based on the following response, is the answer 'Yes' or 'No'?\nResponse: {answer}\nAnswer only with 'Yes' or 'No'."
                    std_answer = self.reprompt_llm(reprompt)
                    std_answer = AnswerProcessor.standardize_yes_no_answer(std_answer)
            
            return std_answer.lower() == reference["answer"].lower(), std_answer
            
        elif question_type == "multiple_choice":
            std_answer = AnswerProcessor.standardize_multiple_choice_answer(answer)
            if std_answer == "Ambiguous":
                if self.reprompt_llm:
                    # 重新询问LLM以消除歧义
                    if isinstance(reference["options"], list):
                        options_str = ", ".join(reference["options"])
                    else: 
                        options_str = ", ".join([f"{k}: {v}" for k, v in reference["options"].items()])
                    reprompt = f"Based on the following response, which option (A, B, C, or D) is selected?\nOptions: {options_str}\nResponse: {answer}\nAnswer only with the letter (A, B, C, or D)."
                    std_answer = self.reprompt_llm(reprompt)
                    std_answer = AnswerProcessor.standardize_multiple_choice_answer(std_answer)
            
            return std_answer == reference["answer"], std_answer
            
        elif question_type == "fill_in_blank":
            std_answer = AnswerProcessor.standardize_fill_in_blank_answer(answer, reference["answer"])
            if std_answer != reference["answer"] and self.reprompt_llm:
                # 对于填空题，请求LLM评估答案是否等效
                reprompt = f"Are these two answers equivalent in meaning?\nAnswer 1: {std_answer}\nAnswer 2: {reference['answer']}\nRespond only with 'Yes' or 'No'."
                evaluation = self.reprompt_llm(reprompt)
                if AnswerProcessor.standardize_yes_no_answer(evaluation) == "Yes":
                    return True, std_answer
            
            # 简单字符串比较
            exact_match = std_answer.lower() == reference["answer"].lower()
            return exact_match, std_answer
        
        return False, answer
    
    def evaluate_fill_in_blank(self) -> float:
        """评估填空题"""
        if not self.questions or "fill_in_the_blank" not in self.questions:
            return 0.0
        
        correct = 0
        total = 0
        
        for q_ref in self.questions["fill_in_the_blank"]:
            # 构建问题文本
            question = QUESTION_TEMPLATE.FILL_IN_BLANK.format(q_ref["question"])
            agent_answers = self._get_agent_answers(question)
            
            
            if not agent_answers:
                continue
                
            # 评估答案
            for answer in agent_answers:
                is_correct, std_answer = self._evaluate_single_answer(answer, "fill_in_blank", q_ref)
                self.evaluated_answers["fill_in_blank"].append({
                    "question": q_ref["question"],
                    "reference": q_ref["answer"],
                    "answer": answer,
                    "standardized": std_answer,
                    "correct": is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = EvaluationMetrics.calculate_accuracy(correct, total)
        self.scores["fill_in_blank"] = accuracy
        return accuracy
    
    def evaluate_yes_no(self) -> float:
        """评估是非题"""
        if not self.questions or "yes_no" not in self.questions:
            return 0.0
        
        correct = 0
        total = 0
        
        for q_ref in self.questions["yes_no"]:
            # 构建问题文本
            question = QUESTION_TEMPLATE.YES_NO.format(q_ref['question'])
            agent_answers = self._get_agent_answers(question)
            
            # 如果没有找到相应的问题和答案，尝试使用不同格式匹配
            if not agent_answers:
                for key in self.results.keys():
                    if q_ref["question"].split(".")[1].strip() in key:
                        agent_answers = self._get_agent_answers(key)
                        break
            
            if not agent_answers:
                continue
                
            # 评估答案
            for answer in agent_answers:
                is_correct, std_answer = self._evaluate_single_answer(answer, "yes_no", q_ref)
                self.evaluated_answers["yes_no"].append({
                    "question": q_ref["question"],
                    "reference": q_ref["answer"],
                    "answer": answer,
                    "standardized": std_answer,
                    "correct": is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = EvaluationMetrics.calculate_accuracy(correct, total)
        self.scores["yes_no"] = accuracy
        return accuracy
    
    def evaluate_multiple_choice(self) -> float:
        """评估多选题"""
        if not self.questions or "multiple_choice" not in self.questions:
            return 0.0
        
        correct = 0
        total = 0
        
        for q_ref in self.questions["multiple_choice"]:
            # 构建问题文本
            
            question = QUESTION_TEMPLATE.MULTIPLE_CHOICE.format(q_ref['question'], q_ref['options'])
            
            agent_answers = self._get_agent_answers(question)
            
            # 如果没有找到相应的问题和答案，尝试使用不同格式匹配
            
            if not agent_answers:
                print("answer not found for question:", question)
                continue
                
            # 评估答案
            for answer in agent_answers:
                is_correct, std_answer = self._evaluate_single_answer(answer, "multiple_choice", q_ref)
                self.evaluated_answers["multiple_choice"].append({
                    "question": q_ref["question"],
                    "reference": q_ref["answer"],
                    "options": q_ref["options"],
                    "answer": answer,
                    "standardized": std_answer,
                    "correct": is_correct
                })
                
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = EvaluationMetrics.calculate_accuracy(correct, total)
        self.scores["multiple_choice"] = accuracy
        return accuracy
    
    def evaluate_all(self) -> Dict[str, float]:
        """评估所有问题类型并计算总分"""
        self.evaluate_fill_in_blank()
        self.evaluate_yes_no()
        self.evaluate_multiple_choice()
        
        # 计算加权平均分数
        self.scores["aggregate"] = EvaluationMetrics.calculate_aggregated_score({
            "fill_in_blank": self.scores["fill_in_blank"],
            "yes_no": self.scores["yes_no"],
            "multiple_choice": self.scores["multiple_choice"]
        })
        
        return self.scores
    
    def generate_report(self) -> Dict:
        """生成详细评估报告"""
        return {
            "task_id": self.task_id,
            "scores": self.scores,
            "details": {
                "fill_in_blank": self.evaluated_answers["fill_in_blank"],
                "yes_no": self.evaluated_answers["yes_no"],
                "multiple_choice": self.evaluated_answers["multiple_choice"]
            }
        }


# 简单的LLM重新提问函数示例
def reprompt_llm(prompt: str) -> str:
    """
    简单的LLM重新提问函数(仅用于示例)
    在实际使用时，应该替换为实际的LLM调用
    """
    # 这里应该实现对LLM的调用
    print(f"Re-prompting LLM with: {prompt}")

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return completion.choices[0].message.content  # 默认返回


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM responses to video storytelling questions")
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True, help="Path to the results JSON file")
    parser.add_argument("--questions_dir", type=str, default="/data/wangshu/wangshu_code/ISG/ISV_eval/NovelConditionedVGen/instance_questions", 
                       help="Directory containing question JSON files")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.prompt_path):
        raise ValueError("Prompt path invalid, please check the file path.")
    with open(args.prompt_path, "r") as f:
        prompts = json.load(f)
    total = 0.0
    for task in prompts:
        task_id = task["id"]
        if os.path.exists(os.path.join(args.results_dir, f"Task_{str(task_id)}", "report.json")):
            print(f"Task {task_id} already evaluated, skipping...")
            with open(os.path.join(args.results_dir, f"Task_{str(task_id)}", "report.json"), 'r', encoding='utf-8') as f:
                report = json.load(f)
            print(report)
            total += report["scores"]["aggregate"]
            continue
        results_path = glob.glob(os.path.join(args.results_dir, f"Task_{task_id}",  "MovieReviewCommission_*.json"))[0]
        # 初始化评估器
        evaluator = ResponseEvaluator(results_path, args.questions_dir, reprompt_llm=reprompt_llm)
    
        # 运行评估
        scores = evaluator.evaluate_all()

        # 打印评估结果
        print("\n===== Evaluation Results =====")
        print(f"Task ID: {evaluator.task_id}")
        print(f"Fill-in-blank accuracy: {scores['fill_in_blank']:.2f}")
        print(f"Yes/No accuracy: {scores['yes_no']:.2f}")
        print(f"Multiple choice accuracy: {scores['multiple_choice']:.2f}")
        print(f"Aggregated score: {scores['aggregate']:.2f}")

        # 生成并保存报告(如果指定了输出路径)
        total += scores['aggregate']
        report = evaluator.generate_report()
        
        with open(os.path.join(args.results_dir, f"Task_{str(task_id)}", "report.json"), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation report saved to: {os.path.join(args.results_dir, f"Task_{str(task_id)}", "report.json")}")

    print(">>> Average score across all tasks: ", total / len(prompts))
if __name__ == "__main__":
    main()
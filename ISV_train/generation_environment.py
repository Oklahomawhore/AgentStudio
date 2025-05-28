import os
import json
import glob
import time
import asyncio
import logging
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import dotenv

import sys
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{file_path}/..")
sys.path.append(f"{file_path}/../ISG_agent")
sys.path.append(f"{file_path}/../ISV_eval")
from ISG_agent.PlanningAgentV2 import Execute_plan, double_check, extract_structure
from ISG_agent.util import GENERATION_MODE
from ISV_eval.agent_academy import get_score, process_question_v3, get_score_for_task
from ISV_eval.eval_instance import ResponseEvaluator
from ISV_eval.prompt_datasets import EnhancedVideoStorytellingDataset
from ISV_eval.question_define import generate_dataset_questions

from ISV_eval.eval_instance import reprompt_llm
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置文件日志记录
file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'generation_env.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

@dataclass
class EnvObservation:
    """环境观察对象，包含当前状态的所有相关信息"""
    text_history: str  # 文本历史记录
    plan_status: str   # 计划状态（创建中、已创建、执行中、已完成）
    video_status: str  # 视频状态（未创建、生成中、已生成）
    feedback: str      # 评估反馈
    task: Dict         # 任务信息

@dataclass
class EnvInfo:
    """环境信息，包含额外的调试和分析信息"""
    plan_file: str              # 计划文件路径
    video_files: List[str]      # 生成的视频文件路径
    execution_time: float       # 执行时间
    eval_report: Optional[Dict] # 评估报告
    false_answers: Optional[Dict] # 错误答案（如果有）

@dataclass
class HistoryItem:
    """历史记录项，用于追踪单个任务的执行历史"""
    task: Dict                  # 原始任务
    plan: List[Dict]            # 执行的计划
    video_files: List[str]      # 生成的视频文件
    response_text: str          # 模型生成的响应文本
    reward: float               # 获得的奖励
    
    def show_text(self):
        """显示可读的文本历史记录"""
        output = f"任务ID: {self.task.get('id', '未知')}\n"
        output += f"计划步骤数: {len(self.plan)}\n"
        output += f"生成视频: {', '.join(self.video_files) if self.video_files else '无'}\n"
        output += f"奖励值: {self.reward:.4f}\n"
        return output
    
    def show_tokens(self, tokenizer=None):
        """显示响应文本的标记化表示"""
        if tokenizer is None:
            return "需要提供tokenizer"
        tokens = tokenizer.encode(self.response_text)
        return tokens

class GenerationEnvironment:
    """
    视频生成环境，实现了基于脚本计划的视频生成过程管理。
    
    提供了类似强化学习环境的接口：reset()、step()，
    返回观察、奖励、是否终止以及信息。
    """
    
    def __init__(
        self, 
        base_dir: str = "../",
        output_dir: str = "results", 
        prompt_json: str = None,
        questions_dir: str = "ISV_eval/NovelConditionedVGen/instance_questions",
        generation_mode: str = GENERATION_MODE.T2V,
        model = None,
        processor = None,
        model_name: str = None,
        regenerate_question=False,
        args = None
    ):
        """
        初始化生成环境
        
        参数:
            base_dir: 基础目录路径
            output_dir: 输出目录
            questions_dir: 问题目录路径
            generation_mode: 生成模式 (t2v, i2v, r2v)
        """
        logger.info(f"初始化生成环境: mode={generation_mode}, output_dir={output_dir}")
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.questions_dir = questions_dir
        self.generation_mode = generation_mode
        self.prompt_json = prompt_json
        self.model = model
        self.processor = processor
        self.regenerate_question = regenerate_question
        self.model_name = model_name
        self.args = args
        
        # 环境状态
        self.current_task = None
        self.current_task_dir = None
        self.text_history = ""
        self.plan = None
        self.plan_file = None
        self.is_video_generated = False
        self.video_files = []
        self.eval_score = 0.0
        self.start_time = 0
        logger.debug("生成环境初始化完成")

        self.false_answers = {}
        self.correct_answers = {}
        
    def reset(self, task: Dict, task_dir: str) -> EnvObservation:
        """
        重置环境状态，准备新任务
        
        参数:
            task: 任务字典，包含标识符和内容
            
        返回:
            observation: 环境观察对象
        """
        task_id = task.get("id", "0000")
        logger.info(f"重置环境，开始处理任务 {task_id}")
        
        self.current_task = task
        self.current_task_dir = task_dir or os.path.join(self.output_dir, f"Task_{task_id}")
        os.makedirs(self.current_task_dir, exist_ok=True)
        
        self.text_history = f"开始处理任务 {task_id}\n"
        self.plan = None
        self.plan_file = os.path.join(self.current_task_dir, f"plan_{task_id}.json")
        self.is_video_generated = False
        self.video_files = []
        self.eval_score = 0.0
        self.start_time = time.time()
        
        # 检查已有的视频文件
        self._check_existing_videos()
        
        logger.debug(f"环境重置完成，任务目录: {self.current_task_dir}")
        return EnvObservation(
            text_history=self.text_history,
            plan_status="未创建" if not os.path.exists(self.plan_file) else "已创建",
            video_status="未创建" if not self.video_files else "已生成",
            feedback="",
            task=self.current_task
        )
    
    def step(self, plan: Optional[List[Dict]] = None, return_false=False) -> Tuple[EnvObservation, float, bool, EnvInfo]:
        """
        执行环境步进，处理计划或生成视频
        
        参数:
            plan: 可选的脚本计划。如果提供，将保存并执行；如果为None，尝试加载现有计划
            
        返回:
            observation: 环境观察
            reward: 奖励值（评估分数）
            done: 是否完成任务
            info: 额外信息
        """
        if self.current_task is None:
            logger.error("环境未初始化，请先调用reset()")
            raise ValueError("环境未初始化，请先调用reset()")
        
        task_id = self.current_task.get("id", "0000")
        self.plan = plan
        logger.info(f"执行环境步进，任务ID: {task_id}, 计划步骤数: {len(plan) if plan else '未提供'}")
        execution_time = time.time() - self.start_time
        
        # 检查是否已生成视频
        self._check_existing_videos()
        if self.is_video_generated:
            logger.info(f"视频已生成，环境终止。视频文件: {self.video_files}")
            self.text_history += "视频已生成，环境终止\n"
            
            # 运行评估
            reward = self._evaluate_videos(return_false=return_false)
            if return_false:
                reward, false_answers = reward
                logger.info(f"Return false answers in info. Reward: {reward:.2f}, False answers: {false_answers}")
            else:
                logger.info("No false answers in info, just return reward. {reward:.2f}")
                reward = reward
            
            return EnvObservation(
                text_history=self.text_history,
                plan_status="已完成",
                video_status="已生成",
                feedback=f"评估分数: {reward:.4f}",
                task=self.current_task
            ), reward, True, EnvInfo(
                plan_file=self.plan_file,
                video_files=self.video_files,
                execution_time=execution_time,
                eval_report=self._get_eval_report(),
                false_answers=false_answers if return_false else None
            )
        
        characters_file = os.path.join(self.current_task_dir, "characters.json")
        story_file = os.path.join(self.current_task_dir, "story.txt")
        is_characters_file = os.path.exists(characters_file)
        is_story_file = os.path.exists(story_file)
            
        # 如果没有有效计划，返回中间状态
        if self.plan is None or type(self.plan) is not list or len(self.plan) == 0 or not is_characters_file or not is_story_file:
            logger.warning("没有有效的执行计划，请提供计划")
            self.text_history += "没有有效的执行计划，请提供计划\n"
            return EnvObservation(
                text_history=self.text_history,
                plan_status="未创建",
                video_status="未创建",
                feedback="需要有效计划",
                task=self.current_task
            ), 0.0, False, EnvInfo(
                plan_file="",
                video_files=[],
                execution_time=execution_time,
                eval_report=None,
                false_answers=None
            )
        
        # 执行计划生成视频
        logger.info(f"开始执行计划生成视频，计划步骤数: {len(self.plan)}")
        self.text_history += "开始执行计划生成视频...\n"
        self._execute_plan()
        
        # 再次检查视频是否已生成
        self._check_existing_videos()
        if self.is_video_generated:
            logger.info(f"视频生成完成! 视频文件: {self.video_files}")
            self.text_history += "视频生成完成!\n"
            
            # 运行评估
            logger.info("开始评估视频质量...")
            if self.regenerate_question:
                # regenerate instance questions
                self.generate_instance_questions()
            result = self._evaluate_videos(return_false=return_false)
            if return_false:
                reward, false_answers = result
            else:
                reward = result
            logger.info(f"视频评估完成，评分: {reward:.4f}")
            
            return EnvObservation(
                text_history=self.text_history,
                plan_status="已完成",
                video_status="已生成",
                feedback=f"评估分数: {reward:.4f}",
                task=self.current_task
            ), reward, True, EnvInfo(
                plan_file=self.plan_file,
                video_files=self.video_files,
                execution_time=execution_time,
                eval_report=self._get_eval_report(),
                false_answers=false_answers if return_false else None
            )
        else:
            # 视频生成失败
            logger.error("视频生成失败，请检查错误日志")
            self.text_history += "视频生成失败，请检查错误日志\n"
            
            return EnvObservation(
                text_history=self.text_history,
                plan_status="已创建",
                video_status="生成失败",
                feedback="视频生成失败",
                task=self.current_task
            ), -0.1, False, EnvInfo(
                plan_file=self.plan_file,
                video_files=[],
                execution_time=execution_time,
                eval_report=None,
                false_answers=None
            )
    
    def _check_existing_videos(self):
        """检查是否存在已生成的视频文件"""
        video_files = glob.glob(os.path.join(self.current_task_dir, "final_video_*.mp4"))
        if video_files:
            logger.debug(f"找到已生成视频文件: {video_files}")
            self.is_video_generated = True
            self.video_files = video_files
            return True
        logger.debug("未找到已生成的视频文件")
        return False
    
    def _save_plan(self, plan):
        """保存脚本计划到文件"""
        try:
            with open(self.plan_file, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=4, ensure_ascii=False)
            self.plan = plan
            logger.debug(f"计划保存成功，共{len(plan)}个步骤")
            return True
        except Exception as e:
            logger.error(f"保存计划失败: {str(e)}")
            self.text_history += f"保存计划失败: {str(e)}\n"
            return False
    
    def _load_plan(self):
        """加载现有计划文件"""
        if os.path.exists(self.plan_file):
            try:
                with open(self.plan_file, 'r', encoding='utf-8') as f:
                    self.plan = json.load(f)
                logger.debug(f"成功加载计划，共{len(self.plan)}个步骤")
                self.text_history += f"加载现有计划: {self.plan_file}\n"
                return True
            except Exception as e:
                logger.error(f"加载计划失败: {str(e)}")
                self.text_history += f"加载计划失败: {str(e)}\n"
        else:
            logger.debug(f"计划文件不存在: {self.plan_file}")
        return False
    
    def _execute_plan(self):
        """执行计划生成视频"""
        if not self.plan:
            logger.warning("没有有效计划可执行")
            self.text_history += "没有有效计划可执行\n"
            return False
        
        try:
            # 加载角色信息
            characters = {}
            characters_file = os.path.join(self.current_task_dir, "characters.json")
            if os.path.exists(characters_file):
                with open(characters_file, 'r', encoding='utf-8') as f:
                    characters = json.load(f)
                logger.debug(f"已加载角色信息，共{len(characters)}个角色")
            else:
                logger.debug("未找到角色信息文件")
            
            # 加载故事信息
            story = ""
            story_file = os.path.join(self.current_task_dir, "story.txt")
            if os.path.exists(story_file):
                with open(story_file, 'r', encoding='utf-8') as f:
                    story = f.read()
                logger.debug(f"已加载故事信息，长度: {len(story)}字符")
            else:
                logger.debug("未找到故事文件")
            
            # 执行计划生成视频
            logger.info(f"开始执行计划生成视频，生成模式: {self.generation_mode}")

            # 检查self.args是否为数据类并转换为字典
            kwargs = {}
            if self.args is not None:
                if hasattr(self.args, '__dataclass_fields__'):  # 判断是否为数据类
                    kwargs = vars(self.args)  # 转换为字典
                else:
                    kwargs = self.args  # 已经是字典类型

            # 将kwargs传递给Execute_plan函数
            Execute_plan(
                self.plan, 
                self.current_task, 
                self.current_task_dir, 
                characters=characters, 
                story=story, 
                mode=self.generation_mode,
                **kwargs
            )
            logger.info("计划执行完成")
            
            return True
        except Exception as e:
            logger.exception(f"执行计划失败: {str(e)}")
            self.text_history += f"执行计划失败: {str(e)}\n"
            error_file = os.path.join(self.current_task_dir, "error.log")
            with open(error_file, 'a') as f:
                f.write(f"执行计划失败: {str(e)}\n")
            return False
    def generate_instance_questions(self):
        # regenerate instance questions
        generate_dataset_questions(self.prompt_json, prev_correct=self.correct_answers, output_dir=os.path.join(self.current_task_dir, "instance_questions"))
        self.questions_dir = os.path.join(self.current_task_dir, "instance_questions")

    def _evaluate_videos(self, return_false=False) -> Union[float, Tuple[float, Dict]]:
        """评估生成的视频质量并返回评分"""
        if not self.is_video_generated or not self.video_files:
            logger.warning("无视频可评估")
            return 0.0
        
        try:
            # 使用异步函数来评估视频
            video_path = self.video_files[0]
            task_id = self.current_task.get("id", "0000")
            logger.info(f"开始评估视频: {video_path}")
            
            # 从questions_dir加载问题
            question_path = os.path.join(self.questions_dir, f"{task_id}.json")
            
            if not os.path.exists(question_path):
                logger.error(f"找不到问题文件: {question_path}")
                self.text_history += f"找不到问题文件: {question_path}\n"
                return 0.0
            
            logger.debug(f"找到问题文件: {question_path}")
            
            # 加载问题文件
            with open(question_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            # 构建问题对象
            dataset = EnhancedVideoStorytellingDataset(self.prompt_json, self.questions_dir)
            questions = dataset.get_story_questions(int(task_id))
            
            # 处理问题
            logger.debug("处理问题中...")
            processed_questions = process_question_v3(questions, video_path)
            logger.debug(f"处理完成，共{len(processed_questions)}个问题")
            
            # 运行异步评估
            logger.info("启动异步评估...")
            kwargs = {}
            if self.model_name:
                kwargs['model_name'] = self.model_name
            result_file = asyncio.run(get_score(
                video_path, 
                processed_questions,
                output_dir=os.path.join(self.current_task_dir, 'review'),
                local_model=self.model,
                local_processor=self.processor,
                task_id=f"Task_{task_id}",
                **kwargs
            ))
            
            if result_file and os.path.exists(result_file):
                logger.debug(f"评估结果文件: {result_file}")
                # 评估回答
                evaluator = ResponseEvaluator(result_file, self.questions_dir, reprompt_llm=reprompt_llm)
                scores = evaluator.evaluate_all()
                false_answers = evaluator.get_false_answers()
                correct_answers = evaluator.get_correct_answers()
                self.false_answers[task_id] = false_answers
                self.correct_answers[task_id] = correct_answers
                self.eval_score = scores['aggregate']
                logger.info(f"评估完成: 填空题={scores['fill_in_blank']:.2f}, "
                          f"是非题={scores['yes_no']:.2f}, "
                          f"多选题={scores['multiple_choice']:.2f}, "
                          f"总分={self.eval_score:.4f}")
                
                # 保存报告
                report = evaluator.generate_report()
                report_path = os.path.join(self.current_task_dir, "report.json")
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.debug(f"评估报告已保存到: {report_path}")
                
                self.text_history += f"评估完成，总分: {self.eval_score:.4f}\n"
                return self.eval_score if not return_false else (self.eval_score, false_answers)
            else:
                logger.error("评估失败，无结果文件")
                self.text_history += "评估失败，无结果文件\n"
                return 0.0 if not return_false else (0.0, {})
                
        except Exception as e:
            logger.exception(f"评估视频时出错: {str(e)}")
            self.text_history += f"评估视频时出错: {str(e)}\n"
            return 0.0 if not return_false else (0.0, {})
    
    def _get_eval_report(self) -> Optional[Dict]:
        """获取评估报告（如果有）"""
        report_path = os.path.join(self.current_task_dir, "report.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                logger.debug(f"成功加载评估报告: {report_path}")
                return report
            except:
                logger.error(f"无法加载评估报告: {report_path}")
        else:
            logger.debug(f"评估报告不存在: {report_path}")
        return None
    
    def run(self, tasks: List[Dict]) -> Tuple[List[str], List[str], List[List[bool]], List[float], List[HistoryItem]]:
        """
        运行一组任务，返回TRL库PPO训练所需的数据格式
        
        参数:
            tasks: 任务列表
            
        返回:
            queries: 查询列表 (每个任务的输入提示)
            responses: 响应列表 (生成的计划文本)
            masks: 掩码列表 (标记哪些令牌是工具输出)
            rewards: 奖励列表 (每个任务的评分)
            histories: 历史记录列表 (详细的任务执行历史)
        """
        logger.info(f"开始运行任务批处理，共{len(tasks)}个任务")
        queries = []
        responses = []
        masks = []
        rewards = []
        histories = []
        
        for i, task in enumerate(tasks):
            task_id = task.get("id", "0000")
            logger.info(f"处理批处理任务 [{i+1}/{len(tasks)}], ID: {task_id}")
            
            # 重置环境，准备新任务
            self.reset(task)
            query = self._prepare_query_from_task(task)
            queries.append(query)
            logger.debug(f"已准备查询字符串，长度: {len(query)}字符")
            
            # 生成计划 (这里假设有一个计划生成方法)
            logger.info(f"为任务 {task_id} 生成计划")
            plan = self._generate_plan_for_task(task)
            plan_text = json.dumps(plan, ensure_ascii=False)
            responses.append(plan_text)
            logger.debug(f"已生成计划文本，长度: {len(plan_text)}字符")
            
            # 执行计划
            logger.info(f"执行任务 {task_id} 的计划")
            observation, reward, done, info = self.step(plan)
            rewards.append(reward)
            logger.info(f"任务 {task_id} 完成状态: {done}, 奖励: {reward:.4f}")
            
            # 生成掩码 (对于视频生成环境，我们标记工具输出部分)
            mask = self._generate_mask_for_response(plan_text)
            masks.append(mask)
            logger.debug(f"已生成掩码，长度: {len(mask)}")
            
            # 记录历史
            history_item = HistoryItem(
                task=task,
                plan=plan,
                video_files=info.video_files if done else [],
                response_text=plan_text,
                reward=reward
            )
            histories.append(history_item)
            logger.debug(f"已记录任务 {task_id} 的历史记录")
            
        logger.info(f"批处理完成，处理了 {len(tasks)} 个任务，成功率: {sum(1 for r in rewards if r > 0)/len(tasks):.2f}")
        return queries, responses, masks, rewards, histories
    
    def _prepare_query_from_task(self, task: Dict) -> str:
        """从任务创建查询字符串"""
        # 为任务创建一个标准格式的提示
        task_id = task.get("id", "0000")
        query_text = f"任务ID: {task_id}\n\n"
        
        # 添加查询内容
        if "Query" in task:
            for item in task["Query"]:
                if item["type"] == "text":
                    query_text += item["content"] + "\n"
                elif item["type"] == "image":
                    query_text += f"[图片: {item['content']}]\n"
        
        # 添加结构要求
        structure = extract_structure(task)
        if "Answer_str" in structure:
            query_text += f"\n输出结构要求: {structure['Answer_str']}\n"
            
        return query_text
    
    def _generate_plan_for_task(self, task: Dict) -> List[Dict]:
        """
        为任务生成计划，参照PlanningAgentV2.py中的main()函数逻辑
        
        使用LLM生成分步骤视频生成计划，包括角色提取、故事创作和详细分镜
        """
        task_id = task.get("id", "0000")
        task_dir = self.current_task_dir
        plan_file = os.path.join(task_dir, f"plan_{task_id}.json")
        characters_file = os.path.join(task_dir, "characters.json")
        story_file = os.path.join(task_dir, "story.txt")
        
        # 检查是否有已存在的计划
        if os.path.exists(plan_file):
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    logger.debug(f"加载现有计划文件: {plan_file}")
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载计划失败: {str(e)}，将生成新计划")
        
        # 记录开始生成计划
        logger.info(f"为任务 {task_id} 生成新的视频计划...")
        self.text_history += f"开始为任务 {task_id} 生成视频计划...\n"
        
        # 从PlanningAgentV2中导入预处理函数
        from ISG_agent.PlanningAgentV2 import (
            preprocess_task,
            PREPRODUCTION_PROMPTS,
            extract_json_from_response,
            transform_character_descriptions,
            extract_plan_from_response
        )
        from openai import OpenAI
        import dotenv
        
        # 加载环境变量并初始化客户端
        dotenv.load_dotenv()
        
        try:
            # 初始化OpenAI或其他LLM客户端
            client = OpenAI(
                api_key=os.getenv("KLING_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL")
            )
            
            # 预处理任务数据
            logger.info("预处理任务数据...")
            messages, Dict, Dict_for_plan = preprocess_task(task, task_dir, plan_model="openai")
            
            # 创建保存预处理数据的目录
            os.makedirs(task_dir, exist_ok=True)
            with open(os.path.join(task_dir, "Dict_for_plan.json"), 'w', encoding='utf-8') as f:
                json.dump(Dict_for_plan, f, indent=4, ensure_ascii=False)
            
            # 执行分步骤生成计划
            characters = {}
            assistant_response = None
            story = ""
            
            self.text_history += "开始执行多步骤计划生成流程...\n"
            
            for step, (step_name, step_prompt) in enumerate(PREPRODUCTION_PROMPTS.items()):
                logger.info(f"执行计划生成步骤 {step+1}: {step_name}")
                self.text_history += f"执行步骤 {step+1}: {step_name}\n"
                
                if assistant_response is not None:
                    messages.append(assistant_response)
                
                # 根据步骤调整提示
                if step == 0:
                    messages.append({"role": "user", "content": f"{step_name}: {step_prompt} \n\n {messages.pop(-1)['content'][-1]['text']}"})
                else:
                    messages.append({"role": "user", "content": f"{step_name}: {step_prompt} \n\n "})
                
                try:
                    # 尝试使用主要LLM模型
                    logger.debug(f"调用主模型生成响应: claude-3-7-sonnet...")
                    completion = client.chat.completions.create(
                        model='claude-3-7-sonnet-20250219',
                        response_format={'type': 'json'} if step_name == "Detailed Storyboarding" else None,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4096
                    )
                    assistant_response = completion.choices[0].message
                    response_text = completion.choices[0].message.content
                    
                    # 处理特殊情况
                    if "</think>" in response_text:
                        response_text = response_text.split("</think>")[-1]
                    if '请求错误' in response_text:
                        raise ValueError("请求错误")
                        
                except Exception as e:
                    # 如果主要模型失败，尝试备用模型
                    logger.warning(f"主模型调用失败，切换到备用模型: {str(e)}")
                    self.text_history += f"主模型调用失败，切换到备用模型: {str(e)}\n"
                    logger.debug("调用备用模型生成响应: gpt-4.5-preview...")
                    completion = client.chat.completions.create(
                        model='gpt-4.5-preview-2025-02-27',
                        response_format={'type': 'json'} if step_name == "Detailed Storyboarding" else None,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=4096
                    )
                    assistant_response = completion.choices[0].message
                    response_text = completion.choices[0].message.content
                
                # 处理角色提取步骤的输出
                if assistant_response and step_name == "Casting Extraction":
                    logger.info("提取角色信息...")
                    self.text_history += "提取角色信息...\n"
                    try:
                        char_json = extract_json_from_response(response_text)
                        characters = json.loads(char_json)
                        characters = transform_character_descriptions(characters)
                        with open(characters_file, 'w', encoding='utf-8') as f:
                            json.dump(characters, f, indent=4, ensure_ascii=False)
                        logger.debug(f"成功保存角色信息，共{len(characters)}个角色")
                        self.text_history += f"成功保存角色信息到 {characters_file}\n"
                    except Exception as e:
                        logger.error(f"角色提取失败: {str(e)}")
                        self.text_history += f"角色提取失败: {str(e)}\n"
                
                # 处理故事编写步骤的输出
                if assistant_response and step_name == "Script Writing":
                    logger.info("保存故事脚本...")
                    self.text_history += "保存故事脚本...\n"
                    story = response_text
                    with open(story_file, 'w', encoding='utf-8') as f:
                        f.write(story)
                    logger.debug(f"成功保存故事脚本，长度: {len(story)}字符")
                    self.text_history += f"成功保存故事脚本到 {story_file}\n"
            
            # 保存所有消息历史记录
            logger.debug("保存消息历史记录...")
            message_file = os.path.join(task_dir, "messages.json")
            with open(message_file, 'w', encoding='utf-8') as f:
                json.dump([self._message_to_json(m) for m in messages], f, indent=4, ensure_ascii=False)
            
            # 提取最终计划
            logger.info("提取视频生成计划...")
            self.text_history += "提取视频生成计划...\n"
            try:
                # 使用临时文件存储计划，使用extract_plan_from_response函数处理
                temp_plan_file = os.path.join(task_dir, "temp_plan.json")
                extract_plan_from_response(response_text, temp_plan_file, characters=characters)
                
                # 加载生成的计划
                with open(temp_plan_file, 'r', encoding='utf-8') as f:
                    plan = json.load(f)
                
                # 保存最终计划
                with open(plan_file, 'w', encoding='utf-8') as f:
                    json.dump(plan, f, indent=4, ensure_ascii=False)
                
                logger.info(f"成功生成计划，共{len(plan)}个步骤")
                self.text_history += f"成功生成计划，共 {len(plan)} 个步骤\n"
                return plan
            except Exception as e:
                logger.error(f"计划提取失败: {str(e)}")
                self.text_history += f"计划提取失败: {str(e)}\n"
                
                # 生成一个简单的默认计划作为后备
                logger.warning("生成默认备用计划...")
                default_plan = [
                    {
                        "Step": 1,
                        "Task": self.generation_mode,
                        "Input_text": f"场景1: {task_id} - 基于任务描述的通用视频场景",
                        "Music_prompt": "",
                        "TTS_prompt": "",
                        "Input_images": [],
                        "Output": "<WAIT>"
                    }
                ]
                
                # 保存默认计划
                with open(plan_file, 'w', encoding='utf-8') as f:
                    json.dump(default_plan, f, indent=4, ensure_ascii=False)
                
                self.text_history += "生成了默认备用计划\n"
                return default_plan
                
        except Exception as e:
            logger.exception(f"计划生成过程中出现错误: {str(e)}")
            self.text_history += f"计划生成过程中出现错误: {str(e)}\n"
            
            # 返回最基本的默认计划
            default_plan = [
                {
                    "Step": 1,
                    "Task": self.generation_mode,
                    "Input_text": "场景1: 生成视频的基本描述",
                    "Music_prompt": "",
                    "TTS_prompt": "",
                    "Input_images": [],
                    "Output": "<WAIT>"
                }
            ]
            
            # 保存默认计划
            with open(plan_file, 'w', encoding='utf-8') as f:
                json.dump(default_plan, f, indent=4, ensure_ascii=False)
            
            return default_plan
    
    def _message_to_json(self, message):
        """将消息对象转换为JSON可序列化的字典"""
        if isinstance(message, dict):
            return message
        return {
            "role": message.role,
            "content": message.content
        }
    
    def _generate_mask_for_response(self, response_text: str) -> List[bool]:
        """
        为响应生成掩码，标识哪些部分是工具输出
        
        参数:
            response_text: 响应文本(计划JSON字符串)
            
        返回:
            mask: 布尔值列表，标识哪些标记是工具输出
        """
        # 在我们的实现中，整个响应都是模型生成的计划，没有工具输出部分
        # 因此我们返回一个全False的掩码列表
        import re
        
        # 使用正则表达式匹配JSON中的特定标记
        # 我们认为JSON中的"Output":"<WAIT>"部分表示工具输出
        tokens = re.findall(r'[^"\s]+|"[^"]*"', response_text)
        mask = [False] * len(tokens)
        
        # 寻找"Output"和"<WAIT>"相近的标记作为工具输出相关部分
        output_index = -1
        for i, token in enumerate(tokens):
            if '"Output"' in token:
                output_index = i
            elif output_index >= 0 and i - output_index < 4 and '"<WAIT>"' in token:
                # 标记"Output":"<WAIT>"这部分为工具输出相关
                mask[output_index:i+1] = [True] * (i - output_index + 1)
                output_index = -1
                
        logger.debug(f"生成掩码，共{len(mask)}个标记，工具输出相关标记数: {sum(mask)}")
        return mask

if __name__ == "__main__":
    # 简单使用示例
    import argparse
    
    parser = argparse.ArgumentParser(description='视频生成环境')
    parser.add_argument('--task', type=str, required=True, help='任务JSON文件')
    parser.add_argument('--output', type=str, default='env_results', help='输出目录')
    parser.add_argument('--mode', choices=['t2v', 'i2v', 'r2v'], default='t2v', help='生成模式')
    parser.add_argument('--run-batch', action='store_true', help='运行批量任务')
    parser.add_argument('--eval-only', action='store_true', help='仅评估现有视频而不生成')
    args = parser.parse_args()
    
    # 加载任务
    with open(args.task, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    if not tasks:
        print("任务文件为空")
        sys.exit(1)
    
    # 创建环境
    env = GenerationEnvironment(output_dir=args.output, generation_mode=args.mode)
    
    if args.run_batch:
        # 批量运行所有任务并评估
        print(f"批量运行 {len(tasks)} 个任务...")
        queries, responses, masks, rewards, histories = env.run(tasks)
        
        # 显示批处理结果
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        print(f"批处理完成，平均奖励: {avg_reward:.4f}")
        
        # 保存批处理结果
        result_summary = {
            "avg_reward": avg_reward,
            "task_rewards": {task.get("id", f"task_{i}"): reward for i, (task, reward) in enumerate(zip(tasks, rewards))},
            "completion_rate": sum(1 for h in histories if h.video_files) / len(histories)
        }
        
        with open(os.path.join(args.output, "batch_results.json"), "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2, ensure_ascii=False)
        
        print(f"批处理结果已保存到 {os.path.join(args.output, 'batch_results.json')}")
    else:
        # 只处理第一个任务
        task = tasks[0]
        print(f"处理单个任务: {task.get('id', '0000')}")
        
        # 重置环境
        obs = env.reset(task)
        print(f"初始观察: {obs}")
        
        if args.eval_only:
            # 只评估现有视频，不生成新视频
            if env._check_existing_videos():
                print(f"找到现有视频: {env.video_files}")
                reward = env._evaluate_videos()
                print(f"评估结果: {reward:.4f}")
            else:
                print("未找到现有视频，无法评估")
        else:
            # 正常执行步骤
            if os.path.exists(env.plan_file):
                # 使用现有计划
                print(f"使用现有计划: {env.plan_file}")
                obs, reward, done, info = env.step()
            else:
                # 生成新计划
                print("生成新计划...")
                plan = env._generate_plan_for_task(task)
                obs, reward, done, info = env.step(plan)
            
            print(f"步骤结果: 完成={done}, 奖励={reward:.4f}")
            if done and info.video_files:
                print(f"视频文件: {info.video_files}")
                
                # 显示评估报告摘要
                if info.eval_report:
                    print("\n评估报告摘要:")
                    for q_type in ["fill_in_blank", "yes_no", "multiple_choice"]:
                        print(f"  {q_type}: {info.eval_report['scores'].get(q_type, 0.0):.2f}")
                    print(f"  总分: {info.eval_report['scores'].get('aggregate', 0.0):.4f}")
            else:
                print("视频生成未完成或失败")

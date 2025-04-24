import os
import json
import torch
import argparse
import wandb
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from generation_environment import GenerationEnvironment
from accelerate import Accelerator
from accelerate.utils import DistributedType

# 配置日志
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加文件日志记录器
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

@dataclass
class ScriptArguments:
    """
    PPO训练的参数
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-32B-AWQ", 
        metadata={"help": "要训练的模型名称或路径"}
    )
    dataset_path: str = field(
        default="/data/wangshu/wangshu_code/ISG/ISV_eval/NovelConditionedVGen/instance_tasks.json",
        metadata={"help": "任务数据集路径"}
    )
    output_dir: str = field(
        default="output",
        metadata={"help": "模型和检查点的输出目录"}
    )
    env_output_dir: str = field(
        default="env_results",
        metadata={"help": "环境结果输出目录"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA的r维度"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout概率"}
    )
    learning_rate: float = field(
        default=1.41e-5,
        metadata={"help": "学习率"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "训练批次大小"}
    )
    mini_batch_size: int = field(
        default=1,
        metadata={"help": "PPO小批次大小"}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "梯度累积步数"}
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "每个PPO更新的epoch数"}
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "最大训练步数"}
    )
    save_steps: int = field(
        default=10,
        metadata={"help": "保存检查点的步数间隔"}
    )
    eval_steps: int = field(
        default=10,
        metadata={"help": "评估的步数间隔"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "随机种子"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "要应用LoRA的目标模块，逗号分隔"}
    )
    use_wandb: bool = field(
        default=True,
        metadata={"help": "是否使用wandb进行日志记录"}
    )
    wandb_project: str = field(
        default="video_agent_ppo",
        metadata={"help": "wandb项目名称"}
    )
    generation_mode: str = field(
        default="t2v",
        metadata={"help": "生成模式: t2v, i2v, r2v"}
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "生成的最大长度"}
    )
    min_length: int = field(
        default=512,
        metadata={"help": "生成的最小长度"}
    )

def init_wandb(args):
    """初始化wandb日志记录"""
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"ppo_{args.model_name_or_path.split('/')[-1]}_{args.generation_mode}"
        )
        logger.info("初始化wandb完成")
    else:
        logger.info("未启用wandb日志记录")

def load_dataset(dataset_path: str):
    """加载任务数据集"""
    logger.info(f"从{dataset_path}加载数据集")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载数据集，共{len(data)}条任务")
        return data
    except Exception as e:
        logger.error(f"加载数据集失败: {str(e)}")
        raise

def prepare_model_and_tokenizer(args):
    """准备模型和分词器，应用量化和LoRA配置"""
    logger.info(f"加载模型和分词器: {args.model_name_or_path}")
    
    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 准备模型进行8位训练
    model = prepare_model_for_kbit_training(model)
    
    # 添加LoRA配置
    target_modules = args.lora_target_modules.split(",")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    
    # 将模型包装为ValueHead模型用于PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # 输出模型参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} ({trainable_params / total_params:.2%})")
    logger.info(f"总参数: {total_params:,}")
    
    return model, tokenizer

def train(args):
    """主训练函数"""
    logger.info("开始训练准备...")
    
    # 初始化wandb
    init_wandb(args)
    
    # 设置随机种子
    transformers.set_seed(args.seed)
    
    # 加载任务数据
    dataset = load_dataset(args.dataset_path)
    
    # 准备模型和分词器
    model, tokenizer = prepare_model_and_tokenizer(args)
    
    # 初始化分布式训练
    accelerator = Accelerator()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置PPO配置
    ppo_config = PPOConfig(
        model_name=args.model_name_or_path,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        max_grad_norm=0.5,
        optimize_device_cache=True,
        seed=args.seed,
        optimize_cuda_cache=True
    )
    
    # 初始化PPO训练器
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator
    )
    
    # 创建环境
    env = GenerationEnvironment(
        base_dir="/data/wangshu/wangshu_code/ISG",
        output_dir=args.env_output_dir,
        generation_mode=args.generation_mode
    )
    
    # 创建长度采样器
    length_sampler = LengthSampler(args.min_length, args.max_length)
    
    # 主训练循环
    global_step = 0
    best_reward = -float('inf')

    logger.info("开始训练循环...")
    
    while global_step < args.max_steps:
        # 采样任务批次
        task_indices = np.random.choice(len(dataset), args.batch_size, replace=False)
        tasks_batch = [dataset[i] for i in task_indices]
        
        # 准备查询
        queries = []
        for task in tasks_batch:
            task_text = f"任务ID: {task.get('id', '0000')}\n"
            if "Query" in task:
                for item in task["Query"]:
                    if item["type"] == "text":
                        task_text += item["content"] + "\n"
                    elif item["type"] == "image":
                        task_text += f"[图片: {item['content']}]\n"
            task_text += (
                "\n请为这个视频生成任务创建详细的分镜脚本计划。"
                "计划应该包括每一个步骤的详细描述，以便于生成高质量的视频。"
                "输出格式应为JSON数组，其中每个对象代表一个步骤，包含Step、Task、Input_text等关键字段。"
            )
            queries.append(task_text)
        
        # 编码查询
        query_tensors = [tokenizer(query, return_tensors="pt").input_ids.to(ppo_trainer.accelerator.device) for query in queries]
        
        # 生成响应
        response_tensors = []
        for query in query_tensors:
            gen_len = length_sampler()
            with torch.no_grad():
                response = ppo_trainer.generate(query, max_new_tokens=gen_len)
                response_tensors.append(response.squeeze())
        
        # 解码响应
        batch_responses = []
        for i, (query, response) in enumerate(zip(query_tensors, response_tensors)):
            decoded_response = tokenizer.decode(response[query.shape[1]:], skip_special_tokens=True)
            batch_responses.append(decoded_response)
            
            logger.debug(f"任务 {i} 生成的响应长度: {len(decoded_response)} 字符")
        
        # 准备计划并运行环境
        batch_rewards = []
        for task, response_text in zip(tasks_batch, batch_responses):
            # 尝试解析生成的计划JSON
            try:
                plan = json.loads(response_text)
                if not isinstance(plan, list):
                    plan = [plan]
            except:
                # 如果解析失败，尝试提取JSON部分
                import re
                json_match = re.search(r'\[[\s\S]*\]', response_text)
                if json_match:
                    try:
                        plan = json.loads(json_match.group(0))
                    except:
                        # 失败时使用简单默认计划
                        plan = [{
                            "Step": 1,
                            "Task": args.generation_mode,
                            "Input_text": f"基于任务 {task.get('id')} 的视频场景",
                            "Music_prompt": "",
                            "TTS_prompt": "",
                            "Input_images": [],
                            "Output": "<WAIT>"
                        }]
                else:
                    # 失败时使用简单默认计划
                    plan = [{
                        "Step": 1,
                        "Task": args.generation_mode,
                        "Input_text": f"基于任务 {task.get('id')} 的视频场景",
                        "Music_prompt": "",
                        "TTS_prompt": "",
                        "Input_images": [],
                        "Output": "<WAIT>"
                    }]
            
            # 重置环境并执行计划
            env.reset(task)
            observation, reward, done, info = env.step(plan)
            
            batch_rewards.append(float(reward))
            
            logger.debug(f"任务 {task.get('id')}: 完成状态={done}, 奖励={reward:.4f}")
            
            if done and reward > 0:
                # 保存有效计划示例
                with open(os.path.join(args.output_dir, f"good_plan_{task.get('id')}.json"), 'w', encoding='utf-8') as f:
                    json.dump(plan, f, ensure_ascii=False, indent=2)
        
        # 计算平均奖励
        mean_reward = np.mean(batch_rewards) if batch_rewards else 0
        logger.info(f"步骤 {global_step}: 平均奖励 = {mean_reward:.4f}")
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                "global_step": global_step,
                "mean_reward": mean_reward,
                "max_reward": max(batch_rewards) if batch_rewards else 0,
                "min_reward": min(batch_rewards) if batch_rewards else 0
            })
        
        # PPO更新
        stats = ppo_trainer.step(query_tensors, response_tensors, batch_rewards)
        ppo_trainer.log_stats(stats, batch_rewards, query_tensors, response_tensors)
        
        # 更新全局步数
        global_step += 1
        
        # 保存模型检查点
        if global_step % args.save_steps == 0 or mean_reward > best_reward:
            if mean_reward > best_reward:
                best_reward = mean_reward
                logger.info(f"新的最佳奖励: {best_reward:.4f}")
                ppo_trainer.save_pretrained(os.path.join(args.output_dir, "best_reward_model"))
            
            save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            ppo_trainer.save_pretrained(save_dir)
            logger.info(f"保存检查点到: {save_dir}")
        
        # 评估
        if global_step % args.eval_steps == 0:
            logger.info(f"步骤 {global_step} - 运行评估...")
            # 这里可以添加额外的评估逻辑
    
    # 保存最终模型
    ppo_trainer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    logger.info("训练完成，保存最终模型")
    
    # 清理wandb
    if args.use_wandb:
        wandb.finish()

def main():
    """主函数：解析参数并开始训练"""
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    logger.info(f"启动PPO训练，参数:\n{json.dumps(vars(args), indent=2)}")
    
    train(args)

if __name__ == "__main__":
    main()

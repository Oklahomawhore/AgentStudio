import os
import json
import torch
import argparse
import wandb
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random
import math, time
import functools

import transformers
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoProcessor,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import GRPOTrainer, GRPOConfig
from custom_trainer import VLMPPO
from trl.core import LengthSampler
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

from generation_environment import GenerationEnvironment
from accelerate import Accelerator
from accelerate.utils import DistributedType

from ISG_agent.PlanningAgentV2 import generate_single_task, Execute_plan, extract_character_from_content, transform_character_descriptions, extract_plan_from_response, extract_json_from_response
from ISG_agent.util import save_plan_json, load_input_json
from ISV_eval.util import resolve_api_from_model_name
from ICL_META import ICL_META
from ISV_train.vgen_dataset import vgen_dataset

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

proprietary_models = [
    "doubao",
    "gpt",
    "gemini",
    "claude",
    "deepseek"
]

@dataclass
class ScriptArguments:
    """
    PPO训练的参数
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-32B-Instruct-AWQ", 
        metadata={"help": "要训练的模型名称或路径"}
    )
    train_method: str = field(
        default='ppo',
        metadata={"help": "训练方法，支持ppo, grpo, icl"}
    )
    dataset_path: str = field(
        default="/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json",
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
    regenerate_question: bool = field(
        default=False,
        metadata={"help": "是否重新生成问题"}
    )
    video_gen_api_base: str = field(
        default=None,
        metadata={"help": "视频生成统一接口API"}
    )
    use_icl: bool = field(
        default=False,
        metadata={"help": "是否使用示例学习进行训练"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "是否使用LoRA进行训练"}
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
        default=1,
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
        default=4096,
        metadata={"help": "生成的最大长度"}
    )
    min_length: int = field(
        default=2048,
        metadata={"help": "生成的最小长度"}
    )
    plan_template: str = field(
        default="/data/wangshu/wangshu_code/ISG/ISG_agent/Prompt/plan_template.json",
        metadata={"help": "生成计划的模板"}
    )
    train_val_ratio: float = field(
        default=0.8,
        metadata={"help": "训练集占总数据的比例"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "是否使用QLoRA进行训练"}
    )
    quantization_bits: int = field(
        default=4,
        metadata={"help": "量化位数，可选4或8"}
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "干运行模式，跳过PPO更新步骤，仅保存查询、响应和奖励"}
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练adapter权重的路径，不提供则不加载adapter"}
    )
    merge_adapter: bool = field(
        default=False,
        metadata={"help": "是否将adapter权重合并到基础模型中"}
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
    
    # 加载分词器
    processor = AutoProcessor.from_pretrained(args.model_name_or_path,use_fast=True)
    
    if args.use_qlora:
        logger.info(f"使用QLoRA配置，量化位数: {args.quantization_bits}位")
        # QLoRA量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.quantization_bits == 4, 
            load_in_8bit=args.quantization_bits == 8,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # 加载基础模型
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        
        # 准备模型进行kbit训练
        model = prepare_model_for_kbit_training(model)
    else:
        # 不使用量化，直接加载模型
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    # 添加LoRA配置
    target_modules = args.lora_target_modules.split(",")
    
    # 加载预训练的adapter（如果指定）
    if args.adapter_name_or_path:
        logger.info(f"从 {args.adapter_name_or_path} 加载预训练adapter")
        try:
            lora_config = LoraConfig.from_pretrained(args.adapter_name_or_path)
            model = get_peft_model(model, lora_config)
            
            # 如果需要，合并adapter权重到基础模型
            if args.merge_adapter:
                logger.info("将adapter权重合并到基础模型")
                model = model.merge_and_unload()
                logger.info("合并完成，将继续应用新的LoRA配置")
        except Exception as e:
            logger.error(f"加载adapter失败: {str(e)}")
            logger.warning("将继续使用未加载adapter的模型")
        finally:
            logger.info("加载adapter完成")
    
    # 如果没有合并adapter或未加载adapter，应用新的LoRA配置
    if args.adapter_name_or_path is None and args.use_lora:
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
    if args.train_method == 'ppo':
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    
    # 输出模型参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable_params:,} ({trainable_params / total_params:.2%})")
    logger.info(f"总参数: {total_params:,}")
    
    # 打印内存使用情况
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
        gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
        logger.info(f"GPU内存使用: 分配 = {gpu_mem_alloc:.2f} GB, 保留 = {gpu_mem_reserved:.2f} GB")
    
    return model, processor

def same_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    transformers.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"设置随机种子: {seed}")

def reward_func(completions=None, task=None, env=None,args=None, **kwargs) -> List[float]:
    if env is None or task is None:
        return [None] * len(completions)
    rewards = []
    for completion, cur_task in zip(completions, task):
        task_dir = os.path.join(args.env_output_dir, f"global_step_{kwargs.get('global_step', 0)}", f"Task_{cur_task['id']}")
        plan_file = os.path.join(task_dir, f"plan_{cur_task.get('id', '0000')}.json")
        try:
            extract_plan_from_response(completion, plan_file)
        except Exception as e:
            logger.error(f"Error extracting plan: {str(e)}")
            logger.info(completion)
            rewards.append(None)
            continue
            # raise ValueError("Error extracting plan")
        if os.path.exists(plan_file):
            plan = load_input_json(plan_file)
        else:
            plan = None
        
        env.reset(cur_task, task_dir=os.path.join(args.env_output_dir, f"global_step_{kwargs.get('global_step', 0)}", f"Task_{task['id']}"), reset_questinos=args.use_icl, task_ids=[task["id"] for task in tasks])
        observation, reward_dict, done, info = env.step(plan, return_false=args.use_icl, return_ref_reward=args.use_icl)
        reward = reward_dict.get("reward", 0.0)
        ref_reward = reward_dict.get("ref_reward", 0.0)
        rewards.append(reward)
    return rewards

def train(args):
    """主训练函数"""
    logger.info("开始训练准备...")
    
    # 初始化wandb
    init_wandb(args)
    
    # 设置随机种子
    same_seed(args.seed)
    
    # 加载任务数据
    dataset = load_dataset(args.dataset_path)
    
    # 训练/验证集划分
    num_tasks = len(dataset)
    indices = list(range(num_tasks))
    np.random.shuffle(indices)
    split_idx = int(num_tasks * args.train_val_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    logger.info(f"总任务数: {num_tasks}, 训练集大小: {len(train_indices)}, 验证集大小: {len(val_indices)}")
    
    # 准备模型和分词器
    use_proprietary_model = False
    for model_name in proprietary_models:
        if model_name in args.model_name_or_path:
            use_proprietary_model = True
    if use_proprietary_model:
        # 使用专有模型时，使用LoRA配置
        model, processor = None, None
        model_key, base_url = resolve_api_from_model_name(args.model_name_or_path, config_file=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ISV_eval/model_mapping.yaml")))
        client = OpenAI(base_url=base_url, api_key=model_key)
    else:
        model, processor = prepare_model_and_tokenizer(args)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # 初始化分布式训练
    # accelerator = Accelerator()
    # accelerator.init_trackers("video_agent_ppo", config=vars(args))
    # model, optimizer, lr_schedular = accelerator.prepare(model, optimizer, lr_schedular)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # 创建环境
    isv_train_dir = os.path.dirname(os.path.abspath(__file__))
    env = GenerationEnvironment(
        base_dir=os.path.join(isv_train_dir, ".."),
        output_dir=args.env_output_dir,
        generation_mode=args.generation_mode,
        prompt_json=os.path.join(isv_train_dir, "..", "ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json"),
        questions_dir=os.path.join(isv_train_dir, "..", "ISV_eval/datasets/NovelConditionedVGen/instance_questions_deepseek"),
        model=model,
        processor=processor,
        regenerate_question=args.regenerate_question,
        args=args,
        model_name=args.model_name_or_path,
    )
    
        
    if args.train_method != 'icl':
        args.use_icl = False
    else:
        args.use_icl = True
    
    # if not use_proprietary_model:
    #     # 设置PPO配置
    if args.train_method == 'grpo':
        # 初始化VLM GRPO训练器
        training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
        ppo_trainer = GRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            reward_funcs=functools.partial(reward_func, env=env, tasks=[dataset[i] for i in train_indices], args=args),
            train_dataset=vgen_dataset(args.dataset_path, args),
            args=training_args,
        )
    elif args.train_method == 'ppo':
        ppo_config = PPOConfig(
            model_name=args.model_name_or_path,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size * 5,
            mini_batch_size=args.mini_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps * 5,
            backward_batch_size=1,
            ppo_epochs=args.ppo_epochs,
            max_grad_norm=0.5,
            optimize_device_cache=True,
            seed=args.seed,
            optimize_cuda_cache=True,
            kl_penalty="kl"
        )
        # 初始化VLM PPO训练器
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=processor.tokenizer,
        )
    
    
    
    # 创建长度采样器
    length_sampler = LengthSampler(args.min_length, args.max_length)
    
    # 主训练循环
    global_step = 0
    best_reward = -float('inf')

    plan_template = json.load(open(args.plan_template, 'r', encoding='utf-8'))
    system_prompt = "You are a creative and storytelling-driven video generation agent. " \
    "Your responsibility is to design the complete workflow for producing short-form video content — " \
    "from concept planning, script writing, casting design, to detailed storyboarding."

    logger.info("开始训练循环...")

    generation_kwargs = {
        "top_k": 20,
        "top_p": 0.95,
        "do_sample": True,
        "temperature": 0.6,
        "max_new_tokens": 2048,
    }
    batch_indices = np.random.choice(train_indices, args.batch_size, replace=False)

    if args.use_icl:
        # proprietary model or ICL mode, can use icl training only
        task_infos = {}
        while global_step < args.max_steps:
            # 从训练集中采样任务批次

            tasks_batch = [dataset[i] for i in batch_indices]
            logger.info(f"global step {global_step} 采样训练任务批次: {batch_indices}")
            # 准备查询
            query_tensors = [] # bs * step_len, seq_len
            response_tensors = [] # bs * step_len, seq_len
            batch_responses = [] # bs * step_len, seq_len
            if os.path.exists(os.path.join(args.env_output_dir, f"intermediate_{global_step}.pt")):
                logger.info(f"加载中间结果: {os.path.join(args.env_output_dir, f'intermediate_{global_step}.pt')}")
                intermediate = torch.load(os.path.join(args.env_output_dir, f"intermediate_{global_step}.pt"), weights_only=False)
                query_tensors = intermediate["query_tensors"]
                response_tensors = intermediate["response_tensors"]
                batch_responses = intermediate["batch_responses"]
            else:
                for i, task in enumerate(tasks_batch):
                    task_text = task["Query"][0]["content"]
                    logger.info(f"开始任务 {task['id']} 的计划生成")
                    assistant_response = None
                    task_dir = os.path.join(args.env_output_dir, f"global_step_{global_step}", f"Task_{task['id']}")
                    os.makedirs(task_dir, exist_ok=True)
                    plan_file = os.path.join(task_dir, f"plan_{task.get('id', '0000')}.json")
                    messages = []
                    for step, (step_name, step_prompt) in enumerate(plan_template.items()):
                        logger.info(f"正在处理步骤 {step_name}")
                        if assistant_response is not None:
                            # assistance is content dict if not string
                            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_response if isinstance(assistant_response, str) else assistant_response.content}]})
                        if step == 0:
                            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
                            if args.use_icl and step_name == "Detailed Storyboarding":
                                messages.append({"role": "user", "content": [{"type":"text", "text" : f"{step_name}: {step_prompt} \n\n #Story:\n {task_text} \n\n {ICL_META.format(task_infos[task['id']]) if task['id'] in task_infos else ''}"}]})
                                logger.info(f"Using task infos for detailed storyboarding: {task_infos.get(task['id'], '')}")
                            else:
                                messages.append({"role": "user", "content": [{"type": "text", "text" : f"{step_name}: {step_prompt} \n\n {task_text}"}]})
                        else:
                            if task['id'] in task_infos and args.use_icl and step_name == "Detailed Storyboarding":
                                messages.append({"role": "user", "content": [{"type":"text", "text" : f"{step_name}: {step_prompt} \n\n #Story:\n {task_text} \n\n {ICL_META.format(task_infos[task['id']])}"}]})
                                logger.info(f"Using task infos for detailed storyboarding: {task_infos[task['id']]}")
                            else:
                                messages.append({"role": "user", "content": [{"type":"text", "text" : f"{step_name}: {step_prompt} \n\n "}]})
                        if not use_proprietary_model:
                            # local model we use processor to encode
                            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                            inputs = processor(text=[text], padding=True, return_tensors="pt")
                            inputs = inputs.to(ppo_trainer.accelerator.device)
                            query_tensors.append(inputs.input_ids.squeeze()) 

                        # 生成响应
                        with torch.no_grad():
                            # LM_head forward
                            if use_proprietary_model:
                                # use openai compatible api
                                completion = client.chat.completions.create(
                                    model=args.model_name_or_path,
                                    messages=messages,
                                    temperature=0.6,
                                    max_completion_tokens=2048,
                                    seed=args.seed
                                )
                                response = completion.choices[0].message.content
                            else:
                                model_inputs = {"input_ids": inputs.input_ids, "attention_mask": torch.ones_like(inputs.input_ids).to(inputs.input_ids.device)}
                                response = model.generate(
                                    **model_inputs,
                                    **generation_kwargs
                                )

                                response_tensors.append(response.squeeze()[inputs.input_ids.shape[1]:])

                        # truncate prompt+respones to max_length
                        if not args.use_icl and len(query_tensors[-1]) + len(response_tensors[-1]) > args.max_length:
                            # assert len(response_tensors[-1]) < args.max_length, "response length is larger than max_length"
                            if len(query_tensors[-1]) > args.max_length:
                                logger.warning(f"query length is larger than max_length, query length: {len(query_tensors[-1])}, max_length: {args.max_length}")
                                query_tensors[-1] = torch.tensor([], dtype=torch.long)
                            else:
                                truncate_length = len(query_tensors[-1]) + len(response_tensors[-1]) - args.max_length
                                query_tensors[-1] = query_tensors[-1][-(len(query_tensors[-1]) - truncate_length):]
                        if not use_proprietary_model:
                            decoded_response = processor.decode(response.squeeze()[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                            response_text = decoded_response
                            assistant_response = decoded_response
                            logger.debug(f"任务 {i} 生成的响应长度: {len(decoded_response)} 字符")
                        else:
                            assistant_response = completion.choices[0].message
                            response_text = completion.choices[0].message.content
                        if assistant_response and step_name == "Casting Extraction":
                            logger.info("正在提取角色中...")
                            try:
                                characters = extract_json_from_response(response_text)
                                characters = json.loads(characters)
                            except Exception as e:
                                logger.error(f"Error extracting characters from completion {response_text}")
                                characters = None
                                # fix possible json format error
                            if characters is not None:
                                characters = transform_character_descriptions(characters)
                                save_plan_json(characters, f"{task_dir}/characters.json")
                            else:
                                continue
                        if assistant_response and step_name == "Script Writing":
                            logger.info("正在提取分镜脚本中...")
                            story = response_text
                            with open(f"{task_dir}/story.txt", "w") as f:
                                f.write(story)

                        # batch_responses.append(decoded_response)


                    if assistant_response is not None:
                        messages.append({"role": "assistant", "content": [{"type": "text", "text":assistant_response if isinstance(assistant_response, str)  else assistant_response.content}]}) 
                    try:
                        extract_plan_from_response(response_text, plan_file)
                    except Exception as e:
                        logger.error(f"Error extracting plan: {str(e)}")
                        logger.info(response_text)
                        # raise ValueError("Error extracting plan")
                    if os.path.exists(plan_file):
                        plan = load_input_json(plan_file)
                    else:
                        plan = None
                    batch_responses.append(plan)
                    save_plan_json(messages, os.path.join(task_dir, "messages.json"))
                # save intermediates
                if not use_proprietary_model and not args.use_icl:
                    logger.info(f"保存中间结果到: {os.path.join(args.env_output_dir, f'intermediate_{global_step}.pt')}")
                    torch.save({"query_tensors": query_tensors, "response_tensors": response_tensors, "batch_responses": batch_responses}, os.path.join(args.env_output_dir, f"intermediate_{global_step}.pt"))

            # 准备计划并运行环境
            batch_rewards = [] # (bs)
            batch_rewards_ref = [] # (bs)
            if os.path.exists(os.path.join(args.env_output_dir, f"intermediate_{global_step}_rewards.pt")):
                logger.info(f"加载奖励: {os.path.join(args.env_output_dir, f'intermediate_{global_step}_rewards.pt')}")
                batch_rewards = torch.load(os.path.join(args.env_output_dir, f"intermediate_{global_step}_rewards.pt"), weights_only=False)["batch_rewards"]
            else:
                for task, plan in zip(tasks_batch, batch_responses):

                    # 重置环境并执行计划
                    env.reset(task, task_dir=os.path.join(args.env_output_dir, f"global_step_{global_step}", f"Task_{task['id']}"), reset_questinos=args.use_icl, task_ids=[task["id"] for task in tasks_batch])
                    observation, reward_dict, done, info = env.step(plan, return_false=args.use_icl, return_ref_reward=args.use_icl)
                    reward = reward_dict.get("reward", 0.0)
                    ref_reward = reward_dict.get("ref_reward", 0.0)
                    task_infos[task['id']] = info.false_answers

                    batch_rewards.append(float(reward))
                    batch_rewards_ref.append(float(ref_reward))

                    logger.debug(f"任务 {task.get('id')}: 完成状态={done}, 奖励={reward:.4f}")

                    if done and reward > 0:
                        # 保存有效计划示例
                        with open(os.path.join(args.output_dir, f"good_plan_{task.get('id')}.json"), 'w', encoding='utf-8') as f:
                            json.dump(plan, f, ensure_ascii=False, indent=2)
                torch.save({"batch_rewards": batch_rewards}, os.path.join(args.env_output_dir, f"intermediate_{global_step}_rewards.pt"))

            # 计算平均奖励
            mean_reward = np.mean(batch_rewards) if batch_rewards else 0
            mean_reward_ref = np.mean(batch_rewards_ref) if batch_rewards_ref else 0
            logger.info(f"步骤 {global_step}: 平均奖励 = {mean_reward:.4f} (参考奖励 = {mean_reward_ref:.4f})")

            # 记录到wandb
            if args.use_wandb:
                wandb.log({
                    "global_step": global_step,
                    "mean_reward": mean_reward,
                    "mean_reward_ref": mean_reward_ref,
                    "max_reward": max(batch_rewards) if batch_rewards else 0,
                    "min_reward": min(batch_rewards) if batch_rewards else 0
                })

            gamma = 0.9
            # reward shaping: (bs) -> (bs * step_len) interleaved (r1, r1, r1, r1, r2, r2, r2, r2, ...)
            batch_rewards = np.array(batch_rewards)
            discounted_rewards = []
            if not use_proprietary_model:
                for reward in batch_rewards:
                    # 为每个步骤生成递减奖励
                    step_rewards = [torch.tensor(reward * (gamma ** (len(plan_template) - i)), dtype=torch.float32).to(ppo_trainer.accelerator.device) for i in range(len(plan_template))]
                    discounted_rewards.extend(step_rewards)
            # batch_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(ppo_trainer.accelerator.device)

            logger.debug(f"query_tensors: {len(query_tensors)}, response_tensors: {len(response_tensors)}, discounted_rewards: {len(discounted_rewards)}")
            logger.debug(f"query_tensors shape: {[t.shape for t in query_tensors]}")
            logger.debug(f"response_tensors shape: {[t.shape for t in response_tensors]}")
            logger.debug(f"discounted_rewards shape: {[t.shape for t in discounted_rewards]}")

            # 保存查询、响应和奖励到磁盘（JSON格式）
            if args.dry_run:
                dry_run_data = {
                    "step": global_step,
                    "tasks": [task.get('id', '0000') for task in tasks_batch],
                    "query_responses": [],
                    "rewards": batch_rewards.tolist() if isinstance(batch_rewards, np.ndarray) else batch_rewards,
                    "mean_reward": float(mean_reward)
                }

                # 为每个查询-响应对准备JSON数据
                for i, (query, response) in enumerate(zip(query_tensors, response_tensors)):
                    step_idx = i // len(plan_template)
                    plan_step = list(plan_template.keys())[i % len(plan_template)]

                    # 解码查询和响应文本
                    query_text = processor.decode(query, skip_special_tokens=True)
                    response_text = processor.decode(response[query.shape[0]:], skip_special_tokens=True)

                    dry_run_data["query_responses"].append({
                        "task_id": tasks_batch[step_idx].get('id', '0000'),
                        "plan_step": plan_step,
                        "query": query_text,
                        "response": response_text,
                        "reward": float(discounted_rewards[i].cpu().numpy()) if hasattr(discounted_rewards[i], 'cpu') else float(discounted_rewards[i])
                    })

                # 保存JSON文件
                dry_run_file = os.path.join(args.output_dir, f"dry_run_step_{global_step}.json")
                with open(dry_run_file, 'w', encoding='utf-8') as f:
                    json.dump(dry_run_data, f, ensure_ascii=False, indent=2)
                logger.info(f"干运行结果已保存至: {dry_run_file}")
            # data_collator = DataCollatorForLanguageModeling(tokenizer=processor.tokenizer, mlm=False)
            # PPO更新
            if not args.dry_run:
                # Value_head forward
                if args.use_icl:
                    # revise plan_template for ICL
                    # plan_template["Detailed Storyboarding"] = plan_template["Detailed Storyboarding"] + ICL_META.format(info.get("false_answers", {}))
                    logger.info(f"Using ICL for training, task infos: {len(task_infos)}")
                else:
                    stats = ppo_trainer.step(query_tensors, response_tensors, discounted_rewards)
                    batch_data = {}
                    batch_data["query"] = query_tensors
                    batch_data["response"] = response_tensors
                    ppo_trainer.log_stats(stats, batch_data, discounted_rewards)

                # model.train()
                # for i in range(math.ceil(len(query_tensors) / args.mini_batch_size)):
                #     optimizer.zero_grad()
                #     start_idx = i * args.mini_batch_size
                #     end_idx = min((i + 1) * args.mini_batch_size, len(query_tensors))
                #     mini_query_tensors = query_tensors[start_idx:end_idx]
                #     mini_response_tensors = response_tensors[start_idx:end_idx]
                #     mini_discounted_rewards = discounted_rewards[start_idx:end_idx]

                #     # 修改这一行，使用torch.cat而不是torch.stack
                #     input_ids = [torch.cat([q, r]) for q, r in zip(mini_query_tensors, mini_response_tensors)]

                #     # truncate input_ids to max_length
                #     input_ids = [ids[-args.max_length:] for ids in input_ids]
                #     model_inputs = [{
                #         "input_ids": ids,
                #         "attention_mask": torch.ones_like(ids),
                #     } for ids in input_ids]
                #     model_inputs = data_collator(model_inputs).to(accelerator.device)

                #     logger.info(f"before forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, Going to forward inputs of size {model_inputs['input_ids'].shape}")
                #     logitss, loss, value = model(**model_inputs)

                #     logger.info(f"after forward: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

                #     logp = torch.nn.functional.log_softmax(logitss, dim=2)


                #     batch_input_ids = torch.stack(input_ids).to(accelerator.device)
                #     log_probs = torch.gather(logp, 2, batch_input_ids.unsqueeze(2)).squeeze(-1)

                #     rewards = torch.tensor(mini_discounted_rewards, device=accelerator.device).unsqueeze(1)  # shape: [bs, 1]
                #     loss = (-log_probs * rewards).sum()

                #     lr_schedular.step()
                #     accelerator.backward(loss)
                #     optimizer.step()
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                #     del logitss, log_probs, loss, value
                #     torch.cuda.empty_cache()
            else:
                logger.info("干运行模式：跳过PPO更新步骤")
                batch_data = {}
                batch_data["query"] = query_tensors
                batch_data["response"] = response_tensors

            # 更新全局步数
            global_step += 1

            # 保存模型检查点
            if not args.dry_run and (global_step % args.save_steps == 0 or mean_reward > best_reward):
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    logger.info(f"新的最佳奖励: {best_reward:.4f}")
                    if not use_proprietary_model and not args.use_icl:
                        ppo_trainer.save_pretrained(os.path.join(args.output_dir, "best_reward_model"))
                    # unwrapped_model = accelerator.unwrap_model(model)
                    # os.makedirs(os.path.join(args.output_dir, "best_reward_model"), exist_ok=True)
                    # unwrapped_model.save_pretrained(os.path.join(args.output_dir, "best_reward_model"))
                    # model.save_pretrained(os.path.join(args.output_dir, "best_reward_model"))

                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                if not use_proprietary_model and not args.use_icl:
                    ppo_trainer.save_pretrained(save_dir)
                # os.makedirs(save_dir, exist_ok=True)
                # unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model.save_pretrained(save_dir)
                # model.save_pretrained(save_dir)
                logger.info(f"保存检查点到: {save_dir}")

            # 评估
            if global_step % args.eval_steps == 0:
                logger.info(f"步骤 {global_step} - 运行评估...")
                # 从验证集中采样进行评估
                eval_batch_indices = np.random.choice(val_indices, min(args.batch_size, len(val_indices)), replace=False)
                eval_tasks_batch = [dataset[i] for i in eval_batch_indices]
                logger.info(f"评估任务批次: {eval_batch_indices}")

                # 这里可以添加验证集评估逻辑
                # 例如：运行与训练相同的生成和奖励计算步骤，但不更新模型

                if args.use_wandb:
                    wandb.log({
                        "global_step": global_step,
                        "eval_step": True
                        # 这里可以添加验证集的评估指标
                    })
    else:
        ppo_trainer.train()
    # 保存最终模型
    if not use_proprietary_model and not args.use_icl:
        ppo_trainer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    # os.makedirs(os.path.join(args.output_dir, "final_model"), exist_ok=True)
    # model.save_pretrained(os.path.join(args.output_dir, "final_model"))
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

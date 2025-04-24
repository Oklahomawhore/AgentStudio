#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY="你的wandb_key"  # 替换为你的wandb key

# 项目路径
PROJECT_DIR="/data/wangshu/wangshu_code/ISG"
OUTPUT_DIR="$PROJECT_DIR/ISV_train/outputs/qwen2-5-vl-ppo-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR

# 训练参数
MODEL_NAME="Qwen/Qwen2.5-VL-32B-AWQ"
DATASET_PATH="$PROJECT_DIR/ISV_eval/NovelConditionedVGen/instance_tasks.json"
BATCH_SIZE=8  # 全局批量大小
LOCAL_BATCH=1  # 每个GPU的批量大小
ACCUM_STEPS=2  # 梯度累积步数

# 启动日志
echo "Starting distributed training with Accelerate at $(date)" | tee -a $OUTPUT_DIR/training.log
echo "Output directory: $OUTPUT_DIR" | tee -a $OUTPUT_DIR/training.log
echo "Using model: $MODEL_NAME" | tee -a $OUTPUT_DIR/training.log

# 使用Accelerate启动分布式训练
accelerate launch --mixed_precision bf16 --num_processes 8 --num_machines 1 \
    --machine_rank 0 --main_process_port 29500 \
    $PROJECT_DIR/ISV_train/train_agent.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --env_output_dir "$OUTPUT_DIR/env_results" \
    --batch_size $BATCH_SIZE \
    --mini_batch_size $LOCAL_BATCH \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --learning_rate 1.5e-5 \
    --ppo_epochs 4 \
    --max_steps 1000 \
    --save_steps 20 \
    --eval_steps 20 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --wandb_project "video_agent_ppo" \
    --generation_mode "t2v" \
    --max_length 2048 \
    --min_length 512 \
    2>&1 | tee -a $OUTPUT_DIR/training.log

echo "Training completed at $(date)" | tee -a $OUTPUT_DIR/training.log

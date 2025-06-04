#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=4,5,6
export WANDB_API_KEY="e6e375cc17f1bdca8c5976d19fc8de07a33daeeb"  # 替换为你的wandb key

# 使用相对路径设置项目目录
# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 项目根目录是脚本目录的上一级
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# 训练参数
# MODEL_NAME="doubao-1-5-thinking-vision-pro-250428"  # 替换为你的模型名称
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"  # 替换为你的模型名称
# MODEL_NAME="Qwen/Qwen2-0.5B-Instruct"

# 提取MODEL_NAME的最后一部分（如果包含"/"）
MODEL_NAME_SHORT=$(basename $MODEL_NAME)
OUTPUT_DIR="$PROJECT_DIR/ISV_train/outputs/${MODEL_NAME_SHORT}-ppo-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR
ADAPTER_NAME="$PROJECT_DIR/ISV_train/outputs/sft_stage2"
DATASET_PATH="$PROJECT_DIR/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json"
PLAN_TEMPLATE="$PROJECT_DIR/ISG_agent/Prompt/plan_template_singleround.json"
BATCH_SIZE=1  # 全局批量大小
LOCAL_BATCH=1  # 每个GPU的批量大小
ACCUM_STEPS=1  # 梯度累积步数

# 启动日志
echo "Starting distributed training with Accelerate at $(date)" | tee -a $OUTPUT_DIR/training.log
echo "Output directory: $OUTPUT_DIR" | tee -a $OUTPUT_DIR/training.log
echo "Using model: $MODEL_NAME" | tee -a $OUTPUT_DIR/training.log

# 启动训练
python $PROJECT_DIR/ISV_train/train_agent.py \
    --train_method "grpo" \
    --model_name_or_path $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --env_output_dir "$OUTPUT_DIR/env_results" \
    --batch_size $BATCH_SIZE \
    --mini_batch_size $LOCAL_BATCH \
    --gradient_accumulation_steps $ACCUM_STEPS \
    --learning_rate 1.5e-5 \
    --ppo_epochs 4 \
    --max_steps 10 \
    --save_steps 20 \
    --eval_steps 20 \
    --use_lora "True" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_qlora "False" \
    --quantization_bits 4 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --wandb_project "video_agent_ppo" \
    --use_wandb "False" \
    --generation_mode "t2v" \
    --max_length 2048 \
    --min_length 512 \
    --dry_run "False" \
    --plan_template $PLAN_TEMPLATE \
    --use_icl "False" \
    --regenerate_question "True" \
    --video_gen_api_base "http://localhost:7999" \
    2>&1 | tee -a $OUTPUT_DIR/training.log

echo "Training completed at $(date)" | tee -a $OUTPUT_DIR/training.log

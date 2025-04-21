#!/bin/bash

# Define datasets and output directory, consistent with 1_video_gen.sh
DATASETS=(
          "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/GPT-story/stories.json" 
          "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json" 
          "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/YouCook2/stories.json")

# Initialize RESULTS_DIR array to match DATASETS length
# RESULTS_DIR=(
#     "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/Vidu-1.5-GPT-story"
#     "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/Vidu-1.5-NovelConditionedVGen"
#     "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/Vidu-1.5-YouCook2"
# )

RESULTS_DIR=(
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_GPT-story"
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning"
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_YouCook2"
)

# Define evaluation results output directory
EVAL_RESULTS_DIR="/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results"

# Define model name
MODEL_NAME="ISG_agent"

# Process each dataset
for i in "${!DATASETS[@]}"; do
    # Get current dataset and result directory
    dataset="${DATASETS[$i]}"
    result_dir="${RESULTS_DIR[$i]}"
    
    # Extract dataset folder name as task name
    TASK_NAME=$(basename $(dirname "$dataset"))
    
    # Build question directory - prompt json path directory name plus instance_questions_deepseek
    QUESTIONS_DIR="/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/${TASK_NAME}/instance_questions_deepseek"
    
    echo "Processing evaluation for dataset: $dataset"
    echo "Task name: $TASK_NAME"
    echo "Results directory: $result_dir"
    echo "Questions directory: $QUESTIONS_DIR"
    
    # Run evaluation script
    python /data/wangshu/wangshu_code/ISG/ISV_eval/agent_academy.py \
        --results-dir "$result_dir" \
        --prompt-json "$dataset" \
        --questions-dir "$QUESTIONS_DIR" \
        --dataset-name "$TASK_NAME" \
        --model-name "$MODEL_NAME" \
        --output-dir "$EVAL_RESULTS_DIR" \
        --method "agent" 
    
    echo "Completed evaluation for $TASK_NAME"
    echo "------------------------"
done
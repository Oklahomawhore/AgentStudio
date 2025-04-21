#!/bin/bash

# Define datasets as in previous scripts
DATASETS=("/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/CVSV/stories.json" 
          "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/GPT-story/stories.json" 
          "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json" 
          "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/YouCook2/stories.json")

# Initialize RESULTS_DIR array to match DATASETS length
RESULTS_DIR=(
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_CVSV"
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_GPT-story"
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning"
    "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_YouCook2"
)

# Define evaluation results output directory
EVAL_RESULTS_DIR="/data/wangshu/wangshu_code/ISG/ISV_eval/eval_results"

# Process each dataset
for i in "${!DATASETS[@]}"; do
    # Get current dataset and result directory
    dataset="${DATASETS[$i]}"
    result_dir="${RESULTS_DIR[$i]}"
    
    # Extract dataset folder name as task name
    TASK_NAME=$(basename $(dirname "$dataset"))
    
    # Build question directory path
    QUESTIONS_DIR="/data/wangshu/wangshu_code/ISG/ISV_eval/questions/${TASK_NAME}/instance_questions_deepseek"
    
    echo "Evaluating results for dataset: $dataset"
    echo "Task name: $TASK_NAME"
    echo "Results directory: $result_dir"
    echo "Questions directory: $QUESTIONS_DIR"
    
    # Create output directory for this task's evaluation
    task_output_dir="${EVAL_RESULTS_DIR}/${TASK_NAME}"
    mkdir -p "$task_output_dir"
    
    # Run evaluation script
    python /data/wangshu/wangshu_code/ISG/ISV_eval/eval_instance.py \
        --prompt_path "$dataset" \
        --results_dir "$result_dir" \
        --questions_dir "$QUESTIONS_DIR" \
        --output_path "${task_output_dir}/evaluation_summary.json"
    
    echo "Completed evaluation for $TASK_NAME"
    echo "------------------------"
done

# Print summary of results across all datasets
echo "Evaluation complete for all datasets."
echo "Results saved in: $EVAL_RESULTS_DIR"

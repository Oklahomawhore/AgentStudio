OUT_DIR="/data/wangshu/wangshu_code/ISG/ISG_agent"
DATASETS=("/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/CVSV/stories.json" "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/GPT-story/stories.json" "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json" "/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/YouCook2/stories.json")
MODEL = "Vidu20"

# 处理每个数据集
for dataset in "${DATASETS[@]}"; do
    # 提取数据集文件夹名称作为任务名
    TASK_NAME=$(basename $(dirname "$dataset"))
    
    # 创建特定任务的输出目录
    TASK_OUT_DIR="$OUT_DIR/results_video_$MODEL_$TASK_NAME"
    mkdir -p "$TASK_OUT_DIR"
    
    echo "Processing dataset: $dataset"
    echo "Task name: $TASK_NAME"
    echo "Output directory: $TASK_OUT_DIR"
    
    # 运行处理脚本
    python /data/wangshu/wangshu_code/ISG/ISG_agent/PlanningAgentV2.py --input_json "$dataset" --out_dir "$TASK_OUT_DIR"
    
    echo "Completed processing $TASK_NAME"
    echo "------------------------"
done
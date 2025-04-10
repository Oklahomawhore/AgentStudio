#!/bin/bash

# 用法函数
usage() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -i, --input_dir    视频文件的输入目录 (必须)"
    echo "  -o, --output_dir   评估结果的输出目录 (默认: eval_results)"
    echo "  -m, --method       评估方法: agent 或 human (默认: agent)"
    echo "  -h, --help         显示此帮助信息"
    exit 1
}

# 设置默认值
INPUT_DIR=""
OUTPUT_DIR="eval_results"
METHOD="agent"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input_dir) INPUT_DIR="$2"; shift ;;
        -o|--output_dir) OUTPUT_DIR="$2"; shift ;;
        -m|--method) METHOD="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
    shift
done

# 检查必需参数
if [ -z "$INPUT_DIR" ]; then
    echo "错误: 必须提供输入目录"
    usage
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 '$INPUT_DIR' 不存在"
    exit 1
fi

# 检查方法是否有效
if [ "$METHOD" != "agent" ] && [ "$METHOD" != "human" ]; then
    echo "错误: 方法必须是 'agent' 或 'human'"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 创建汇总报告文件
SUMMARY_FILE="$OUTPUT_DIR/summary_report.txt"
echo "视频评估汇总报告 - $(date)" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "输入目录: $INPUT_DIR" >> "$SUMMARY_FILE"
echo "评估方法: $METHOD" >> "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 计数器
TOTAL=0
SUCCESS=0
FAILED=0

# 查找所有视频文件
echo "开始批量评估视频..."
VIDEO_FILES=$(find "$INPUT_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" -o -name "*.mkv" -o -name "*.wmv" -o -name "*.flv" \))

# 计算总数
TOTAL=$(echo "$VIDEO_FILES" | wc -l)
echo "找到 $TOTAL 个视频文件"

# 处理每个视频文件
for VIDEO_PATH in $VIDEO_FILES; do
    VIDEO_NAME=$(basename "$VIDEO_PATH")
    echo ""
    echo "正在评估 ($((SUCCESS+FAILED+1))/$TOTAL): $VIDEO_NAME"
    echo "----------------------------------------"
    
    # 调用agent_academy.py进行评估
    RESULT=$(python /data/wangshu/wangshu_code/ISG/ISV_eval/agent_academy.py --video_path "$VIDEO_PATH" --method "$METHOD")
    
    # 检查执行状态
    if [ $? -eq 0 ]; then
        SUCCESS=$((SUCCESS+1))
        echo "$VIDEO_NAME: $RESULT" >> "$SUMMARY_FILE"
        echo "✓ 评估完成: $RESULT"
    else
        FAILED=$((FAILED+1))
        echo "$VIDEO_NAME: 评估失败" >> "$SUMMARY_FILE"
        echo "✗ 评估失败"
    fi
done

# 生成总结
echo "" >> "$SUMMARY_FILE"
echo "评估总结" >> "$SUMMARY_FILE"
echo "----------------" >> "$SUMMARY_FILE"
echo "总计视频: $TOTAL" >> "$SUMMARY_FILE"
echo "成功评估: $SUCCESS" >> "$SUMMARY_FILE"
echo "失败评估: $FAILED" >> "$SUMMARY_FILE"

# 打印总结信息
echo ""
echo "评估完成!"
echo "----------------"
echo "总计视频: $TOTAL"
echo "成功评估: $SUCCESS"
echo "失败评估: $FAILED"
echo "汇总报告保存至: $SUMMARY_FILE"

exit 0

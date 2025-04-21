import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse
from collections import defaultdict
import re

plt.rcParams['font.family'] = 'SimHei'
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='可视化评估结果')
    parser.add_argument('--eval_dir', type=str, required=True, 
                        help='评估结果目录路径')
    parser.add_argument('--group_by', type=str, default='all',
                        choices=['all', 'model', 'custom'],
                        help='如何分组数据: all-所有平均, model-按模型名称, custom-自定义正则表达式')
    parser.add_argument('--pattern', type=str, default=r'(sora|dalle|claude|gpt)',
                        help='用于分组的正则表达式模式，仅在 group_by=custom 时使用')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件的路径，如不指定则显示结果')
    parser.add_argument('--title', type=str, default='评估结果比较',
                        help='输出标题')
    parser.add_argument('--format', type=str, default='graph',
                        choices=['graph', 'table'],
                        help='输出格式: graph-图表, table-Markdown表格')
    return parser.parse_args()


def extract_model_name(dirname):
    """从目录名中提取模型名称"""
    model_patterns = {
        'sora': r'sora',
        'dalle': r'dalle',
        'sd': r'stable_diffusion|sd',
        'midjourney': r'midjourney|mj',
        'claude': r'claude',
        'gpt': r'gpt',
        'gemini': r'gemini',
        'llava': r'llava',
        'pika' : r'pika',
        'hunyuan' : r'hunyuan',
        'alpha': r'alpha',
        'ray': r'ray2',
        'pika2' : r'pika2',
        'wan2' : r'wan2',
    }
    
    for model, pattern in model_patterns.items():
        if re.search(pattern, dirname.lower()):
            return model
    return 'unknown'


def extract_custom_group(dirname, pattern):
    """使用自定义正则表达式从目录名中提取分组"""
    match = re.search(pattern, dirname.lower())
    if match and match.groups():
        return match.group(1)
    return 'other'


def load_evaluation_results(eval_dir, group_by='all', pattern=None):
    """
    加载评估结果
    
    Args:
        eval_dir: 包含评估结果的目录
        group_by: 分组方式 ('all', 'model', 'custom')
        pattern: 自定义正则表达式模式
        
    Returns:
        grouped_results: 按组分类的结果字典
    """
    grouped_results = defaultdict(list)
    
    # 遍历评估目录
    for subdir in os.listdir(eval_dir):
        subdir_path = os.path.join(eval_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        # 确定分组键
        if group_by == 'all':
            group_key = 'all'
        elif group_by == 'model':
            group_key = extract_model_name(subdir)
        elif group_by == 'custom':
            group_key = extract_custom_group(subdir, pattern)
        else:
            group_key = 'unknown'
            
        # 查找并解析 report.json 文件
        for file in os.listdir(subdir_path):
            if file.endswith('_report.json'):
                file_path = os.path.join(subdir_path, file)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        grouped_results[group_key].append(data)
                except Exception as e:
                    print(f"无法解析 {file_path}: {e}")
                    
    return grouped_results


def calculate_average_scores(results_list):
    """计算一组结果的平均分"""
    if not results_list:
        return {}
        
    avg_scores = {
        'part_1': 0,
        'part_2': 0,
        'part_3': 0,
        'part_4': 0, 
        'part_5': 0,
        'overall': 0
    }
    
    for result in results_list:
        for part in avg_scores.keys():
            if part in result:
                avg_scores[part] += result[part]['percentage']
                
    for part in avg_scores.keys():
        if results_list:  # 避免除零错误
            avg_scores[part] /= len(results_list)
            
    return avg_scores


def generate_markdown_table(grouped_results, title='评估结果比较'):
    """生成Markdown格式的表格"""
    if not grouped_results:
        return "没有找到评估结果数据"
    
    # 计算每个组的平均分数
    avg_scores = {}
    for group, results in grouped_results.items():
        avg_scores[group] = calculate_average_scores(results)
    
    # 准备表格
    markdown = f"# {title}\n\n"
    
    # 表头
    headers = ["组别", "Part 1", "Part 2", "Part 3", "Part 4", "Part 5", "Overall"]
    markdown += "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # 表格内容
    parts = ['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'overall']
    
    # 按组名字母顺序排序
    sorted_groups = sorted(grouped_results.keys())
    
    for group in sorted_groups:
        scores = avg_scores[group]
        row = [group.capitalize()]
        for part in parts:
            if part in scores:
                row.append(f"{scores[part]:.1f}%")
            else:
                row.append("N/A")
        markdown += "| " + " | ".join(row) + " |\n"
    
    # 添加样本数量信息
    markdown += "\n## 样本数量\n\n"
    markdown += "| 组别 | 样本数 |\n"
    markdown += "| --- | --- |\n"
    
    for group in sorted_groups:
        markdown += f"| {group.capitalize()} | {len(grouped_results[group])} |\n"
    
    return markdown


def plot_results(grouped_results, title='评估结果比较', output=None):
    """绘制评估结果图表"""
    # 准备数据
    groups = list(grouped_results.keys())
    if not groups:
        print("没有找到任何数据进行可视化")
        return
    
    # 计算每个组的平均分数
    avg_scores = {}
    for group, results in grouped_results.items():
        avg_scores[group] = calculate_average_scores(results)
    
    # 准备图表数据
    categories = ['Part 1', 'Part 2', 'Part 3', 'Part 4', 'Part 5', 'Overall']
    group_data = []
    
    for group in sorted(groups):  # 按字母顺序排序组名
        group_scores = []
        for part in ['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'overall']:
            if part in avg_scores[group]:
                group_scores.append(avg_scores[group][part])
            else:
                group_scores.append(0)
        group_data.append(group_scores)
    
    # 设置图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置x轴位置
    x = np.arange(len(categories))
    width = 0.8 / len(groups)  # 条形宽度
    
    # 绘制条形图
    for i, (group, data) in enumerate(zip(sorted(groups), group_data)):
        offset = width * i - width * (len(groups) - 1) / 2
        bars = ax.bar(x + offset, data, width, label=group.capitalize())
        
        # 在条形上方添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    # 添加标签、标题和图例
    ax.set_xlabel('评估类别')
    ax.set_ylabel('得分百分比 (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 105)  # 将y轴限制在0-105%之间
    
    # 添加网格线以便于查看
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存或显示图表
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {output}")
    else:
        plt.show()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载评估结果
    grouped_results = load_evaluation_results(
        args.eval_dir, 
        group_by=args.group_by, 
        pattern=args.pattern
    )
    
    # 打印找到的分组
    print(f"找到的分组: {list(grouped_results.keys())}")
    for group, results in grouped_results.items():
        print(f"组 '{group}': {len(results)} 个评估结果")
    
    # 根据指定格式输出结果
    if args.format == 'graph':
        plot_results(grouped_results, title=args.title, output=args.output)
    else:  # table format
        markdown_table = generate_markdown_table(grouped_results, title=args.title)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(markdown_table)
            print(f"Markdown表格已保存至: {args.output}")
        else:
            print("\n" + markdown_table)


if __name__ == "__main__":
    main()



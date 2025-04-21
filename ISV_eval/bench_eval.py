import json
import os
import glob
from scipy.stats import pearsonr
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体
def load_vbench_results(file_path):
    """
    加载VBench评测结果
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 提取视频文件路径和对应的评分
    results_dict = {}
    video_results = data['imaging_quality'][1]
    for item in video_results:
        if 'video_path' in item:
            # 提取文件名作为key
            video_path = item['video_path']
            video_name = os.path.basename(video_path)
            results_dict[video_name] = item['video_results']
    
    return results_dict

def load_local_reports(eval_results_dir):
    """
    加载本地报告文件
    """
    results_dict = {}
    report_files = glob.glob(os.path.join(eval_results_dir, '*/*_report.json'))
    
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            # 假设文件名在report中有存储，或者从路径中提取
            video_name_dir = os.path.basename(os.path.dirname(report_file))
            video_name = None
            
            # 从报告中尝试获取文件名
            if 'file_path' in data:
                video_name = os.path.basename(data['file_path'])
            else:
                # 如果报告中没有文件路径，则使用目录名作为视频名称
                video_name = video_name_dir + '.mp4'
            
            # 假设得分存储在report的某个字段中
            # 这里需要根据实际的JSON结构调整
            score = data['overall']['percentage']
            
            results_dict[video_name] = score  # 假设得分是百分比，转换为0-1范围
            
        except Exception as e:
            print(f"处理文件 {report_file} 时出错: {str(e)}")
    
    return results_dict

def calculate_correlation(dict1, dict2):
    """
    计算两个字典中共有键的值之间的皮尔逊相关系数
    """
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    if not common_keys:
        print("没有找到共有的视频文件")
        return None, []
    
    print(f"找到 {len(common_keys)} 个共有视频文件")
    
    values1 = []
    values2 = []
    common_items = []
    
    for key in common_keys:
        values1.append(dict1[key])
        values2.append(dict2[key])
        common_items.append((key, dict1[key], dict2[key]))
    
    correlation, p_value = pearsonr(values1, values2)
    
    print(f"皮尔逊相关系数: {correlation}")
    print(f"p值: {p_value}")
    
    return (correlation, p_value), common_items

def plot_correlation(common_items, correlation, output_file=None):
    """
    绘制相关性散点图
    """
    x = [item[1] for item in common_items]
    y = [item[2] for item in common_items]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6)
    
    # 添加1:1参考线
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    
    plt.title(f'相关性散点图 (Pearson r = {correlation[0]:.4f}, p = {correlation[1]:.4e})')
    plt.xlabel('Vbench评分')
    plt.ylabel('AgentAcademy评分')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"散点图已保存至 {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='计算两个视频评测基准之间的相关性')
    parser.add_argument('--vbench', default='/data/wangshu/wangshu_code/VBench/evaluation_results/results_2025-04-11-15:04:14_eval_results.json',
                        help='VBench评测结果的JSON文件路径')
    parser.add_argument('--reports', default='eval_results',
                        help='包含本地评测报告的目录')
    parser.add_argument('--output', default='correlation_plot.png',
                        help='输出散点图的文件路径')
    
    args = parser.parse_args()
    
    print(f"加载VBench结果: {args.vbench}")
    vbench_results = load_vbench_results(args.vbench)
    print(f"找到 {len(vbench_results)} 个VBench评分")
    
    print(f"加载本地报告: {args.reports}")
    local_results = load_local_reports(args.reports)
    print(f"找到 {len(local_results)} 个本地评分")
    
    correlation, common_items = calculate_correlation(vbench_results, local_results)
    
    if correlation and len(common_items) > 0:
        plot_correlation(common_items, correlation, args.output)
        
        # 保存结果到CSV
        output_csv = 'correlation_results.csv'
        with open(output_csv, 'w') as f:
            f.write("视频文件,基准1评分,基准2评分\n")
            for item in common_items:
                f.write(f"{item[0]},{item[1]},{item[2]}\n")
        print(f"详细结果已保存至 {output_csv}")

if __name__ == "__main__":
    main()

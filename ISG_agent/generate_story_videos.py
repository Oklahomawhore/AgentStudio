import os
import json
import time
import requests
import logging
from pathlib import Path
import sys
import dotenv

from api_interface import kling_text2video_agent

# 添加ISG_agent目录到Python路径以便导入app_service_forVidu模块
sys.path.append(os.path.join(os.path.dirname(__file__), "ISG_agent"))
from util import download_video_and_save_as_mp4

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('story_video_generation.log'),
        logging.StreamHandler()
    ]
)

# 加载环境变量
dotenv.load_dotenv()

# 定义路径
STORIES_PATH = '/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/NovelConditionedVGen/video_storytelling_novel.json'
OUTPUT_DIR = '/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/Kling1.6-NovelConditionedVGen'

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    """主函数：读取故事并生成视频"""
    try:
        # 确保输出目录存在
        ensure_directory_exists(OUTPUT_DIR)
        
        # 读取故事数据
        with open(STORIES_PATH, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        
        logging.info(f"加载了 {len(stories)} 个故事")
        
        # 处理每个故事
        for story in stories:
            story_id = story['id']
            task_dir = os.path.join(OUTPUT_DIR, f"Task_{story_id.zfill(4)}")
            ensure_directory_exists(task_dir)
            
            output_path = os.path.join(task_dir, f"story_{story_id}_video.mp4")
            
            # 检查视频是否已存在
            if os.path.exists(output_path):
                logging.info(f"故事 {story_id} 的视频已存在，跳过生成")
                continue
            
            # 生成并保存视频
            url = "http://localhost:7903/generate_video"  # Backend Flask API endpoint for text2video
            data = {
                "prompt": story["Query"][0]['content'],
                "seconds_per_screenshot" : 1
            }
            # return "videos/ChGIFWdqfWwAAAAAAAqcQg-0_raw_video_1.mp4", [test_img]
            response = requests.post(url, json=data)
            if response.status_code == 200:
                video_path, screenshots = response.json().get("video_file", ""), response.json().get("screenshots",[])
            else:
                raise Exception(f"Error {response.status_code}: {response.text}")
            
            # move video to output path
            if video_path:
                
                os.rename(video_path, output_path)
                logging.info(f"故事 {story_id} 的视频已保存到 {output_path}")
            else:
                logging.error(f"无法为故事 {story_id} 生成视频")
            
            if video_path:
                logging.info(f"成功生成故事 {story_id} 的视频")
            else:
                logging.error(f"无法为故事 {story_id} 生成视频")
        
        logging.info("所有故事处理完成")
        
    except Exception as e:
        logging.error(f"执行脚本时发生错误: {str(e)}")

if __name__ == "__main__":
    main()

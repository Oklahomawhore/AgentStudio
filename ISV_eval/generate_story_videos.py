import os
import json
import time
import requests
import logging
from pathlib import Path
import sys
import dotenv

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
STORIES_PATH = '/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/GPT-story/stories.json'
OUTPUT_DIR = '/data/wangshu/wangshu_code/ISG/ISV_eval/datasets/GPT-story-vidu-1.5/'

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def generate_story_video(story, output_path):
    """为故事生成视频并保存到指定路径"""
    story_id = story['id']
    content = story['Query'][0]['content']
    
    # 提取标题作为提示，如果没有标题则使用内容的前100个字符
    prompt = content
    
    logging.info(f"生成视频 - 故事ID: {story_id}, 提示: {prompt}")
    
    try:
        # 生成视频
        task_id = generate_video_request({"prompt": prompt})
        if not task_id:
            logging.error(f"无法为故事 {story_id} 创建视频生成任务")
            return None
        
        # 轮询视频状态
        logging.info(f"视频生成中 - 故事ID: {story_id}, 任务ID: {task_id}")
        video_url = None
        retry_count = 0
        max_retries = 20  # 增加重试次数以处理较长的生成时间
        
        while retry_count < max_retries:
            video_url = check_text2video_status(task_id)
            if video_url:
                break
            logging.info(f"等待视频生成 - 故事ID: {story_id}, 尝试次数: {retry_count+1}/{max_retries}")
            retry_count += 1
            time.sleep(30)  # 每30秒检查一次
        
        if not video_url:
            logging.error(f"视频生成超时或失败 - 故事ID: {story_id}")
            return None
        
        # 下载视频
        logging.info(f"视频已生成，准备下载 - 故事ID: {story_id}")
        video_path, screenshots = download_video_and_save_as_mp4(video_url, output_file=output_path, seconds_per_screenshot=1)
        
        logging.info(f"视频已保存 - 故事ID: {story_id}, 路径: {video_path}")
        return video_path
        
    except Exception as e:
        logging.error(f"处理故事 {story_id} 时发生错误: {str(e)}")
        return None

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
            video_path = generate_story_video(story, output_path)
            
            if video_path:
                logging.info(f"成功生成故事 {story_id} 的视频")
            else:
                logging.error(f"无法为故事 {story_id} 生成视频")
        
        logging.info("所有故事处理完成")
        
    except Exception as e:
        logging.error(f"执行脚本时发生错误: {str(e)}")

if __name__ == "__main__":
    main()

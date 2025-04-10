import os
import io
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
import base64 
from typing import Tuple, List, Dict
from threading import Lock
import glob
import dotenv
from pathlib import Path
import json
import time


dotenv.load_dotenv()

from tqdm import tqdm
from openai import OpenAI
from dashscope.utils.oss_utils import preprocess_message_element

make_dir_lock = Lock()

batch_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def format_time_to_hhmmss(seconds):
    """
    将秒数转换为 HH:mm:ss 格式
    
    Args:
        seconds: 秒数（可以是浮点数）
        
    Returns:
        HH:mm:ss 格式的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def download_video(video_url, file_name=None, save_directory="videos") -> str:
    """
    Downloads a video from the given URL and saves it as an MP4 file.

    Args:
        video_url (str): URL of the video to download.
        save_directory (str): Directory to save the downloaded video.
        filename (str): Name of the saved MP4 file.

    Returns:
        str: Path to the saved video file.
    """
    try:
        # Ensure the save directory exists
        with make_dir_lock:
            os.makedirs(save_directory, exist_ok=True)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        # Download the video
        response = requests.get(video_url, stream=True, headers=headers)
        response.raise_for_status()

        # Define the full file path
        video_path = os.path.join(save_directory, file_name if file_name is not None else video_url.split("/")[-1].split('?')[0])

        # Save the video
        with open(video_path, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=8192):
                video_file.write(chunk)

        print(f"Video successfully downloaded and saved to {video_path}")
        return video_path

    except Exception as e:
        print(f"Error in downloading video: {e}")
        return None

def capture_screenshots(video_path: str, seconds_per_screenshot: float = 1.0, end_time: float = None) -> List[str]:
    """
    Captures screenshots from a video file at specified intervals.

    Args:
        video_path (str): Path to the video file.
        seconds_per_screenshot (float): Interval in seconds for capturing screenshots.
        end_time (float): Time in seconds to stop capturing screenshots. If None, captures until end of video.

    Returns:
        List[str]: List of base64-encoded screenshots.
    """
    try:
        base64_screenshots = []
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        frame_interval = round(fps * seconds_per_screenshot)  # Frames to skip for each screenshot
        frame_interval = max(1, frame_interval)  # Ensure at least 1 frame interval

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps
            if end_time and current_time > end_time:
                break

            if frame_count % frame_interval == 0:  # Capture frame
                # Convert frame (numpy array) to PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                pil_image = Image.fromarray(frame_rgb)

                # Encode the screenshot to base64
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                base64_screenshots.append(img_base64)

            frame_count += 1

        cap.release()
        print(f"Captured {len(base64_screenshots)} screenshots.")
        return base64_screenshots

    except Exception as e:
        print(f"Error in capturing screenshots: {e}")
        return []

def download_video_and_save_as_mp4(video_url, file_name=None, save_directory="videos", seconds_per_screenshot=1) -> Tuple[str, List[str]]:
    """
    Downloads a video and captures screenshots using the separate download and screenshot functions.

    Args:
        video_url (str): URL of the video to download.
        save_directory (str): Directory to save the downloaded video and screenshots.
        filename (str): Name of the saved MP4 file.
        seconds_per_screenshot (int): Interval in seconds for capturing screenshots.

    Returns:
        Tuple[str, List[str]]: Tuple containing video file path and list of base64-encoded screenshots.
    """
    video_path = download_video(video_url, file_name, save_directory)
    if video_path:
        screenshots = capture_screenshots(video_path, seconds_per_screenshot)
        return video_path, screenshots
    return None, []


def process_image(file_path, output_path):
    # Load the image
    img = Image.open(file_path)
    width, height = img.size
    aspect_ratio = width / height
    max_size = 10 * 1024 * 1024  # 10 MB

    # Check file size
    if os.path.getsize(file_path) > max_size:
        raise ValueError("The file size exceeds 10MB. Consider compressing it first.")

    # Check resolution
    if width < 300 or height < 300:
        # Upscale to meet resolution
        scale_factor = max(300 / width, 300 / height)
        img = img.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.Resampling.BICUBIC)

    # Check aspect ratio
    if aspect_ratio < 1 / 2.5 or aspect_ratio > 2.5:
        # Adjust aspect ratio by cropping
        target_aspect_ratio = max(min(aspect_ratio, 2.5), 1 / 2.5)
        if target_aspect_ratio > aspect_ratio:
            new_height = int(width / target_aspect_ratio)
            crop = (0, (height - new_height) // 2, width, (height + new_height) // 2)
        else:
            new_width = int(height * target_aspect_ratio)
            crop = ((width - new_width) // 2, 0, (width + new_width) // 2, height)
        img = img.crop(crop)

    # Save the processed image
    img.save(output_path, optimize=True)

def add_fix_to_filename(file_path, suffix="fix") -> str:
    # Split the file path into directory, filename, and extension
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)

    # Append '_fix' to the filename
    new_filename = f"{name}_{suffix}{ext}"

    # Construct the new file path
    new_file_path = os.path.join(directory, new_filename)
    
    print(f"Renamed: {file_path} -> {new_file_path}")

    return new_file_path


import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from qwen_vl_utils import process_vision_info


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
        {
            "type": "video",
            "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
            "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
            'fps': 3.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
        }]
    },
]


def prepare_message_for_vllm(content_messages, tmp_dir="tmp"):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                video_message = [{'content': [part_message]}]
                if not os.path.exists(os.path.join(tmp_dir, os.path.basename(part_message["video"]).split('.')[0])):
                    os.makedirs(os.path.join(tmp_dir, os.path.basename(part_message["video"]).split('.')[0]))
                else:
                    # load processed screenshots
                    file_list = glob.glob(os.path.join(tmp_dir, os.path.basename(part_message["video"]).split('.')[0], "screenshot*"))
                    if len(file_list) > 0:
                        base64_frames = file_list
                        part_message = {
                            "type": "video",
                            "video": base64_frames,
                        }
                        new_content_list.append(part_message)  
                        continue
                image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                assert video_inputs is not None, "video_inputs should not be None"
                video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                fps_list.extend(video_kwargs.get('fps', []))

                
                # encode image with base64
                base64_frames = []
                for i, frame in enumerate(video_input):
                    img = Image.fromarray(frame)
                    file_path = os.path.join(tmp_dir, os.path.basename(part_message["video"]).split('.')[0], f"screenshot_{i}.jpg")
                    img.save(file_path, format="jpeg")
                    base64_frames.append(file_path)

                part_message = {
                    "type": "video",
                    "video": base64_frames,
                }
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}


def extract_scene(input_video, output_path, start_time, end_time):
    """
    Extract a scene from the video and save it to a file.

    Args:
        input_video (str): Path to the input video
        output_path (str): Path to save the output clip
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
    """
    duration = end_time - start_time
    os.system(f'{os.getenv("FFMPEG_PATH")} -i "{input_video}" -ss {start_time} -t {duration} -c:v libx264 -c:a aac -strict experimental -b:a 128k "{output_path}" -y -loglevel error')
    return output_path


def cut_video_into_scenes(video_path, threshold=10.0, min_scene_length=1, output_dir=None, plot_results=False):
    """
    Cut video into scenes based on pixel distance spikes between consecutive frames.

    Args:
        video_path (str): Path to the input video file
        threshold (float): Threshold for scene change detection (higher = fewer scenes)
        min_scene_length (int): Minimum number of frames for a scene
        output_dir (str, optional): Directory to save scene clips. If None, clips are not saved.
        plot_results (bool): Whether to generate and display a plot of frame differences

    Returns:
        list: A list of dictionaries containing scene information:
              - 'start_frame': starting frame number
              - 'end_frame': ending frame number
              - 'start_time': starting time in seconds
              - 'end_time': ending time in seconds
              - 'path': path to the saved clip (if output_dir is provided)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize variables for scene detection
    prev_frame = None
    frame_diffs = []
    avg_diffs = []
    thresholds = []
    frame_numbers = []
    scene_boundaries = [0]  # Start with the first frame

    # Process video frames to detect scene changes
    print(f"Analyzing video for scene changes: {video_path}")
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and resize for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width//4, height//4))

        if prev_frame is not None:
            # Calculate mean absolute difference between frames
            diff = cv2.absdiff(gray, prev_frame)
            mean_diff = np.mean(diff)
            frame_diffs.append(mean_diff)
            frame_numbers.append(i+1)

            # Check for scene change
            if len(frame_diffs) > 1:
                # Use a moving average to smooth out noise
                avg_diff = sum(frame_diffs[-3:]) / min(len(frame_diffs), 3)
                avg_diffs.append(avg_diff)
                thresholds.append(threshold * avg_diff)

                # If the difference is above threshold and enough frames have passed since last scene
                if mean_diff > threshold and i - scene_boundaries[-1] >= min_scene_length:
                    scene_boundaries.append(i)
                    print(f"Scene change detected at frame {i} (time: {i/fps:.2f}s)")

        prev_frame = gray

    # Add the last frame as a scene boundary
    if scene_boundaries[-1] != frame_count - 1:
        scene_boundaries.append(frame_count - 1)

    # Plot results if requested
    if plot_results:
        plt.figure(figsize=(15, 8))

        # Plot frame differences
        plt.plot(frame_numbers, frame_diffs, 'b-', alpha=0.6, linewidth=1, label='帧间差异')

        # Plot moving average if we have enough frames
        if len(avg_diffs) > 0:
            # We need to align the x-axis for avg_diffs (starts at frame 3)
            avg_diff_frames = frame_numbers[1:len(avg_diffs)+2]
            plt.plot(avg_diff_frames, avg_diffs, 'g-', linewidth=2, label='滑动平均 (3帧)')
            plt.plot(avg_diff_frames, thresholds, 'r--', linewidth=2, label=f'阈值 (x{threshold})')

        # Mark scene boundaries
        for boundary in scene_boundaries:
            if boundary > 0 and boundary < frame_count - 1:  # Skip first and last
                plt.axvline(x=boundary, color='red', linestyle='-', alpha=0.5)
                plt.text(boundary, plt.ylim()[1]*0.9, f"{boundary}\n({boundary/fps:.1f}s)",
                         horizontalalignment='center', color='red', fontsize=8)

        # Add labels and title
        plt.xlabel('帧编号')
        plt.ylabel('平均像素差异')
        plt.title(f'场景检测分析 - {os.path.basename(video_path)}\n检测到 {len(scene_boundaries)-1} 个场景')
        plt.legend()
        plt.grid(True, alpha=0.3)


        # Save plot if output directory exists
        if output_dir:
            plot_path = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + "_scene_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"分析图表已保存至: {plot_path}")

        plt.tight_layout()
        plt.savefig('scene_analysis.png')

    # Create scenes list
    scenes = []
    video_name = os.path.basename(video_path).split('.')[0]

    for i in range(len(scene_boundaries) - 1):
        start_frame = scene_boundaries[i]
        end_frame = scene_boundaries[i+1]

        scene_info = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_frame / fps,
            'end_time': end_frame / fps,
        }

        # Save scene clip if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            scene_path = os.path.join(output_dir , f"{video_name}_scene_{i+1:03d}.mp4")

            # Extract and save the scene
            extract_scene(video_path, scene_path, scene_info['start_time'], scene_info['end_time'])
            scene_info['path'] = scene_path
        else:
            scene_info['path'] = None
        scene_info['duration'] = scene_info['end_time'] - scene_info['start_time']
        scene_info['scene_number'] = i + 1

        scenes.append(scene_info)

    cap.release()
    print(f"检测到 {len(scenes)} 个场景")
    return scenes

def upload_file(file_path):
    print(f"正在上传包含请求信息的JSONL文件...")
    file_object = batch_client.files.create(file=Path(file_path), purpose="batch")
    print(f"文件上传成功。得到文件ID: {file_object.id}\n")
    return file_object.id

def create_batch_job(input_file_id):
    print(f"正在基于文件ID，创建Batch任务...")
    # 请注意：选择Embedding文本向量模型进行调用时,endpoint的值需填写"/v1/embeddings"
    batch = batch_client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"Batch任务创建完成。 得到Batch任务ID: {batch.id}\n")
    return batch.id
def check_job_status(batch_id):
    print(f"正在检查Batch任务状态...")
    batch = batch_client.batches.retrieve(batch_id=batch_id)
    print(f"Batch任务状态: {batch.status}\n")
    return batch.status
def get_output_id(batch_id):
    print(f"正在获取Batch任务中执行成功请求的输出文件ID...")
    batch = batch_client.batches.retrieve(batch_id=batch_id)
    print(f"输出文件ID: {batch.output_file_id}\n")
    return batch.output_file_id
def get_error_id(batch_id):
    print(f"正在获取Batch任务中执行错误请求的输出文件ID...")
    batch = batch_client.batches.retrieve(batch_id=batch_id)
    print(f"错误文件ID: {batch.error_file_id}\n")
    return batch.error_file_id
def download_results(output_file_id, output_file_path):
    print(f"正在打印并下载Batch任务的请求成功结果...")
    content = batch_client.files.content(output_file_id)
    # 打印部分内容以供测试
    print(f"打印请求成功结果的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存结果文件至本地
    content.write_to_file(output_file_path)
    print(f"完整的输出结果已保存至本地输出文件result.jsonl\n")
def download_errors(error_file_id, error_file_path):
    print(f"正在打印并下载Batch任务的请求失败信息...")
    content = batch_client.files.content(error_file_id)
    # 打印部分内容以供测试
    print(f"打印请求失败信息的前1000个字符内容: {content.text[:1000]}...\n")
    # 保存错误信息文件至本地
    content.write_to_file(error_file_path)
    print(f"完整的请求失败信息已保存至本地错误文件error.jsonl\n")
def _create_jsonl_request(id, model, messages):
    """创建符合DashScope批处理API要求的JSONL请求格式"""
    return {
        "custom_id" : str(id),
        "method" : "POST",
        "url" : "/v1/chat/completions",
        "body" : {
            "model": model,
            "messages": messages,
        }
    }

def _preprocess_messages(model: str, messages: List[dict],
                             api_key: str):
        """
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": ""},
                        {"text": ""},
                    ]
                }
            ]
        """
        has_upload = False
        for message in messages:
            content = message['content']
            for elem in content:
                if not isinstance(elem,
                                  (int, float, bool, str, bytes, bytearray)):
                    
                    is_upload = preprocess_message_element(
                        model, elem, api_key)
                    if is_upload and not has_upload:
                        has_upload = True
        return has_upload

def inference_batch(messages: List[List[Dict]], model: str, job_id=None) -> List[str]:
    # prepare input jsonl
    input_file = os.path.join(os.path.dirname(__file__),'tmp', f"input_{str(job_id)}.jsonl")
    output_file = os.path.join(os.path.dirname(__file__),'tmp', f"output_{str(job_id)}.jsonl")
    error_file = os.path.join(os.path.dirname(__file__),'tmp', f"error_{str(job_id)}.jsonl")
    if os.path.exists(output_file):
        results = []
        with open(output_file, "r") as f:
            for line in f:
                # 解析JSONL格式的输出
                # 这里假设每行都是一个有效的JSON对象
                # 如果有多行输出，可以根据需要进行处理
                try:
                    data = json.loads(line)
                    if data["response"]["status_code"] == 200:
                        results.append(data["response"]["body"]["choices"][0]["message"]["content"])
                    else:
                        results.append(None)
                except json.JSONDecodeError:
                    print(f"无法解析的行: {line}")
                    continue
        return results
    count = 0
    for message  in messages:
        uploaded = _preprocess_messages(model, message, os.getenv("DASHSCOPE_API_KEY"))
        if uploaded:
            count += 1
    print(f"uploaded {count} in {len(messages)} messages.")
    with open(input_file, "w") as f:
        for i, message in enumerate(messages):
            # 创建正确格式的JSONL请求
            jsonl_request = _create_jsonl_request(i, model, message)
            f.write(f"{json.dumps(jsonl_request, ensure_ascii=False)}\n")
    
    try:
        # Step 1: 上传包含请求信息的JSONL文件,得到输入文件ID,如果您需要输入OSS文件,可将下行替换为：input_file_id = "实际的OSS文件URL或资源标识符"
        input_file_id = upload_file(input_file)
        # Step 2: 基于输入文件ID,创建Batch任务
        batch_id = create_batch_job(input_file_id)
        # Step 3: 检查Batch任务状态直到结束
        status = ""
        while status not in ["completed", "failed", "expired", "cancelled"]:
            status = check_job_status(batch_id)
            print(f"等待任务完成...")
            time.sleep(10)  # 等待10秒后再次查询状态
        # 如果任务失败,则打印错误信息并退出
        if status == "failed":
            batch = batch_client.batches.retrieve(batch_id)
            print(f"Batch任务失败。错误信息为:{batch.errors}\n")
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            return
        # Step 4: 下载结果：如果输出文件ID不为空,则打印请求成功结果的前1000个字符内容，并下载完整的请求成功结果到本地输出文件;
        # 如果错误文件ID不为空,则打印请求失败信息的前1000个字符内容,并下载完整的请求失败信息到本地错误文件.
        output_file_id = get_output_id(batch_id)
        if output_file_id:
            download_results(output_file_id, output_file)
            # Load output jsonl and return list of str
            results = []
            with open(output_file, "r") as f:
                for line in f:
                    # 解析JSONL格式的输出
                    # 这里假设每行都是一个有效的JSON对象
                    # 如果有多行输出，可以根据需要进行处理
                    try:
                        data = json.loads(line)
                        if data["response"]["status_code"] == 200:
                            results.append(data["response"]["body"]["choices"][0]["message"]["content"])
                        else:
                            results.append(None)
                    except json.JSONDecodeError:
                        print(f"无法解析的行: {line}")
                        continue
            return results
        error_file_id = get_error_id(batch_id)
        if error_file_id:
            download_errors(error_file_id, error_file)
            print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"参见错误码文档: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
import os
import io
import cv2
import requests
from PIL import Image
import base64 
from typing import Tuple, List
from threading import Lock
import re
import dotenv
from urllib.parse import urlparse
import time
import uuid
import json
import hashlib, pickle

dotenv.load_dotenv()

make_dir_lock = Lock()

def replace_characters_in_content(content, characters):
    """
    Replace all character references in the content with their descriptions.
    Supports both <#character_name#> and <character_name> patterns.

    Args:
        content (str): The input string containing character references.
        characters (dict): A dictionary mapping character names to descriptions.

    Returns:
        str: The content with character references replaced by descriptions.
    """
    # Regex pattern to match <#character_name#> or <character_name>
    pattern = r"<#(.*?)#>|<(.*?)>"

    character_list = []
    # Function to replace the matched pattern with the description
    def replace(match):
        # Extract the character name from either group
        character_name = match.group(1) or match.group(2)
        if character_name in characters and character_name not in character_list:
            character_list.append(character_name)
        # Replace with the description or keep the original pattern if not found
        return f"{character_name} ({characters.get(character_name, f'<{character_name}>')})"

    # Use re.sub to replace all occurrences of the pattern
    return re.sub(pattern, replace, content), character_list

# Function to download the generated image and encode it to base64
def IMGGEN_download_image_and_convert_to_base64(image_url):
    try:
        image_data = requests.get(image_url).content
        image_base64 = base64.b64encode(image_data).decode('utf-8')  # Encoding the image in base64
        return image_base64
    except Exception as e:
        print(f"Error in downloading and encoding image: {e}")
        return None

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
        
        url_filename = "video_" + str(int(time.time())) + '_' + str(uuid.uuid4()) + ".mp4"

        video_path = os.path.join(save_directory, file_name if file_name is not None else url_filename)

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

class GENERATION_MODE:
    I2V = "i2v" # image to video (using last frame or generated storyboard as first frame)
    T2V = "t2v" # text to video (cross scene consistency by character description)
    R2V = "r2v" # reference to video


def get_aliyun_sign_url(file_path, folder=None):
    import oss2
    from oss2.credentials import EnvironmentVariableCredentialsProvider
    
    # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

    # 填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    endpoint = "https://oss-cn-shanghai.aliyuncs.com"

    # 填写Endpoint对应的Region信息，例如cn-hangzhou。注意，v4签名下，必须填写该参数
    region = "cn-shanghai"
    # 填写Bucket名称，例如examplebucket。
    bucketName = "video-storage-6999"
    # 创建Bucket实例，指定存储空间的名称和Region信息。
    bucket = oss2.Bucket(auth, endpoint, bucketName, region=region)

    # 本地文件的完整路径
    local_file_path = file_path  

    # 填写Object完整路径，完整路径中不能包含Bucket名称。例如exampleobject.txt。
    if folder:
        objectName = folder + '/' + os.path.basename(local_file_path)
    else:
        objectName = os.path.basename(local_file_path)

    # 使用put_object_from_file方法将本地文件上传至OSS
    bucket.put_object_from_file(objectName, local_file_path)

    return bucket.sign_url('GET', objectName, 2 * 60 * 60)  # 2小时有效期

# Helper function to generate a unique hash based on the prompt
def generate_hash(prompt):
    # Generate a SHA-256 hash of the prompt string
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

# Save results to disk using pickle
def save_to_disk(content, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(content, f)

# Check if the result exists, if so return the content
def load_from_disk(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_error_file(task_dir, error_message):
    os.makedirs(task_dir, exist_ok=True)
    error_file = os.path.join(task_dir, "error.log")
    with open(error_file, "a") as f:
        f.write(error_message + "\n")

def load_input_json(json_file:str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_input_txt(txt_file:str):
    with open(txt_file, 'r') as f:
        data = f.read()
    return data
def save_plan_json(json_data,file):
    dir_name = os.path.dirname(file)
    print(f"Directory: {dir_name}")
    os.makedirs(dir_name, exist_ok=True)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

def save_result_json(json_data,task_dir):
    os.makedirs(task_dir, exist_ok=True)
    with open(f'{task_dir}/result.json', 'w') as f:
        json.dump(json_data, f, indent=4)
        
        
def get_image_media_type(image_data):
    """
    Detects the image media type based on the magic number in the binary data.
    """
    if image_data[:4] == b'\x89PNG':
        return "image/png"
    elif image_data[:2] == b'\xFF\xD8':
        return "image/jpeg"
    else:
        raise ValueError("Unsupported image format")
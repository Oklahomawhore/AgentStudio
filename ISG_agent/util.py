import os
import io
import cv2
import requests
from PIL import Image
import base64 
from typing import Tuple, List

def download_video_and_save_as_mp4(video_url,file_name=None, save_directory="videos", seconds_per_screenshot=1) -> Tuple[str, List[str]]:
    """
    Downloads a video from the given URL, saves it as an MP4 file, and captures screenshots.

    Args:
        video_url (str): URL of the video to download.
        save_directory (str): Directory to save the downloaded video and screenshots.
        filename (str): Name of the saved MP4 file.
        seconds_per_screenshot (int): Interval in seconds for capturing screenshots.

    Returns:
        dict: A dictionary containing the video file path and a list of base64-encoded screenshots.
    """
    try:
        # Ensure the save directory exists
        os.makedirs(save_directory, exist_ok=True)

        
        # Download the video
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        # Define the full file path
        video_path = os.path.join(save_directory, file_name if file_name is not None else video_url.split("/")[-1])


        # Save the video
        with open(video_path, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=8192):
                video_file.write(chunk)

        print(f"Video successfully downloaded and saved to {video_path}")

        # Capture screenshots
        base64_screenshots = []
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
        frame_interval = int(fps * seconds_per_screenshot)  # Frames to skip for each screenshot

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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

        return video_path, base64_screenshots

    except Exception as e:
        print(f"Error in downloading video or capturing screenshots: {e}")
        return None


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
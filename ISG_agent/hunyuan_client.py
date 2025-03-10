import requests
import json
import os
from typing import Optional, Dict, Any, Union, BinaryIO
from pathlib import Path
from typing import Tuple, List
from util import capture_screenshots
import uuid


class HunyuanClient:
    def __init__(self, base_url: str = "http://localhost:6006"):
        self.base_url = base_url.rstrip('/')

    def check_status(self) -> Dict[str, Any]:
        """
        Check service status including GPU availability and loaded models.
        
        Returns:
            Dict containing status information
        """
        url = f"{self.base_url}/status"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def generate_from_text(self,
                          task: str,
                          prompt: str,
                          size: str = "1280*720",
                          frame_num: Optional[int] = None,
                          sample_steps: Optional[int] = None,
                          sample_shift: Optional[float] = None,
                          sample_solver: str = "unipc",
                          sample_guide_scale: float = 5.0,
                          base_seed: int = -1,
                          output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate video or image from text prompt.
        
        Args:
            task: One of "t2v-14B", "t2v-1.3B", or "t2i-14B"
            prompt: Text description of desired output
            size: Output dimensions
            frame_num: Number of frames (default: 81 for video, 1 for image)
            sample_steps: Number of sampling steps
            sample_shift: Sampling shift factor
            sample_solver: "unipc" or "dpm++"
            sample_guide_scale: Classifier free guidance scale
            base_seed: Random seed (-1 for random)
            output_path: If provided, save output to this path
            
        Returns:
            Bytes of generated content if output_path is None, else path to saved file
        """
        url = f"{self.base_url}/generate/text"
        
        payload = {
            "task": task,
            "prompt": prompt,
            "size": size,
            "sample_solver": sample_solver,
            "sample_guide_scale": sample_guide_scale,
            "base_seed": base_seed
        }
        
        # Add optional parameters if provided
        if frame_num is not None:
            payload["frame_num"] = frame_num
        if sample_steps is not None:
            payload["sample_steps"] = sample_steps
        if sample_shift is not None:
            payload["sample_shift"] = sample_shift

        response = requests.post(url, json=payload)
        print(response.status_code)
        response.raise_for_status()
        
        if output_path:
            ext = ".mp4" if task.startswith("t2v") else ".png"
            output_path = str(Path(output_path).with_suffix(ext))
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        return response.content

    def generate_from_image(self,
                          image_path: str,
                          prompt: str,
                          size: str = "1280*720",
                          frame_num: int = 81,
                          sample_steps: int = 40,
                          sample_shift: Optional[float] = None,
                          sample_solver: str = "unipc",
                          sample_guide_scale: float = 5.0,
                          base_seed: int = -1,
                          output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate video from input image and text prompt.
        
        Args:
            image_path: Path to input image file
            prompt: Text description of desired output
            size: Output dimensions (use * for separator, e.g. "1280*720")
            frame_num: Number of frames
            sample_steps: Number of sampling steps
            sample_solver: "unipc" or "dpm++"
            sample_guide_scale: Classifier free guidance scale
            base_seed: Random seed (-1 for random)
            output_path: If provided, save output to this path
            
        Returns:
            Bytes of generated video if output_path is None, else path to saved file
        """
        url = f"{self.base_url}/generate/image"
        
        config = {
            "task": "i2v-14B",
            "prompt": prompt,
            "size": size
        }

        files = {
            'image': open(image_path, 'rb'),
        }

        response = requests.post(url, files=files,data={'config': json.dumps(config)})
        response.raise_for_status()
        
        if output_path:
            output_path = str(Path(output_path).with_suffix('.mp4'))
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return output_path
        return response.content
    
    def text2video(self, prompt_list: Union[str, List[str]], seconds_per_screenshot: int = 1) -> Tuple[str, List[str]]:
        """
        Generate video from text prompt(s).
        
        Args:
            prompt_list: Single prompt string or list of prompts
            seconds_per_screenshot: Interval for taking screenshots
            
        Returns:
            Tuple of (video_file_path, list_of_screenshot_paths)
        """
        video_path = f"videos/{uuid.uuid4()}.mp4"
        video_bytes = self.generate_from_text(task='t2v',prompt=prompt_list, output_path=video_path)
        screenshots = capture_screenshots(video_path, seconds_per_screenshot)
        return video_path, screenshots
    

    def image2video(self, 
                    image_url: str, 
                    prompt: str, 
                    seconds_per_screenshot: int = 1) -> Tuple[str, List[str]]:
        """
        Generate video from input image and text prompt.
        
        Args:
            image_url: URL or path to input image
            prompt: Text description for the video
            seconds_per_screenshot: Interval for taking screenshots
            
        Returns:
            Tuple of (video_file_path, list_of_screenshot_paths)
        """
        video_path = f"videos/{uuid.uuid4()}.mp4"
        video_bytes = self.generate_from_image(prompt=prompt,image_path= image_url, output_path=video_path)
        screenshots = capture_screenshots(video_path, seconds_per_screenshot)
        return video_path, screenshots
# Example usage
if __name__ == "__main__":
    client = HunyuanClient()
    
    try:
        # Check service status
        status = client.check_status()
        print("Service status:", status)
        
        
        # Generate video from image
        video_bytes = client.generate_from_image(
            image_path="nezha.jpg",
            prompt="哪吒从天空中抓到一只如意金箍棒，一棒劈开了东方明珠",
            output_path="image2video_output.mp4"
        )
        print("Image-to-video generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}") 
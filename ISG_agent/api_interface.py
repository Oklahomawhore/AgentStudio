import requests
from retry import retry
from typing import Tuple, List
from img import test_img

BASE_URL = "http://localhost:7899"

Flux_URL = "http://localhost:7900"

MORPH_URL = "http://localhost:7901"

KLING_URL = "http://localhost:7903"

Hunyuan_URL = "http://localhost:7905"

Hunyuan_i2v_URL = "http://localhost:7906"

Vidu_URL = "http://localhost:7988"


@retry(tries=3, delay=1)
def generate_video_agent(prompt_list,seconds_per_screenshot=1):
    url = f"{BASE_URL}/generate_video"
    data = {
        "prompt_list": prompt_list,
        "seconds_per_screenshot": seconds_per_screenshot
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("screenshots", [])
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

@retry(tries=3, delay=1)
def generate_image_agent(prompt):
    # url = f"{BASE_URL}/generate_image"
    url = f"{Flux_URL}/generate_image"
    data = {
        "prompt": prompt
    }
    response = requests.post(url, json=data)
    #FIXME: return mock data
    if response.status_code == 200:
        return response.json().get("generated_image_base64", "")
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


@retry(tries=3, delay=1)
def edit_image_agent(prompt, image_input):
    url = f"{BASE_URL}/edit_image"
    data = {
        "prompt": prompt,
        "image_input": image_input,
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("edited_image_base64", "")
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


@retry(tries=3, delay=1)
def generate_3d_video_agent(input_list, screenshots_per_second=1,proportions=[1/12, 2/12, 10/12, 11/12]):
    print(input_list), print(screenshots_per_second),print(proportions)
    url = f"{BASE_URL}/generate_3d_video"
    data = {
        "input_list": input_list,
        "screenshots_per_second": screenshots_per_second,
        "proportions": proportions
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("screenshots", [])
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


    
@retry(tries=3, delay=1)
def morph_images_agent(img_path1, img_path2, prompt):
    url = f"{MORPH_URL}/morph"
    data = {
        "img_path1": img_path1,
        "img_path2": img_path2,
        "prompt": prompt
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("frames", [])
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
    
#KLING Service


def kling_img2video_agent(image_url, prompt, seconds_per_screenshot=1) -> Tuple[str, List[str]]:
    url = f"{Vidu_URL}/generate_image2video"  # Backend Flask API endpoint for img2video
    data = {
        "image": image_url,
        "prompt": prompt,
        "seconds_per_screenshot" : seconds_per_screenshot
    }
    # return "videos/ChGIFWdqfWwAAAAAAAqcQg-0_raw_video_1.mp4", [test_img]
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("video_file", ""), response.json().get("screenshots",[])
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def kling_ref2video_agent(image_url, prompt, seconds_per_screenshot=1) -> Tuple[str, List[str]]:
    url = f"{Vidu_URL}/generate_reference2video"  # Backend Flask API endpoint for img2video
    data = {
        "image": image_url,
        "prompt": prompt,
        "seconds_per_screenshot" : seconds_per_screenshot,
        "resolution" : '720p'
    }
    # return "videos/ChGIFWdqfWwAAAAAAAqcQg-0_raw_video_1.mp4", [test_img]
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("video_file", ""), response.json().get("screenshots",[])
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def kling_text2video_agent(prompt_list, seconds_per_screenshot=1):
    url = f"{Hunyuan_URL}/generate_video"  # Backend Flask API endpoint for text2video
    data = {
        "prompt": prompt_list,
        "seconds_per_screenshot" : seconds_per_screenshot
    }
    # return "videos/ChGIFWdqfWwAAAAAAAqcQg-0_raw_video_1.mp4", [test_img]
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("video_file", ""), response.json().get("screenshots",[])
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


def kling_imggen_agent(prompt):
    url = f"{KLING_URL}/generate_image"  # Backend Flask API endpoint for imggen
    data = {
        "prompt": prompt,
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json().get("generated_image_base64", "")
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")
import requests
import time
import base64
from flask import Flask, request, jsonify
from threading import Lock
import dotenv
import os
from util import download_video_and_save_as_mp4, process_image, add_fix_to_filename
import base64

dotenv.load_dotenv()

app = Flask(__name__)

# Lock for handling concurrent video generation requests
text2video_generator_lock = Lock()

# API credentials
TEXT_API_URL = 'http://zzzzapi.com/kling/v1/videos/text2video'
TEXT_STATUS_URL = 'http://zzzzapi.com/kling/v1/videos/text2video/'

img2video_generator_lock = Lock()

# API credentials
IMG_API_URL = 'http://zzzzapi.com/kling/v1/videos/image2video'
IMG_STATUS_URL = 'http://zzzzapi.com/kling/v1/videos/image2video/'

img_generator_lock = Lock()

# API credentials with updated names
IMGGEN_API_URL = 'http://zzzzapi.com/kling/v1/images/generations'
IMGGEN_STATUS_URL = 'http://zzzzapi.com/kling/v1/images/generations/'



# Function to generate video request
def generate_video_request(data):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': os.getenv('KLING_API_KEY'),
        }

        # Prepare the data to send in the POST request
        payload = {
            "model_name": "kling-v1",
            "prompt": data['prompt'],
            "negative_prompt": data.get('negative_prompt', ''),
            "cfg_scale": data.get('cfg_scale', 0.5),
            "mode": data.get('mode', 'std'),
            "aspect_ratio": data.get('aspect_ratio', '16:9'),
            "duration": str(data.get('duration', 5)),
            "callback_url": data.get('callback_url', ''),
            "external_task_id": data.get('external_task_id', '')
        }

        # POST request to generate the video
        response = requests.post(TEXT_API_URL, json=payload, headers=headers)
        response_data = response.json()

        # Check if the request was successful
        if response_data['code'] == 0:
            task_id = response_data['data']['task_id']
            return task_id
        else:
            return None
    except Exception as e:
        print(f"Error in generating video: {e}")
        return None

# Function to check video status
def check_text2video_status(task_id):
    try:
        headers = {
            'Authorization': os.getenv('KLING_API_KEY'),
            'Content-Type': 'application/json',
        }

        # Send GET request to check video status
        response = requests.get(f"{TEXT_STATUS_URL}{task_id}", headers=headers)
        response_data = response.json()

        if response_data['code'] == 0:
            task_status = response_data['data']['task_status']
            if task_status == 'succeed':
                # Retrieve the video URL
                video_url = response_data['data']['task_result']['videos'][0]['url']
                return video_url
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error in checking video status: {e}")
        return None

@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with a "prompt" key.'}), 400

        with text2video_generator_lock:
            # Step 1: Request to generate video and get task_id
            task_id = generate_video_request(data)

            if not task_id:
                return jsonify({'error': 'Failed to initiate video generation.'}), 500

            # Step 2: Periodically check for task completion status
            task_status = None
            video_url = None
            retry_count = 0
            max_retries = 8
            while task_status != 'succeed':
                task_status = check_text2video_status(task_id)

                # If the task has succeeded, we retrieve the video URL
                if task_status is not None:
                    video_url = task_status
                    break
                else:
                    # If task is still running, wait before checking again
                    # If task is still running, wait before checking again (check every 1 minute)
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(60)

        if video_url:
            # Step 3: Download video and convert to base64
            video_path, screenshots = download_video_and_save_as_mp4(video_url, seconds_per_screenshot=data['seconds_per_screenshot'])

            if video_path:
                return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
            else:
                return jsonify({'error': 'Failed to download or encode the video.'}), 500
        else:
            return jsonify({'error': 'Task did not succeed or video not available.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


## Image 2 video

# Function to generate video request for image2video
def generate_image2video_request(data):
    try:
        headers = {
            'Authorization': os.getenv('KLING_API_KEY'),
            'Content-Type': 'application/json'
        }

        try:
            process_image(data["image"], add_fix_to_filename(data["image"], "processed"))
        except Exception as e:
            raise Exception(f"Error in processing image: {e}")
        # Read the file in binary mode and encode it to Base64
        with open(data['image'], "rb") as image_file:
            base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        
        # Prepare the data to send in the POST request
        payload = {
            "model_name": "kling-v1",
            "mode": data.get('mode', 'std'),
            "duration": str(data.get('duration', 5)),
            "image": base64_encoded_image,
            "prompt": data['prompt'],
            "cfg_scale": data.get('cfg_scale', 0.5),
            "static_mask": data.get('static_mask', ''),
            "dynamic_masks": data.get('dynamic_masks', [])
        }

        # POST request to generate the video with extended timeout
        response = requests.post(IMG_API_URL, json=payload, headers=headers, timeout=300)  # 5 minutes timeout
        response_data = response.json()

        # Check if the request was successful
        if response_data['code'] == 0:
            task_id = response_data['data']['task_id']
            return task_id
        else:
            return None
    except Exception as e:
        print(f"Error in generating video: {e}")
        return None

# Function to check video status
def check_img2video_status(task_id):
    try:
        headers = {
            'Authorization': os.getenv('KLING_API_KEY'),
            'Content-Type': 'application/json'
        }

        # Send GET request to check video status with extended timeout
        response = requests.get(f"{IMG_STATUS_URL}{task_id}", headers=headers, timeout=300)  # 5 minutes timeout
        response_data = response.json()

        if response_data['code'] == 0:
            task_status = response_data['data']['task_status']
            if task_status == 'succeed':
                # Retrieve the video URL
                video_url = response_data['data']['task_result']['videos'][0]['url']
                return video_url
            else:
                return None
        else:
            return None
    except Exception as e:
        print(f"Error in checking video status: {e}")
        return None

@app.route('/generate_image2video', methods=['POST'])
def generate_image2video():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'image' not in data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "image" and "prompt" keys.'}), 400

        with img2video_generator_lock:
            # Step 1: Request to generate video and get task_id
            task_id = generate_image2video_request(data)

            if not task_id:
                return jsonify({'error': 'Failed to initiate video generation.'}), 500

            # Step 2: Periodically check for task completion status
            task_status = None
            video_url = None
            retry_count = 0
            max_retries = 8
            while retry_count < max_retries and task_status != 'succeed':
                task_status = check_img2video_status(task_id)

                # If the task has succeeded, we retrieve the video URL
                if task_status is not None:
                    video_url = task_status
                    break
                else:
                    # If task is still running, wait before checking again (check every 1 minute)
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(60)

        if video_url:
            # Step 3: Download video and convert to base64
            video_path, screenshots = download_video_and_save_as_mp4(video_url, seconds_per_screenshot=data['seconds_per_screenshot'])

            if video_path:
                return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
            else:
                return jsonify({'error': 'Failed to download or encode the video.'}), 500
        else:
            return jsonify({'error': 'Task did not succeed or video not available.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

### IMG GEN


# Function to generate image
def IMGGEN_generate_image_request(data):
    try:
        headers = {
            'Authorization': os.getenv('KLING_API_KEY'),
            'Content-Type': 'application/json'
        }

        payload = {
            "model": data.get('model', 'kling-v1'),
            "prompt": data['prompt'],
            "negative_prompt": data.get('negative_prompt', ''),
            "image": data.get('image', ''),
            "image_fidelity": data.get('image_fidelity', 0.5),
            "n": data.get('n', 1),
            "aspect_ratio": data.get('aspect_ratio', '16:9'),
            "callback_url": data.get('callback_url', '')
        }

        # Send POST request to generate image
        response = requests.post(IMGGEN_API_URL, json=payload, headers=headers, timeout=300)  # 5 minutes timeout
        response_data = response.json()

        if response_data['code'] == 0:
            task_id = response_data['data']['task_id']
            return task_id
        else:
            return None
    except Exception as e:
        print(f"Error in generating image: {e}")
        return None

# Function to check image generation status
def IMGGEN_check_image_status(task_id):
    try:
        headers = {
            'Authorization': os.getenv('KLING_API_KEY'),
            'Content-Type': 'application/json'
        }

        # Send GET request to check task status
        response = requests.get(f"{IMGGEN_STATUS_URL}{task_id}", headers=headers, timeout=300)  # 5 minutes timeout
        response_data = response.json()

        if response_data['code'] == 0:
            task_status = response_data['data']['task_status']
            if task_status == 'succeed':
                # Retrieve the image URL
                image_url = response_data['data']['task_result']['images'][0]['url']
                return {"task_status" : task_status, "image_url" : image_url}
            else:
                return {"task_status" : task_status, "image_url" : ""}
        else:
            return {"task_status" : response["code"], "image_url" : ""}
    except Exception as e:
        print(f"Error in checking image status: {e}")
        return {"task_status" : "Exception", "image_url" : "", "Exception" : str(e)}

# Function to download the generated image and encode it to base64
def IMGGEN_download_image_and_convert_to_base64(image_url):
    try:
        image_data = requests.get(image_url).content
        image_base64 = base64.b64encode(image_data).decode('utf-8')  # Encoding the image in base64
        return image_base64
    except Exception as e:
        print(f"Error in downloading and encoding image: {e}")
        return None

# Flask route to handle image generation requests
@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" key.'}), 400
        with img_generator_lock:
            # Step 1: Request to generate image and get task_id
            task_id = IMGGEN_generate_image_request(data)
    
            if not task_id:
                return jsonify({'error': 'Failed to initiate image generation.'}), 500
    
            # Step 2: Periodically check for task completion status
            task_status = None
            image_url = None
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries and task_status != 'succeed':
                task_status = IMGGEN_check_image_status(task_id)
    
                # If the task has succeeded, retrieve the image URL
                if task_status['task_status'] == 'succeed':
                    image_url = task_status['image_url']
                    break
                elif task_status['task_status'] == 'Exception':
                    return jsonify({'error' : task_status['Exception']}), 503
                else:
                    # If task is still processing, wait before checking again (check every 1 minute)
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(10)
        if image_url:
            # Step 3: Download image and convert to base64
            base64_image = IMGGEN_download_image_and_convert_to_base64(image_url)

            if base64_image:
                return jsonify({'generated_image_base64': base64_image}), 200
            else:
                return jsonify({'error': 'Failed to download or encode the image.'}), 500
        else:
            return jsonify({'error': 'Task did not succeed or image not available.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7903, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(port=args.port, host='0.0.0.0')
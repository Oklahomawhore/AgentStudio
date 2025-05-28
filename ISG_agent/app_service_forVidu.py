import requests
import time
import base64
from flask import Flask, request, jsonify
from threading import Lock
import dotenv
import os
from util import download_video_and_save_as_mp4, process_image, add_fix_to_filename
import base64
import json
import os.path
from logging import Logger

from util import get_aliyun_sign_url

dotenv.load_dotenv()

app = Flask(__name__)

logging = Logger(__name__)

# API endpoints for Vidu
TEXT_API_URL = 'https://api.vidu.cn/ent/v2/text2video'
TEXT_STATUS_URL = 'https://api.vidu.cn/ent/v2/tasks/'

# Function to generate video request
def generate_video_request(data):
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {os.getenv("VIDU_API_KEY")}',
        }

        # Prepare the data to send in the POST request
        payload = {
            "model": "viduq1",
            "style": "general",
            "prompt": data["prompt"],
            "duration": "5",
            "seed": "2025",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "movement_amplitude": "auto"
        }

        # POST request to generate the video
        response = requests.post(TEXT_API_URL, json=payload, headers=headers)
        response_data = response.json()

        # Check if the request was successful
        if response.status_code == 200:
            task_id = response_data.get('task_id')
            if task_id:
                return task_id
            else:
                print("No task_id in response")
                return None
        else:
            print(f"Error: {response.status_code} - {response_data.get('message', 'Unknown error')}")
            # 生成 curl 命令用于调试
            auth_header = f"Token {os.getenv('VIDU_API_KEY')}"
            payload_str = json.dumps(payload).replace('"', '\\"')
            curl_command = f'curl -X POST "{TEXT_API_URL}" -H "Content-Type: application/json" -H "Authorization: {auth_header}" -d "{payload_str}"'
            print(f"DEBUG curl command: {curl_command}")
            return None
    except Exception as e:
        print(f"Error in generating video: {e}")
        return None

# Function to check video status
def check_text2video_status(task_id):
    try:
        headers = {
            'Authorization': f'Token {os.getenv("VIDU_API_KEY")}',
            'Content-Type': 'application/json',
        }

        # Send GET request to check video status
        response = requests.get(f"{TEXT_STATUS_URL}{task_id}/creations", headers=headers)
        response_data = response.json()

        if response.status_code == 200:
            status = response_data.get('state')
            if status == 'success':
                # Retrieve the video URL
                video_url = response_data.get('creations')[0].get('url')
                return video_url
            elif status == 'failed':
                print(f"Task {response_data['id']} failed: {response_data['err_code']}")
                return None
            else:
                # Still processing
                return None
        else:
            print(f"Error: {response.status_code} - {response.text}")
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

        # Step 1: Request to generate video and get task_id
        task_id = generate_video_request(data)
        if not task_id:
            return jsonify({'error': 'Failed to initiate video generation.'}), 500
            
        # Step 2: Periodically check for task completion status
        video_url = None
        retry_count = 0
        max_retries = 60
        while retry_count < max_retries:
            video_url = check_text2video_status(task_id)
            # If the task has succeeded, we retrieve the video URL
            if video_url is not None:
                break
            else:
                # If task is still running, wait before checking again
                retry_count += 1
                time.sleep(30)  # Check every 30 seconds

        if video_url:
            # Step 3: Download video and save
            seconds_per_screenshot = data.get('seconds_per_screenshot', 1)
            video_path, screenshots = download_video_and_save_as_mp4(video_url, seconds_per_screenshot=seconds_per_screenshot)

            if video_path:
                return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
            else:
                return jsonify({'error': 'Failed to download or encode the video.'}), 500
        else:
            return jsonify({'error': 'Task did not succeed or video not available.'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_image2video', methods=['POST'])
def generate_image2video():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'image' not in data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "image" and "prompt" keys.'}), 400

        # Process image first if needed
        try:
            if os.path.isfile(data["image"]):
                process_image(data["image"], add_fix_to_filename(data["image"], "processed"))
                image_path = add_fix_to_filename(data["image"], "processed")
            else:
                return jsonify({'error': 'Image file not found'}), 400
        except Exception as e:
            return jsonify({'error': f"Error in processing image: {e}"}), 500

        # Read the file in binary mode and encode it to Base64
        signed_url = get_aliyun_sign_url(image_path, folder='images')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {os.getenv("VIDU_API_KEY")}',
        }

        # Prepare the data to send in the POST request
        payload = {
            "model": "vidu2.0",
            "image": signed_url,
            "seed" : 2025,
            "duration" : 8
        }

        # POST request to generate the video
        # Note: Assuming Vidu has an image-to-video endpoint
        image2video_url = 'https://api.vidu.cn/ent/v2/img2video'
        response = requests.post(image2video_url, json=payload, headers=headers)
        response_data = response.json()

        # Check if the request was successful
        if response.status_code == 200:
            task_id = response_data.get('task_id')
            if not task_id:
                return jsonify({'error': 'Failed to get task_id for image-to-video generation.'}), 500
                
            # Check status periodically
            video_url = None
            retry_count = 0
            max_retries = 20
            while retry_count < max_retries:
                # Using the same status URL assuming it works for image2video tasks too
                status_response = requests.get(f"{TEXT_STATUS_URL}{task_id}/creations", headers=headers)
                status_data = status_response.json()
                
                if status_response.status_code == 200:
                    status = status_data.get('status')
                    if status == 'success':
                        video_url = status_data.get('createions')[0].get('url')
                        break
                    elif status == 'failed':
                        break
                
                # Wait before checking again
                retry_count += 1
                time.sleep(30)  # Check every 30 seconds

            if video_url:
                # Download video and save
                seconds_per_screenshot = data.get('seconds_per_screenshot', 1)
                video_path, screenshots = download_video_and_save_as_mp4(video_url, seconds_per_screenshot=seconds_per_screenshot)

                if video_path:
                    return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
                else:
                    return jsonify({'error': 'Failed to download or encode the video.'}), 500
            else:
                return jsonify({'error': 'Task did not succeed or video not available.'}), 500
        else:
            return jsonify({'error': f"API error: {response.status_code} - {response_data.get('message', 'Unknown error')}"}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_reference2video', methods=['POST'])
def generate_reference2video():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'image' not in data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "reference_video" and "prompt" keys.'}), 400
        elif len(data['image']) not in [1,2,3]:
            return jsonify({'error': 'Invalid input. Expected JSON with "image" key as a list of 1, 2 or 3 images.'}), 400

        encoded_images = []
        for i in range(len(data['image'])):
            if not os.path.isfile(data["image"][i]):
                return jsonify({'error': f'Image file {data["image"][i]} not found'}), 400
            
            # preprocess image file (resize, smaller)
            try:
                if os.path.isfile(data["image"][i]):
                    image_path =  add_fix_to_filename(data["image"][i], "processed")
                    process_image(data["image"][i], image_path)
                    
                else:
                    return jsonify({'error': 'Image file not found'}), 400
            except Exception as e:
                return jsonify({'error': f"Error in processing image: {e}"}), 500
            # Read the file in binary mode and encode it to Base64
            signed_url = get_aliyun_sign_url(image_path, folder='images')
            encoded_images.append(signed_url)
            logging.info(f"Encoded image {i} {data['image'][i]}: {signed_url}")
            
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {os.getenv("VIDU_API_KEY")}',
        }
        # Prepare the data to send in the POST request
        payload = {
            "model" : "vidu2.0",
            "prompt": data['prompt'],
            "images" : encoded_images,
            "resolution": data.get('resolution', '360p'),
            "duration": data.get('duration', 4),
            "seed" : 2025,
            "aspect_ratio" : data.get('aspect_ratio', '16:9'),
        }
        # POST request to generate the video using reference-to-video API
        reference2video_url = 'https://api.vidu.cn/ent/v2/reference2video'
        response = requests.post(reference2video_url, json=payload, headers=headers)
        response_data = response.json()
        logging.info(f"Create task response: {response_data}")
        # Check if the request was successful
        if response.status_code == 200:
            task_id = response_data.get('task_id')
            if not task_id:
                return jsonify({'error': 'Failed to get task_id for reference-to-video generation.'}), 500
            
            print(f"Created new task with ID: {task_id}")
        else:
            return jsonify({'error': f"API error: {response.status_code} - {response_data.get('message', 'Unknown error')}"}), response.status_code
    
        # 任务ID获取后，检查状态
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {os.getenv("VIDU_API_KEY")}',
        }
        
        # Check status periodically
        video_url = None
        retry_count = 0
        max_retries = 20
        status = ''

        while retry_count < max_retries:
            status_response = requests.get(f"{TEXT_STATUS_URL}{task_id}/creations", headers=headers)
            status_data = status_response.json()
            
            if status_response.status_code == 200:
                status = status_data.get('state')
                if status == 'success':
                    video_url = status_data.get('creations')[0].get('url')
                    logging.info(f"Task {task_id} succeeded: {video_url}")
                    break
                elif status == 'failed':
                    logging.error(f"Task {task_id} failed: {status_data}")
                    break
                elif status =='queueing':
                    logging.info(f"Task {task_id} is still in queue, waiting for processing...")
                elif status == 'processing':
                    logging.info(f"Task {task_id} is being processed...")
            else:
                logging.info(f"Error: {status_response.status_code} - {status_data.get('message', 'Unknown error')}")
            # Wait before checking again
            retry_count += 1
            time.sleep(30)  # Check every 30 seconds

        if video_url:
            # Download video and save
            seconds_per_screenshot = data.get('seconds_per_screenshot', 1)
            video_path, screenshots = download_video_and_save_as_mp4(video_url, seconds_per_screenshot=seconds_per_screenshot)

            if video_path:
                return jsonify({'video_file': video_path, 'screenshots': screenshots, 'task_id': task_id}), 200
            else:
                return jsonify({'error': 'Failed to download or encode the video.', 'task_id': task_id}), 500
        else:
            return jsonify({'error': f'Task did not succeed or video not available. Last status {status} Task id {task_id}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=6005, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(port=args.port, host='0.0.0.0')

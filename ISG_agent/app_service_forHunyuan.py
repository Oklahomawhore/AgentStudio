from util import download_video_and_save_as_mp4
import replicate
from flask import Flask, request, jsonify
import replicate
import dotenv
import uuid
import time
import threading
import os

dotenv.load_dotenv()

from dashscope import VideoSynthesis, ImageSynthesis
from http import HTTPStatus

from util import IMGGEN_download_image_and_convert_to_base64


app = Flask(__name__)


# Global variable to store the last execution time
last_execution_time = 0
execution_lock = threading.Lock()

@app.route('/generate_video', methods=['POST'])
def generate_video_replicate():
    global last_execution_time

    with execution_lock:
        current_time = time.time()
        time_since_last_call = current_time - last_execution_time
        
        if time_since_last_call < 2:
            time.sleep(2 - time_since_last_call)  # Wait for the remaining time

        last_execution_time = time.time()  # Update execution time after waiting

    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" key.'}), 400

        # Step 1: Call Replicate API to generate video
        try:
            # Wan2.1
            rsp = VideoSynthesis.call(model='wanx2.1-t2v-turbo',
                              prompt=data['prompt'],size='1280*720')
            if rsp.status_code == HTTPStatus.OK:
                output = rsp.output.video_url
                if output and isinstance(output, str):
                    print(output)
                else:
                    # 200, error message in output.message
                    return jsonify({'error': f'Failed to get valid video URL from API response {rsp.output}'}), 502
            else:
                print(f"Wanx API returned non 200 status code: {rsp.status_code} - {rsp.message}")
                return jsonify({'error': f'Wanx API error: {rsp.message}'}), rsp.status_code
        except Exception as e:
            print(f"Exception in calling Aliyun VideoSynthesis API: {e}")
            return jsonify({'error': f"Error in generating video with Aliyun API: {str(e)}"}), 501
        
        video_url = output
        unique_id = str(uuid.uuid4())
        
        # Step 2: Download video and save locally
        try:
            video_path, screenshots = download_video_and_save_as_mp4(
                video_url,
                file_name=f"{unique_id}.mp4",
                seconds_per_screenshot=data.get("seconds_per_screenshot", 1)
            )
        except Exception as e:
            return jsonify({'error': f"Error in downloading or saving the video: {str(e)}"}), 500
            
        # Return the video file path and screenshots
        if video_path:
            return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
        else:
            return jsonify({'error': 'Failed to download or save the video.'}), 500
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500
    
@app.route('/generate_image2video', methods=['POST'])
def generate_image2video_replicate():
    global last_execution_time

    with execution_lock:
        current_time = time.time()
        time_since_last_call = current_time - last_execution_time
        
        if time_since_last_call < 2:
            time.sleep(2 - time_since_last_call)  # Wait for the remaining time

        last_execution_time = time.time()  # Update execution time after waiting
    
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data or 'image' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" and "image" keys.'}), 400
        
        # Step 1: Call API to generate video
        try:
            rsp = VideoSynthesis.call(model='wanx2.1-i2v-turbo',
                              prompt=data['prompt'], img_url=data['image'],size='1280*720')
            if rsp.status_code == HTTPStatus.OK:
                output = rsp.output.video_url
                if output and isinstance(output, str):
                    print(output)
                else:
                    return jsonify({'error': f'Failed to get valid video URL from API response, response {rsp}'}), 502
            else:
                print(f"Wanx API returned non 200 status code: {rsp.status_code} - {rsp.message}")
                return jsonify({'error': f'Wanx API error: {rsp.message}'}), rsp.status_code
        except Exception as e:
            print(f"Exception in calling Aliyun VideoSynthesis API: {e}")
            return jsonify({'error': f"Error in generating video with Aliyun API: {str(e)}"}), 501
        
        video_url = output
        unique_id = str(uuid.uuid4())
        
       # Step 2: Download video and save locally
        try:
            video_path, screenshots = download_video_and_save_as_mp4(
                video_url,
                file_name=f"{unique_id}.mp4",
                seconds_per_screenshot=data.get("seconds_per_screenshot", 1)
            )
        except Exception as e:
            return jsonify({'error': f"Error in downloading or saving the video: {str(e)}"}), 500
            
        # Return the video file path and screenshots
        if video_path:
            return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
        else:
            return jsonify({'error': 'Failed to download or save the video.'}), 500
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

@app.route('/generate_image', methods=['POST'])
def image_generation():
    global last_execution_time

    with execution_lock:
        current_time = time.time()
        time_since_last_call = current_time - last_execution_time
        
        if time_since_last_call < 2:
            time.sleep(2 - time_since_last_call)  # Wait for the remaining time

        last_execution_time = time.time()  # Update execution time after waiting
    
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data:
            print(data)
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" key.'}), 400

        # Step 1: Call API to generate image
        try:
            # Wan2.1
            rsp = ImageSynthesis.call(api_key=os.getenv("DASHSCOPE_API_KEY"), model='wanx2.1-t2i-turbo',
                              prompt=data['prompt'], n=1,
                          size='1024*1024')
            if rsp.status_code == HTTPStatus.OK:
                if not rsp.output.results or len(rsp.output.results) == 0:
                    return jsonify({'error': 'API returned empty results'}), 500
                output = rsp.output.results[0]
                print(output)
            else:
                print(f"Aliyun API error: {rsp.status_code} - {rsp.message}")
                return jsonify({'error': f'Aliyun API error: {rsp.message}'}), rsp.status_code
        except Exception as e:
            return jsonify({'error': f"Error in generating image with Aliyun API: {str(e)}"}), 500
            
        # Check if the output contains the image URL
        if not output or "url" not in output:
            return jsonify({'error': 'Aliyun API did not return a valid image URL.'}), 500
            
        # Download image and convert to base64
        try:
            base64_image = IMGGEN_download_image_and_convert_to_base64(output["url"])
            if base64_image:
                return jsonify({'generated_image_base64': base64_image}), 200
            else:
                return jsonify({'error': 'Failed to download or encode the image.'}), 500
        except Exception as e:
            return jsonify({'error': f"Error processing the image: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7905, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(port=args.port, host='0.0.0.0')
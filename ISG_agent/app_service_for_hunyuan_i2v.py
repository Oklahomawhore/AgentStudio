from util import download_video_and_save_as_mp4
from hunyuan_client import HunyuanClient
from flask import Flask, request, jsonify
import replicate
import dotenv
import uuid
import time
import threading

dotenv.load_dotenv()

app = Flask(__name__)
hunyuan_client = HunyuanClient()  # Initialize the Hunyuan client

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
            time.sleep(2 - time_since_last_call)

        last_execution_time = time.time()

    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" key.'}), 400

        try:
            # Using Hunyuan client instead of Replicate
            video_path, screenshots = hunyuan_client.text2video(
                prompt_list=data["prompt"],
                seconds_per_screenshot=data.get("seconds_per_screenshot", 1)
            )
        except Exception as e:
            return jsonify({'error': f"Error in generating video with Hunyuan API: {str(e)}"}), 500

        if video_path:
            return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
        else:
            return jsonify({'error': 'Failed to generate or save the video.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_image2video', methods=['POST'])
def generate_image2video_replicate():
    global last_execution_time

    with execution_lock:
        current_time = time.time()
        time_since_last_call = current_time - last_execution_time
        
        if time_since_last_call < 2:
            time.sleep(2 - time_since_last_call)

        last_execution_time = time.time()

    try:
        data = request.get_json()

        if not data or 'prompt' not in data or 'image' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" and "image" keys.'}), 400

        try:
            # Using Hunyuan client instead of Replicate
            video_path, screenshots = hunyuan_client.image2video(
                image_url=data["image"],
                prompt=data["prompt"],
                seconds_per_screenshot=data.get("seconds_per_screenshot", 1)
            )
        except Exception as e:
            return jsonify({'error': f"Error in generating video with Hunyuan API: {str(e)}"}), 500

        if video_path:
            return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
        else:
            return jsonify({'error': 'Failed to generate or save the video.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7906, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(port=args.port, host='0.0.0.0') 
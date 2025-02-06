from util import download_video_and_save_as_mp4
import replicate
from flask import Flask, request, jsonify
import replicate
import dotenv
import uuid
import time
import threading

dotenv.load_dotenv()

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


        # Acquire the lock to prevent concurrent requests
        
        # Step 1: Call Replicate API to generate video
        try:
            # output = replicate.run(
                # "tencent/hunyuan-video:847dfa8b01e739637fc76f480ede0c1d76408e1d694b830b5dfb8e547bf98405",
                # input={
                    # "width": data.get("width", 854),
                    # "height": data.get("height", 480),
                    # "prompt": data["prompt"],
                    # "flow_shift": data.get("flow_shift", 7),
                    # "infer_steps": data.get("infer_steps", 50),
                    # "video_length": data.get("video_length", 129),
                    # "embedded_guidance_scale": data.get("embedded_guidance_scale", 6)
                # }
            # )
            # kling-v1.6
            output = replicate.run(
                "kwaivgi/kling-v1.6-standard",
                input={
                    "prompt": data["prompt"],
                    "duration": 5,
                    "cfg_scale": 0.5,
                    "aspect_ratio": "16:9",
                    "negative_prompt": ""
                }
            )
        except Exception as e:
            return jsonify({'error': f"Error in generating video with Replicate API: {str(e)}"}), 500
        # Check if the output contains the video URL
        if not output or not isinstance(output, str):
            return jsonify({'error': f'Replicate API did not return a valid video URL {output}'}), 500
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
        return jsonify({'error': str(e)}), 500
    

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

        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" key.'}), 400

        
        
        # Step 1: Call Replicate API to generate video
        try:
            # output = replicate.run(
            #     "tencent/hunyuan-video:847dfa8b01e739637fc76f480ede0c1d76408e1d694b830b5dfb8e547bf98405",
            #     input={
            #         "width": data.get("width", 854),
            #         "height": data.get("height", 480),
            #         "prompt": data["prompt"],
            #         "flow_shift": data.get("flow_shift", 7),
            #         "infer_steps": data.get("infer_steps", 50),
            #         "video_length": data.get("video_length", 129),
            #         "embedded_guidance_scale": data.get("embedded_guidance_scale", 6)
            #     }
            # )
            # kling-v1.6
            output = replicate.run(
                "kwaivgi/kling-v1.6-standard",
                input={
                    "prompt": data["prompt"],
                    "duration": 5,
                    "cfg_scale": 0.5,
                    "aspect_ratio": "16:9",
                    "negative_prompt": ""
                }
            )
        except Exception as e:
            return jsonify({'error': f"Error in generating video with Replicate API: {str(e)}"}), 500
        # Check if the output contains the video URL
        if not output or not isinstance(output, str):
            return jsonify({'error': 'Replicate API did not return a valid video URL.'}), 500
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
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7905, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(port=args.port, host='0.0.0.0')
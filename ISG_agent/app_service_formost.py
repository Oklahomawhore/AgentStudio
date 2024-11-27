# unified_app.py

import os
from flask import Flask, request, jsonify
from threading import Lock
import logging

# Import the model classes
from Models.Dynami_video_generator import VideoGenerator
from Models.video_3d_generator import Video3DGenerator
# from image_generator import ImageGenerator for ablation study
from Models.image_editorimage_editor import ImageEditor
# from image_editor_weak import ImageEditor for ablation study

# Initialize the Flask app
app = Flask(__name__)

# Initialize models with thread-safe locks
video_generator_lock = Lock()
video_generator = VideoGenerator()

video_3d_generator_lock = Lock()
video_3d_generator = Video3DGenerator(device="cuda")


image_generator_lock = Lock()
image_generator = ImageGenerator()

image_editor_lock = Lock()
image_editor = ImageEditor()

# --------------------------------------------
# Video Generation Endpoint
# --------------------------------------------
@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        data = request.get_json()
        if not data or 'prompt_list' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with a "prompt_list" key.'}), 400

        prompt_list = data['prompt_list']
        seconds_per_screenshot = data['seconds_per_screenshot']

        with video_generator_lock:
            base64_screenshots = video_generator.generate_video(prompt_list, seconds_per_screenshot)

        return jsonify({'screenshots': base64_screenshots}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------
# 3D Generation Endpoint
# --------------------------------------------
@app.route('/generate_3d_video', methods=['POST'])
def generate_3d_video():
    try:
        data = request.get_json()
        if not data or 'input_list' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with an "input_list" key.'}), 400

        input_list = data['input_list']
        screenshots_per_second = data.get('screenshots_per_second', 1)
        proportions = data.get('proportions', [1/12, 2/12, 10/12, 11/12])
        # print(input_list), print(screenshots_per_second)
        with video_3d_generator_lock:
            base64_screenshots = video_3d_generator.generate_video(input_list, screenshots_per_second, proportions)

        return jsonify({'screenshots': base64_screenshots}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------
# Image Generation Endpoint
# --------------------------------------------
@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with a "prompt" key.'}), 400

        prompt = data['prompt']

        with image_generator_lock:
            base64_image = image_generator.generate_image(prompt)

        return jsonify({'generated_image_base64': base64_image}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------------------------
# Image Editing Endpoint
# --------------------------------------------
@app.route('/edit_image', methods=['POST'])
def edit_image():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data or 'image_input' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "prompt" and "image_input" keys.'}), 400

        prompt = data['prompt']
        image_input = data['image_input']

        with image_editor_lock:
            edited_image_base64 = image_editor.edit_image(prompt, image_input)

        return jsonify({'edited_image_base64': edited_image_base64}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5000
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7899, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)

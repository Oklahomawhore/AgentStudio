import os
from flask import Flask, request, jsonify
from threading import Lock
import logging

# Import the ImageMorph class
from Models.image_morph import ImageMorph  # Ensure this matches your file structure

# Initialize the Flask app
app = Flask(__name__)

# Set your model paths here
MODEL_PATH = "Models/stable-diffusion-v1-5"
VAE_PATH = "Models//sd-vae-ft-mse"
LORA_PATH = "Models/DreamMover/lora_tmp/pytorch_lora_weights.bin" 

# Initialize ImageMorph with a thread-safe lock
morph_lock = Lock()
image_morph = ImageMorph(MODEL_PATH, VAE_PATH, LORA_PATH)

# --------------------------------------------
# Image Morphinqg Endpoint
# --------------------------------------------
@app.route('/morph', methods=['POST'])
def morph_images():
    try:
        data = request.get_json()
        if not data or 'img_path1' not in data or 'img_path2' not in data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "img_path1", "img_path2", and "prompt" keys.'}), 400

        img_path1 = data['img_path1']
        img_path2 = data['img_path2']
        prompt = data['prompt']

        if not os.path.isfile(img_path1) or not os.path.isfile(img_path2):
            return jsonify({'error': 'One or both image paths do not exist.'}), 404
        with morph_lock:
            frames = image_morph.generate_frames(img_path1, img_path2, prompt)

        return jsonify({'frames': frames}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7901, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)

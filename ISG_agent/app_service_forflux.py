import os
from flask import Flask, request, jsonify
from threading import Lock
import logging

from Models.image_generator import ImageGenerator

app = Flask(__name__)


image_generation_lock = Lock()
image_generation = ImageGenerator()

# --------------------------------------------
# Image Generation Endpoint
# --------------------------------------------
@app.route('/generate_image', methods=['POST'])
def generate_image_agent():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with a "prompt" key.'}), 400

        prompt = data['prompt']

        with image_generation_lock:
            base64_image = image_generation.generate_image(prompt)

        return jsonify({'generated_image_base64': base64_image}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
if __name__ == '__main__':
    # Run the app on port 5000
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7900, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)

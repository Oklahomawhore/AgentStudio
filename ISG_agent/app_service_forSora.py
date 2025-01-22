from util import download_video_and_save_as_mp4
import replicate
from flask import Flask, request, jsonify
import replicate
import dotenv
from threading import Lock
import uuid
from openai import OpenAI
import os
import base64
from util import download_video_and_save_as_mp4
import re




dotenv.load_dotenv()
OpenAIClient = OpenAI(
   api_key=os.getenv("KLING_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)


app = Flask(__name__)

text2video_generator_lock = Lock()
image2video_generator_lock = Lock()

@app.route('/generate_video', methods=['POST'])
def generate_video_replicate():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "messages" key.'}), 400

        messages = []
        
        messages.append({"role": "user" , "content": data['prompt']})

        # Acquire the lock to prevent concurrent requests
        with text2video_generator_lock:
            try:
                
                # Call OpenAI Chat Completions API with streaming
                # completion = OpenAIClient.chat.completions.create(
                #     model="sora-9:16-480p-5s",
                #     messages=messages
                # )
                # # Write streamed response to file
            
                # response = completion.choices[0].message.content
                response = """
                ```json
{
  "user_prompt": "A 28-year-old Asian software developer named John, with short black hair and glasses, is seated at his desk in a cozy, dimly lit home office. He is wearing a casual t-shirt and jeans, intensely focused on his computer screen. The camera captures close-up shots of his hands flying over the keyboard, interspersed with wide-angle shots that show the entire room filled with tech gadgets and programming posters. Occasionally, John pauses to adjust his glasses, showing a contemplative expression, before diving back into his work, highlighting the urgency and intensity of his crucial project.",
  "image_url": ""
}
```



> ID: `task_01jj4nyysdfscbqrv7rkj3f2pf`
> 排队中..
> 生成中...
> 生成完成 ✅


> 视频信息 480x854

![https://filesystem.site/cdn/20250121/F8K35Gm76LJ4o3PZ4vuvgJKWGlc0gP.webp](https://filesystem.site/cdn/20250121/F8K35Gm76LJ4o3PZ4vuvgJKWGlc0gP.webp)
[在线播放▶️](https://videos.openai.com/vg-assets/assets%2Ftask_01jj4nyysdfscbqrv7rkj3f2pf%2Ftask_01jj4nyysdfscbqrv7rkj3f2pf_genid_e1e88891-1e08-4940-b442-4f8e904b0b6d_25_01_21_14_59_038036%2Fvideos%2F00000_81274901_watermarked.mp4?st=2025-01-21T13%3A52%3A45Z&se=2025-01-27T14%3A52%3A45Z&sks=b&skt=2025-01-21T13%3A52%3A45Z&ske=2025-01-27T14%3A52%3A45Z&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skoid=aa5ddad1-c91a-4f0a-9aca-e20682cc8969&skv=2019-02-02&sv=2018-11-09&sr=b&sp=r&spr=https%2Chttp&sig=atLLhk6hMHoy5EfzmVuFazvr9wU9Ay7%2BQRwXjVW3CXQ%3D&az=oaivgprodscus)
                """
                url = response.split("在线播放")[-1].split("(")[-1].split(")")[0]
                video_path, screenshots = download_video_and_save_as_mp4(url)
                return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
                
            except Exception as e:
                return jsonify({'error': f"Error during OpenAI API call: {str(e)}"}), 500

        return jsonify({'response_file': file_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/generate_image2video', methods=['POST'])
def generate_image2video_replicate():
    try:
        # Extract input data from the request
        data = request.get_json()

        if not data or 'prompt' not in data or 'image' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "messages" key.'}), 400

        messages = []
        
        messages.append({"role": "user" , "content": data['prompt']})
        
        file_name = f"sora-{str(uuid.uuid4())}.mp4"
        file_path = os.path.join('videos', file_name)

        # Ensure the "responses" directory exists
        os.makedirs('videos', exist_ok=True)

        # Acquire the lock to prevent concurrent requests
        with image2video_generator_lock:
            try:
                
                
                # Call OpenAI Chat Completions API with streaming
                # completion = OpenAIClient.chat.completions.create(
                #     model="sora-9:16-480p-5s",
                #     messages=messages
                # )
                
                
                # response = completion.choices[0].message.content
                response = """
                ```json
{
  "user_prompt": "A 28-year-old Asian software developer named John, with short black hair and glasses, is seated at his desk in a cozy, dimly lit home office. He is wearing a casual t-shirt and jeans, intensely focused on his computer screen. The camera captures close-up shots of his hands flying over the keyboard, interspersed with wide-angle shots that show the entire room filled with tech gadgets and programming posters. Occasionally, John pauses to adjust his glasses, showing a contemplative expression, before diving back into his work, highlighting the urgency and intensity of his crucial project.",
  "image_url": ""
}
```



> ID: `task_01jj4nyysdfscbqrv7rkj3f2pf`
> 排队中..
> 生成中...
> 生成完成 ✅


> 视频信息 480x854

![https://filesystem.site/cdn/20250121/F8K35Gm76LJ4o3PZ4vuvgJKWGlc0gP.webp](https://filesystem.site/cdn/20250121/F8K35Gm76LJ4o3PZ4vuvgJKWGlc0gP.webp)
[在线播放▶️](https://videos.openai.com/vg-assets/assets%2Ftask_01jj4nyysdfscbqrv7rkj3f2pf%2Ftask_01jj4nyysdfscbqrv7rkj3f2pf_genid_e1e88891-1e08-4940-b442-4f8e904b0b6d_25_01_21_14_59_038036%2Fvideos%2F00000_81274901_watermarked.mp4?st=2025-01-21T13%3A52%3A45Z&se=2025-01-27T14%3A52%3A45Z&sks=b&skt=2025-01-21T13%3A52%3A45Z&ske=2025-01-27T14%3A52%3A45Z&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skoid=aa5ddad1-c91a-4f0a-9aca-e20682cc8969&skv=2019-02-02&sv=2018-11-09&sr=b&sp=r&spr=https%2Chttp&sig=atLLhk6hMHoy5EfzmVuFazvr9wU9Ay7%2BQRwXjVW3CXQ%3D&az=oaivgprodscus)
                """
                url = response.split("在线播放")[-1].split("(")[-1].split(")")[0]
                video_path, screenshots = download_video_and_save_as_mp4(url)
                return jsonify({'video_file': video_path, 'screenshots': screenshots}), 200
            except Exception as e:
                return jsonify({'error': f"Error during OpenAI API call: {str(e)}"}), 500

        return jsonify({'response_file': file_path}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=7909, type=int, help='port to listen on')
    args = parser.parse_args()
    app.run(port=args.port, host='0.0.0.0')
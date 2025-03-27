import os
from openai import OpenAI
import dotenv
from util import capture_screenshots
import base64
import dashscope
from http import HTTPStatus

dotenv.load_dotenv()

from moviepy import VideoFileClip

def get_video_length(video_path):
    clip = VideoFileClip(video_path)
    duration = clip.duration  # Duration in seconds
    clip.close()
    return duration


def score_video(video_path, prompt=None):
    # Get length of video
    video_length = get_video_length(video_path)
    print("Video length (s) :", video_length)
    
    # 决定截图间隔
    # 1. 对于极短视频(<1秒)，固定截取4张图
    # 2. 对于短视频(1-5秒)，每0.2-0.5秒一张，确保至少5张
    # 3. 对于中等视频(5-30秒)，截取约20-40张
    # 4. 对于长视频(>30秒)，每秒不超过1.5张，避免生成过多图片
    
    # 首先计算理想的截图数量，确保在4-512范围内
    if video_length < 1.0:
        # 极短视频：固定截取4张
        ideal_screenshots = 4
    elif video_length < 5.0:
        # 短视频：至少5张，但不超过25张
        ideal_screenshots = min(25, max(5, int(video_length / 0.2)))
    elif video_length < 30.0:
        # 中等视频：20-40张之间
        ideal_screenshots = min(40, max(20, int(video_length * 1.2)))
    else:
        # 长视频：每秒1-1.5张，但总数不超过512
        ideal_screenshots = min(512, max(30, int(video_length * 1.0)))
    
    # 根据理想截图数量计算间隔
    seconds_per_screenshot = video_length / ideal_screenshots
    
    # 确保不会因为浮点数精度问题而跳过截图
    seconds_per_screenshot = min(seconds_per_screenshot, video_length)
    
    # 最后验证截图数量是否在允许范围内
    expected_screenshots = int(video_length / seconds_per_screenshot) + 1
    
    # 如果预期截图数量超出范围，调整间隔
    if expected_screenshots < 4:
        seconds_per_screenshot = video_length / 4
        print("调整：截图数量太少，已增加到4张")
    elif expected_screenshots > 512:
        seconds_per_screenshot = video_length / 512
        print("调整：截图数量太多，已限制在512张以内")
    
    print(f"截图间隔：每{seconds_per_screenshot:.3f}秒一张，预计{int(video_length / seconds_per_screenshot) + 1}张截图")
    
    screenshots = capture_screenshots(video_path, seconds_per_screenshot=seconds_per_screenshot)
    print(f"共截取了{len(screenshots)}张图片")
    
    # 如果实际截图数量不在范围内，发出警告
    if not (4 <= len(screenshots) <= 512):
        print(f"警告：实际截图数量为{len(screenshots)}，不在预期范围(4-512)内")
        return "Video too short."
    
    paths = []
    for i, screenshot in enumerate(screenshots):

        output_path = os.path.join('/data/wangshu/wangshu_code/ISG/ISV_eval/tmp', f"screenshot_{i}.jpg")
        # Decode base64 string to bytes
        img_bytes = base64.b64decode(screenshot)
        # Write bytes to file
        with open(output_path, 'wb') as f:
            f.write(img_bytes)
        paths.append(output_path)
    # client = OpenAI(
    #     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    #     api_key=os.getenv("DASHSCOPE_API_KEY"),
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # )
    # completion = client.chat.completions.create(
    #     model="qwen2.5-vl-72b-instruct",
    #     messages=[{"role": "user","content": [
    #         {"type": "video","video": [f"file://{path}" for path in paths]},
    #         {"type": "text","text": prompt if prompt else "Judge the quality of this video, first state your reasons and then give a score from 1-10"},
    #     ]}]
    # )
    # print(completion.choices[0].message.content)

    messages = [{
        "role": "user",
                "content": [
                    {"video":paths},
                    {"text": prompt if prompt else "Judge the quality of this video, first state your reasons and then give a score from 1-10"}
                 ]
                 }]
    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model='qwen2.5-vl-72b-instruct',
        messages=messages
    )
    if response.status_code != HTTPStatus.OK:
        print(response.message)
    try:
        print(response["output"]["choices"][0]["message"].content[0]["text"])
    except:
        print(response)
    return response["output"]["choices"][0]["message"].content[0]["text"]
    # return completion.choices[0].message.content



if __name__ == '__main__':
    score_video("/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_newplanning/Task_0004/final_video_b14ba07e-f317-4d65-b517-bf0085ff82dd.mp4", prompt="Watch this video, and deduct the where, what and why of this video. After the coprehension, try to put the pieces together and tell the whole story if you can, if you cannot, just say you can't understand it.")
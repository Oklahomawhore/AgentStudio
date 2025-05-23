import moviepy.editor as mp
import pysrt
import os
import glob
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

result_dir = "/data/wangshu/wangshu_code/ISG/ISG_agent/results_video_novel"

result_json = glob.glob(f"{result_dir}/**/result.json")


ClaudeClient = OpenAI(
   api_key=os.getenv("CLAUDE_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

# Step 1: Concatenate videos and save the final file
def concatenate_videos(output_videos, final_video_path="final_video.mp4"):
    video_clips = [mp.VideoFileClip(video["content"]) for video in output_videos if video["type"] == "video"]
    final_clip = mp.concatenate_videoclips(video_clips, method="compose")
    final_clip.write_videofile(final_video_path, codec="libx264")
    print(f"Final video saved as {final_video_path}")

# Step 2: Translate text into a single SRT
def create_srt_file(text_content, srt_file_path="final_subtitles.srt"):
    subtitles = pysrt.SubRipFile()
    lines = text_content.strip().split("\n")
    idx = 1
    shift_time = 0.0  # Start time shift in seconds

    for i in range(0, len(lines), 4):  # SRT blocks are every 4 lines
        if i + 2 < len(lines):
            index = idx
            times = lines[i + 1].split(" --> ")
            print(times)
            start = pysrt.SubRipTime.from_string(times[0])
            end = pysrt.SubRipTime.from_string(times[1])
            start.shift(seconds=shift_time)
            end.shift(seconds=shift_time)
            text = lines[i + 2]
            subtitle = pysrt.SubRipItem(index=index, start=start, end=end, text=text)
            subtitles.append(subtitle)
            idx += 1

    subtitles.save(srt_file_path, encoding="utf-8")
    print(f"Final subtitles saved as {srt_file_path}")

import ffmpeg

def add_subtitles_to_video(input_video_path, input_srt_path, output_video_path):
    """
    Adds subtitles from an SRT file to an MP4 video.

    :param input_video_path: str, path to the input MP4 video file
    :param input_srt_path: str, path to the input SRT subtitle file
    :param output_video_path: str, path to save the output video with subtitles
    """
    try:
        # Run ffmpeg command to add subtitles
        ffmpeg.input(input_video_path).output(output_video_path, vf=f"subtitles={input_srt_path}").run()
        print(f"Subtitles added successfully! Output saved at: {output_video_path}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode()}")


for result in result_json:
    with open(result, "r") as f:
        result_json = json.load(f)
        # Process the JSON
        video_outputs = [item for item in result_json["output"] if item["type"] == "video"]
        text_content = next(item["content"] for item in result_json["output"] if item["type"] == "text")

        # Run video concatenation
        concatenate_videos(video_outputs, f"{result_dir}/Task_{result_json['id']}/final_video_{result_json['id']}.mp4")

        content = []


        content.append({"type": "text", "text": text_content})
        messages = []
        messages.append({
            "role":"system",
            "content":"In this task, generate a subtitle of the generated video contents each 5 seconds long in the srt format based on the story, it is `text-only` response with or without image input. You should simulate as if the image is generated by yourself to maintain the continuity. You should **Never** apologizes or response any form of negative acknowledgment like \"I apologize\", \"The image doesn't show\" of the image regarding the presence or absence of certain elements in the image. You should use \"This image\" instead of \"The image\" Never compare the differences and discrepancies between the text instruction and the image, only focus on the similar part. The response should be harmonious with both the image and the instruction, ensuring that any contradictions or irrelevant details are ignored. If you cannot extract any aligning information, you should focus more on the instruction. Do not output too many unimportant things if the instruction didn't ask even if the instruction wants you to output in detail.",
        })
        messages.append({
            "role": "user",
            "content": content,
        })
        try:
            completion = ClaudeClient.chat.completions.create(
                model="claude-3-5-sonnet-20240620",  # Replace with your model name
                max_tokens=8192,
                messages=messages,
            )
            response = completion.choices[0]
        except Exception as e:
            raise ValueError(f"Error calling the API: {e}")
        with open(f"{result_dir}/Task_{result_json['id']}/final_video_{result_json['id']}.srt", "w") as f:
            f.write(response.message.content)
        
        add_subtitles_to_video(f"{result_dir}/Task_{result_json['id']}/final_video_{result_json['id']}.mp4", f"{result_dir}/Task_{result_json['id']}/final_video_{result_json['id']}.srt", f"{result_dir}/Task_{result_json['id']}/final_video_{result_json['id']}_subtitled.mp4")

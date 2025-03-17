import os
import torch
from moviepy import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from api_interface import kling_imggen_agent, kling_img2video_agent, kling_text2video_agent, morph_images_agent
import ChatTTS
import torchaudio
import time
import hashlib
import pickle
import replicate
import dotenv
from threading import Lock
import uuid
import requests
import json
from pydub import AudioSegment  # For concatenating WAV files
import re
import base64

dotenv.load_dotenv()

tts_lock = Lock()
# In this file, we decide which genration backend to call and
# hide the details of generation logic from the main code

# Helper function to generate a unique hash based on the prompt
def generate_hash(prompt):
    print(f"hashing {prompt}")
    # Generate a SHA-256 hash of the prompt string
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

# Save results to disk using pickle
def save_to_disk(content, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(content, f)

# Check if the result exists, if so return the content
def load_from_disk(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

# Generate character images based on prompts
def gen_img(prompt):
    try:
        prompt_hash = generate_hash(prompt)
        file_path = f"imgs/{prompt_hash}.pkl"

        # check if there exists a cached image for the prompt
        existing_image = load_from_disk(file_path)
        if existing_image is not None:
            print(f"<GEN_IMAGE> prompt: {prompt} already exists")
            return existing_image
        img_base64 = kling_imggen_agent(prompt.replace("-",","))
        save_to_disk(img_base64, file_path)
    except Exception as e:
        print(f"Error in generating charactor: {str(e)}") 
    # save base64 img to file and return path
    return img_base64

def gen_video(prompt, image, i2v=False):
    try:
        prompt_hash = generate_hash(prompt + image)  # Combine image path for uniqueness
        file_path = f"videos/{prompt_hash}.pkl"
        
        # Check if the video already exists
        existing_video = load_from_disk(file_path)

        if existing_video is not None:
            print(f"<GEN_VIDEO> prompt: {prompt} image: {image} already exists")
            return existing_video
        if i2v:
            video_list, screenshot_list = kling_img2video_agent(image, prompt)
        else:
            video_list, screenshot_list = kling_text2video_agent(prompt)
        # time.sleep(10)
        # with open("imgs/dog1.txt", "r") as f:
        #     img_b64 = f.read().strip()
        # video_list, screenshot_list = "/data/wangshu/wangshu_code/ISG/ISG_agent/videos/118f0090-14c2-40a7-b6b5-4a1c62ae0531.mp4", [img_b64, img_b64]
        save_to_disk((video_list, screenshot_list), file_path)

    except Exception as e:
        print(f"Error in generating video: {str(e)}")
    
    return video_list, screenshot_list

# Music Generation using hash as input (returns file path)
def gen_music(prompt, duration):
    print(f"Generating music for prompt: {prompt} for duration {duration} seconds")
    try:
        # Generate hash from the prompt
        prompt_hash = generate_hash(prompt)
        file_path = f"music/{prompt_hash}.wav"
        
        # Check if the music already exists
        # time.sleep(10)
        # return "/data/wangshu/wangshu_code/ISG/ISG_agent/music/replicate-prediction-5r4erq6jx5rgm0cgg14v40n4x4.wav"
        if os.path.exists(file_path):
            print(f"<GEN_MUSIC> prompt: {prompt} already exists")
            return file_path
        
        # music generation code
        input = {
            "prompt": prompt,
            "duration": duration
        }

        output = replicate.run(
            "ardianfe/music-gen-fn-200e:96af46316252ddea4c6614e31861876183b59dce84bad765f38424e87919dd85",
            input=input
        )
        # Send a GET request to the URL to fetch the audio file
        response = requests.get(output)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Write the content of the response to a .wav file
            with open(f"music/{prompt_hash}.wav", "wb") as file:
                file.write(response.content)
        else:
            print(f"Failed to download the file: {response.status_code}")
        #=> output.wav written to disk

        return f"music/{prompt_hash}.wav"
    except Exception as e:
        print(f"Error in generating music: {str(e)}")
        return None
import re

import re

def extract_conversation(prompt, speaker):
    """
    Extracts the conversation for a given speaker from the prompt.
    Supports both <#speaker#> and <speaker> patterns, and removes text enclosed in parentheses.

    Args:
        prompt (str): The input prompt containing the conversation.
        speaker (str): The name of the speaker.

    Returns:
        str: The cleaned conversation for the speaker.
    """
    print(prompt)
    if prompt != '' and speaker not in prompt:
        # Split on the first colon when speaker is not explicitly in the prompt
        try:
            conversation = prompt.split("：", 1)[1]
        except:
            conversation = prompt.split(':', 1)[1]
    else:
        # Regex pattern to match <#speaker#>: or <speaker>:
        pattern = rf"<#?{re.escape(speaker)}#?>(.*?)(<#?.*?#?>|$)"
        match = re.search(pattern, prompt)

        if match:
            # Extract the conversation following the speaker's tag
            conversation = match.group(1).strip()
        else:
            # Fallback: No match found
            conversation = ""

    # Remove anything enclosed in () or （）
    conversation = re.sub(r"[\(（][^)）]*[\)）]", "", conversation).strip()
    conversation = conversation.replace('"', '').replace('“','')


    return conversation

def gen_tts(prompt, voice_direction, speaker_embeddings_cache_dir="speaker_embeddings"):
    if prompt == "":
        return None
    assert isinstance(prompt, str), "Prompt must be a string"
    assert isinstance(voice_direction, dict), "Voice direction must be a dictionary"

    combined_file = f"audio/{prompt.replace('/', '').replace('：', '').replace(' ', '')}.wav"

    if os.path.exists(combined_file):
        print(f"Combined audio for prompt {prompt} already exists")
        return combined_file

    with tts_lock:
        # Extract all speaker-conversation pairs

        chat = ChatTTS.Chat()
        chat.load(compile=False)  # Set to True for better performance

        os.makedirs(speaker_embeddings_cache_dir, exist_ok=True)
        os.makedirs("audio", exist_ok=True)

        saved_wavs = []

        for speaker, description in voice_direction.items():
            
            conversation = extract_conversation(prompt, speaker)

            print(f"[TTS] {speaker} : {conversation}")
            # Generate hash for the speaker embedding
            speaker_hash = generate_hash(speaker)
            speaker_file_path = os.path.join(speaker_embeddings_cache_dir, f"{speaker_hash}.pkl")

            # Load or sample speaker embedding
            if os.path.exists(speaker_file_path):
                with open(speaker_file_path, 'rb') as f:
                    rand_spk = pickle.load(f)
                print(f"Recovered speaker for {speaker}")
            else:
                rand_spk = chat.sample_random_speaker()
                print(f"Sampled new speaker for {speaker}")
                with open(speaker_file_path, 'wb') as f:
                    pickle.dump(rand_spk, f)

            # Set parameters for TTS inference
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=rand_spk,  # Use the selected speaker embedding
                temperature=0.3,    # Custom temperature
                top_P=0.7,          # Top P decode
                top_K=20,           # Top K decode
            )

            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt='[oral_2][laugh_0][break_6]',  # Control speech attributes here
            )

            # Generate the waveform
            wavs = chat.infer(
                conversation,  # Use the extracted text for the speaker
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )

            # Save individual WAV file
            file_name = f"audio/{speaker}_{conversation[:10].replace('/', '').replace(' ', '')}.wav"
            try:
                torchaudio.save(file_name, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
            except Exception as e:
                torchaudio.save(file_name, torch.from_numpy(wavs[0]), 24000)

            saved_wavs.append(file_name)

        # Concatenate all WAV files into one
        
        combined_audio = None
        pause = AudioSegment.silent(duration=1000)  # 1-second silent audio segment
        for wav_file in saved_wavs:
            audio_segment = AudioSegment.from_wav(wav_file)
            combined_audio = (
                audio_segment if combined_audio is None else combined_audio + pause + audio_segment
            )

        
        combined_audio.export(combined_file, format="wav")
        print(f"Combined audio saved to {combined_file}")

        return combined_file

def concat_video(video_list, music_path, tts_paths, task_dir):
    """
    Concatenate multiple MP4 videos and add dialogue (TTS) and background music to the audio track.
    
    Args:
        video_list: List of video file paths (MP4s) to be concatenated.
        music_path: File path to the background music (WAV/MP3).
        tts_path: File path to the dialogue (WAV/MP3).
    
    Returns:
        final_video_path: File path to the concatenated video with audio.
    """
    
    try:
        
        # Load video clips
        video_clips = [VideoFileClip(video[0]) for video in video_list]

        # Concatenate video clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Load audio clips
        music_audio = AudioFileClip(music_path).with_volume_scaled(0.5)
        music_audio = music_audio.subclipped(0, 5 * len(video_list))
        
        
        tts_clip = [AudioFileClip(tts_path).with_start(5.0 * index) for index, tts_path in enumerate(tts_paths) if tts_path is not None]

        # Combine audio tracks (dialogue + music)
        final_audio = CompositeAudioClip([music_audio, *tts_clip])
        
        final_video = final_video.with_audio(final_audio)

        # Save the final video
        output_path = f"{task_dir}/final_video_{uuid.uuid4()}.mp4"
        final_video.write_videofile(output_path)
        return output_path

    except Exception as e:
        print(f"Error during video concatenation: {e}")
        return None

def generate_all(video_task, music_task, tts_task, task_dir):
    # Create separate executors for each task category
    with ThreadPoolExecutor() as executor:
        start = time.time()
        
        # First generate videos that need screenshots
        t2v_tasks = [task for task in video_task if task[1] == ""]
        vid_results_stage_1 = list(executor.map(gen_video, *zip(*t2v_tasks)))
        
        # Process remaining tasks sequentially if they need last frame
        vid_results = []
        last_screenshot = None
        
        for i, task in enumerate(video_task):
            if task[1] == "":
                # Tasks already processed in stage 1
                result = vid_results_stage_1.pop(0)
                # Save screenshot to disk
                screenshot_path = f"{task_dir}/screenshot_{i}.png"
                image_bytes = base64.b64decode(result[1][-1])
                with open(screenshot_path, "wb") as f:
                    f.write(image_bytes)
                last_screenshot = screenshot_path
                vid_results.append(result)
            elif task[1] == "<LastFrame>":
                # Execute i2v tasks one by one to use last frame
                if last_screenshot:
                    result = gen_video(task[0], last_screenshot, i2v=True)
                    # Save new screenshot
                    screenshot_path = f"{task_dir}/screenshot_{i}.png"
                    image_bytes = base64.b64decode(result[1][-1])
                    with open(screenshot_path, "wb") as f:
                        f.write(image_bytes)
                    last_screenshot = screenshot_path
                    vid_results.append(result)
                else:
                    raise Exception("Need last frame for an i2v task.")
        # Generate TTS and music in parallel
        tts_results = executor.map(gen_tts, *zip(*tts_task))
        music_result = executor.submit(gen_music, " ".join([task if task is not None else "" for task in music_task]), len(video_task) * 5)

        final = concat_video(vid_results, music_result.result(), list(tts_results), task_dir)
        end = time.time()
        print(f"Time taken: {end-start:.6f}")

    return final


if __name__ == "__main__":
    # generate_all([("丛林老者正在山野里打猎，他的脸上写满了沧桑", "image1"), ("老者正在打猎，突然间他的背后出现一只老虎的身影，老者大惊失色" ,"image2")], ["竹林小调，笛子旋律，轻快的", "管弦乐队，突然爆发的旋律，渲染紧张气氛"], [("我自山上来，今天来打猎","v1"), ("啊，哪来的母老虎","v2")])

    prompt = "<狐女>（温柔）：“我是狐狸精，为报恩而来。” <周铻>（友好）：“侄儿，兰州的生活还习惯吗？”"
    prompt2 = "画外音：传说，他能以风洗手。"
    conversation = extract_conversation(prompt2, "narratator")
    print(conversation)
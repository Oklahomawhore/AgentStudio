import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from threading import Lock
import uuid
import re
import base64
from typing import List,Tuple, Dict
from functools import partial

import torch
import pickle
import dotenv
import ChatTTS
from moviepy import *
from pydub import AudioSegment  # For concatenating WAV files
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from uuid import uuid4

from api_interface import kling_imggen_agent, kling_img2video_agent, kling_text2video_agent, morph_images_agent, kling_ref2video_agent
from util import generate_hash, save_to_disk, load_from_disk

dotenv.load_dotenv()

tts_lock = Lock()
chat = ChatTTS.Chat()
chat.load(compile=False)
# In this file, we decide which genration backend to call and
# hide the details of generation logic from the main code



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
        img_base64 = kling_imggen_agent("A single-character portrait of : " + prompt.replace("-",","))
        save_to_disk(img_base64, file_path)
    except Exception as e:
        print(f"Error in generating charactor: {str(e)}") 
    # save base64 img to file and return path
    return img_base64

def gen_video(prompt, image, i2v=False, api_base=None):
    try:
        hash_string = str(prompt) + str(image)
        prompt_hash = generate_hash(hash_string)  # Combine image path for uniqueness
        file_path = f"videos/{prompt_hash}.pkl"
        
        # Check if the video already 
        existing_video = load_from_disk(file_path)

        extra_kwargs = {}
        if api_base is not None:
            extra_kwargs["api_base"] = api_base

        if existing_video is not None:
            print(f"<GEN_VIDEO> prompt: {prompt} image: {image} already exists")
            return existing_video
        if image != '':
            assert(isinstance(image, str) or isinstance(image, list)), "Image must be a string or a list of strings"
            if isinstance(image, str):
                video_list, screenshot_list = kling_img2video_agent(image, prompt, **extra_kwargs)
            elif isinstance(image, list):
                video_list, screenshot_list = kling_ref2video_agent(image, prompt, **extra_kwargs)
            else:
                raise ValueError(f"Unexpected image argument type, expected str or list, got {type(image)}")
        else:
            video_list, screenshot_list = kling_text2video_agent(prompt, **extra_kwargs)
        # time.sleep(10)
        # with open("imgs/dog1.txt", "r") as f:
        #     img_b64 = f.read().strip()
        # video_list, screenshot_list = "/data/wangshu/wangshu_code/ISG/ISG_agent/videos/118f0090-14c2-40a7-b6b5-4a1c62ae0531.mp4", [img_b64, img_b64]
        save_to_disk((video_list, screenshot_list), file_path)

    except Exception as e:
        print(f"Error in generating video: {str(e)} prompt: {prompt} image: {image}")
    
    return video_list, screenshot_list

# Music Generation using hash as input (returns file path)
def gen_music(prompts: List[str], durations: List[float]):
    """
    Generate music based on unique prompts, combining durations for identical prompts
    
    Args:
        prompts: List of music prompt strings
        durations: List of durations in seconds for each video clip
    
    Returns:
        str: Path to the generated music file
    """
    print(f"Processing {len(prompts)} music prompts with {len(durations)} duration values")
    
    if not prompts or not durations:
        print("No valid music prompts or durations provided")
        return None
    
    # 创建字典来存储每个唯一提示词及其对应的总持续时间
    prompt_to_duration = {}
    
    # 合并相同提示词的持续时间
    for prompt, duration in zip(prompts, durations):
        if prompt is None:
            continue
            
        if prompt in prompt_to_duration:
            prompt_to_duration[prompt] += duration
        else:
            prompt_to_duration[prompt] = duration
    
    print(f"Found {len(prompt_to_duration)} unique music prompts after combining")
    
    # 现在我们有了每个唯一提示词及其总持续时间
    combined_prompts = []
    combined_durations = []
    
    for prompt, total_duration in prompt_to_duration.items():
        combined_prompts.append(prompt)
        combined_durations.append(total_duration)
        print(f"Combined prompt: '{prompt}' with total duration: {total_duration:.2f}s")
    
    # 如果只有一个唯一的提示词，直接生成并返回
    if len(combined_prompts) == 1:
        return generate_single_music(combined_prompts[0], combined_durations[0])
    
    # 如果有多个唯一提示词，为每个提示词生成音乐并合并
    return generate_and_combine_music(combined_prompts, combined_durations)

def generate_single_music(prompt: str, duration: float) -> str:
    """生成单个音乐片段"""
    print(f"Generating music for prompt: '{prompt}' for duration {duration:.2f} seconds")
    try:
        # Generate hash from the prompt and duration
        prompt_hash = generate_hash(f"{prompt}_{duration:.1f}")
        file_path = os.path.abspath(f"music/{prompt_hash}.wav")

        # Check if the music already exists
        if os.path.exists(file_path):
            print(f"<GEN_MUSIC> prompt: '{prompt}' already exists")
            return file_path

        # Music generation code
        # TODO: try subprocess execution to avoid OOM.
        model = MusicGen.get_pretrained('facebook/musicgen-stereo-melody-large')
        model.set_generation_params(duration=duration)
        wav = model.generate([prompt]) 

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            file_path = audio_write(os.path.abspath(f'music/{prompt_hash}'), one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
        return file_path
    except Exception as e:
        print(f"Error in generating music: {str(e)}")
        return None

def generate_and_combine_music(prompts: List[str], durations: List[float]) -> str:
    """为多个提示词生成音乐并合并"""
    music_files = []
    
    # 为每个提示词生成音乐
    for prompt, duration in zip(prompts, durations):
        music_file = generate_single_music(prompt, duration)
        if music_file:
            music_files.append(music_file)
    
    if not music_files:
        print("Failed to generate any music files")
        return None
    
    if len(music_files) == 1:
        return music_files[0]
    
    # 合并所有生成的音乐文件
    combined_hash = generate_hash("_".join([f"{p}_{d:.1f}" for p, d in zip(prompts, durations)]))
    combined_file = os.path.abspath(f"music/combined_{combined_hash}.wav")
    
    # 检查合并文件是否已存在
    if os.path.exists(combined_file):
        print(f"Combined music file already exists: {combined_file}")
        return combined_file
    
    try:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(combined_file), exist_ok=True)
        
        # 合并音频文件
        combined_audio = None
        for file in music_files:
            segment = AudioSegment.from_wav(file)
            if combined_audio is None:
                combined_audio = segment
            else:
                combined_audio += segment
        
        # 导出合并后的文件
        combined_audio.export(combined_file, format="wav")
        print(f"Successfully combined music files into: {combined_file}")
        return combined_file
    
    except Exception as e:
        print(f"Error combining music files: {str(e)}")
        # 如果合并失败，返回第一个成功生成的音乐文件
        return music_files[0]


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
            conversation = prompt.split("：", 1)[-1]
        except IndexError as e:
            conversation = prompt.split(':', 1)[-1]
        finally:
            print(prompt)
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
    global chat

    combined_file = f"audio/{prompt.replace('/', '').replace('：', '').replace(' ', '')}.wav"

    if os.path.exists(combined_file):
        print(f"Combined audio for prompt {prompt} already exists")
        return combined_file

    
        # Extract all speaker-conversation pairs

    os.makedirs(speaker_embeddings_cache_dir, exist_ok=True)
    os.makedirs("audio", exist_ok=True)

    saved_wavs = []

    for speaker, description in voice_direction.items():
        
        conversation = extract_conversation(prompt, speaker)
        print(f"[TTS] {speaker} : {conversation}")
        if len(conversation) == 0:
            conversation = prompt
        
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
        with tts_lock:
            print(f"Infering {conversation}...")
            wavs = chat.infer(
                conversation,  # Use the extracted text for the speaker
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )
        # Save individual WAV file

        file_name = os.path.abspath(f"audio/{str(uuid4())}.wav")
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

def concat_video(video_clips: List[VideoFileClip], music_path, tts_paths, task_dir):
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
        total_length = 0.0
        prefix_sum = []
        for clip in video_clips:
            prefix_sum.append(total_length)
            total_length += clip.duration

        # Concatenate video clips
        final_video = concatenate_videoclips(video_clips, method="compose")

        # Load audio clips
        if os.path.exists(music_path):
            music_audio = AudioFileClip(music_path).with_volume_scaled(0.5)
            music_audio = music_audio.subclipped(0, min(total_length, music_audio.duration))
        
        if len(tts_paths) > 0:
            tts_clip = [AudioFileClip(tts_path).with_start(prefix_sum[index]) for index, tts_path in enumerate(tts_paths) if tts_path is not None]

        # Combine audio tracks (dialogue + music)
        if len(tts_paths) > 0:
            final_audio = CompositeAudioClip([music_audio, *tts_clip])
            final_video = final_video.with_audio(final_audio)

        # Save the final video
        output_path = os.path.abspath(f"{task_dir}/final_video_{uuid.uuid4()}.mp4")
        final_video.write_videofile(output_path)
        return output_path

    except Exception as e:
        print(f"Error during video concatenation: {e}")
        return None

def generate_all(video_task, music_task, tts_task, task_dir, video_gen_api_base=None):
    # Create separate executors for each task category
    with ThreadPoolExecutor(max_workers=5, thread_name_prefix='gen_video_threads') as executor:
        start = time.time()
        
        # Use partial to create a new function with api_base parameter set
        gen_video_with_api = partial(gen_video, api_base=video_gen_api_base)
        
        # First generate videos that need screenshots
        t2v_tasks = [task for task in video_task if task[1] != "<LastFrame>"]
        vid_results_stage_1 = list(executor.map(gen_video_with_api, *zip(*t2v_tasks)))
        
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
                    # Also use the api_base parameter here
                    result = gen_video(task[0], last_screenshot, i2v=True, api_base=video_gen_api_base)
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
        # tts_results = executor.map(gen_tts, *zip(*tts_task))
        tts_results = []
        video_clips = [VideoFileClip(video[0]) for video in vid_results]
        video_lengths = []
        for clip in video_clips:
            video_lengths.append(clip.duration)
        # music_result = executor.submit(gen_music, music_task, video_lengths)
        final = concat_video(video_clips, "", list(tts_results), task_dir)
        end = time.time()
        print(f"Time taken: {end-start:.6f}")

    return final


if __name__ == "__main__":
    # generate_all([("丛林老者正在山野里打猎，他的脸上写满了沧桑", "image1"), ("老者正在打猎，突然间他的背后出现一只老虎的身影，老者大惊失色" ,"image2")], ["竹林小调，笛子旋律，轻快的", "管弦乐队，突然爆发的旋律，渲染紧张气氛"], [("我自山上来，今天来打猎","v1"), ("啊，哪来的母老虎","v2")])

    prompt = "<狐女>（温柔）：“我是狐狸精，为报恩而来。” <周铻>（友好）：“侄儿，兰州的生活还习惯吗？”"
    prompt2 = "画外音：传说，他能以风洗手。"
    conversation = extract_conversation(prompt2, "narratator")
    print(conversation)
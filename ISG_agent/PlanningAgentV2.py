import os
import json
from argparse import ArgumentParser
from Prompt.PromptVideo_multiple import PLANNING_PROMPT
import base64
from tqdm import tqdm
import re
from ToolAgent import tool_agent
from PIL import Image, ImageFile
from openai import OpenAI
from anthropic import  Anthropic
import http.client
import dotenv
import replicate
import oss2, requests
import uuid

dotenv.load_dotenv()

benchmark_file = "../ISV_eval/VideoStoryTelling/video_storytelling_mini.json" # <path to benchmark.jsonl>

OpenAIClient = OpenAI(
   api_key=os.getenv("OPENAI_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)
ClaudeClient = OpenAI(
   api_key=os.getenv("CLAUDE_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

IMAGE_ROOT = "../ISV_eval/VideoStoryTelling/"

benchmark_file = "../ISV_eval/VideoStoryTelling/video_storytelling_mini.json"

def save_error_file(task_dir, error_message):
    os.makedirs(task_dir, exist_ok=True)
    error_file = os.path.join(task_dir, "error.log")
    with open(error_file, "a") as f:
        f.write(error_message + "\n")

def load_input_json(json_file:str):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_plan_json(json_data,file):
    dir_name = os.path.dirname(file)
    print(f"Directory: {dir_name}")
    os.makedirs(f"{dir_name}", exist_ok=True)
    with open(file, 'w') as f:
        json.dump(json_data, f, indent=4)

def save_result_json(json_data,task_dir):
    os.makedirs(task_dir, exist_ok=True)
    with open(f'{task_dir}/result.json', 'w') as f:
        json.dump(json_data, f, indent=4)
        
        
def get_image_media_type(image_data):
    """
    Detects the image media type based on the magic number in the binary data.
    """
    if image_data[:4] == b'\x89PNG':
        return "image/png"
    elif image_data[:2] == b'\xFF\xD8':
        return "image/jpeg"
    else:
        raise ValueError("Unsupported image format")

def preprocess_task(task, task_dir, plan_model):
    if plan_model == "claude":
        Query = task["Query"]
        Dict = {"text":"","images":[]}
        Dict_for_plan = {"text":"","images":[]}
        for seg in Query:
            if seg['type'] == "text":
                Dict['text'] += seg['content']
                Dict_for_plan['text'] += seg['content']
            elif seg['type'] == "image":
                image_path = os.path.join(IMAGE_ROOT, seg['content'])
                bak = 0
                try:
                    # print(os.path.getsize(image_path))
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                        o_size = os.path.getsize(image_path)
                        img_bak = image_path
                        if (o_size*1.33) > 5242880:
                            bak = 1
                            print(f"Image size is too large: {(1.33*o_size)} bytes for {image_path}")
                            print("Compression Implemented")
                            ImageFile.LOAD_TRUNCATED_IMAGES = True
                            while (o_size*1.33) > 5242880:
                                img = Image.open(img_bak)
                                x,y = img.size
                                
                                out = img.resize((int(x*0.9),int(y*0.9)))
                                
                                try:
                                    img_base_dir = os.path.dirname(image_path)
                                    img_file_prefix = os.path.basename(image_path).split(".")[0]
                                    img_bak = img_file_prefix + "_bak.png"
                                    img_bak = os.path.join(task_dir,img_bak)
                                    print(img_bak)
                                    out.save(os.path.abspath(img_bak),quality=95)
                                    print(f"Compressed Image saved to {img_bak}")
                                except Exception as e:
                                    print(e)
                                    break
                                o_size = os.path.getsize(img_bak)
                                    
                        if bak == 1:
                            with open(img_bak, 'rb') as f:
                                image_data = f.read()
                        img_base64 = base64.b64encode(image_data).decode("utf-8")
                        img_media_type = get_image_media_type(image_data)
                        Dict['images'].append({"content":img_base64,"media_type":img_media_type})
                        Dict_for_plan['images'].append(os.path.abspath(img_bak))
                except Exception as e:
                    print(f"Error reading image: {str(e)} for {image_path}")
                    continue
        content = []
        for img in Dict["images"]:
            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": img['media_type'], 
                                    "data": img['content'],
                                },
                            })
        content.append({
            "type": "text",
            "text": "###Task:"+Dict['text'] + "\n\n"
        })
        
        content.append({
            "type": "text",
            "text": "###Structure Requirement:" + f"{extract_structure(task)['Answer_str']}"+ "\n\n"
        })
        message = []
        message.append({
            'role': 'user',
            'content':content
        })
        return message,Dict,Dict_for_plan
    elif plan_model == "openai":
        Query = task["Query"]
        Dict = {"text": "", "images": []}
        Dict_for_plan = {"text": "", "images": []}
        for seg in Query:
            if seg['type'] == "text":
                Dict['text'] += seg['content']
                Dict_for_plan['text'] += seg['content']
            elif seg['type'] == "image":
                image_path = os.path.join(IMAGE_ROOT, seg['content'])
                bak = 0
                try:
                    with open(image_path, 'rb') as f:
                        image_data = f.read()
                        o_size = os.path.getsize(image_path)
                        img_bak = image_path
                        if (o_size * 1.33) > 5242880:
                            bak = 1
                            print(f"Image size is too large: {(1.33 * o_size)} bytes for {image_path}")
                            print("Compression Implemented")
                            ImageFile.LOAD_TRUNCATED_IMAGES = True
                            while (o_size * 1.33) > 5242880:
                                img = Image.open(img_bak)
                                x, y = img.size
                                out = img.resize((int(x * 0.9), int(y * 0.9)))
                                try:
                                    img_base_dir = os.path.dirname(image_path)
                                    img_file_prefix = os.path.basename(image_path).split(".")[0]
                                    img_bak = img_file_prefix + "_bak.png"
                                    img_bak = os.path.join(task_dir, img_bak)
                                    print(img_bak)
                                    out.save(os.path.abspath(img_bak), quality=95)
                                    print(f"Compressed Image saved to {img_bak}")
                                except Exception as e:
                                    print(e)
                                    break
                                o_size = os.path.getsize(img_bak)
                        if bak == 1:
                            with open(img_bak, 'rb') as f:
                                image_data = f.read()
                        img_base64 = base64.b64encode(image_data).decode("utf-8")
                        img_media_type = get_image_media_type(image_data)
                        Dict['images'].append({"content": img_base64, "media_type": img_media_type})
                        Dict_for_plan['images'].append(os.path.abspath(img_bak))
                except Exception as e:
                    print(f"Error reading image: {str(e)} for {image_path}")
                    continue
        # Construct the messages for OpenAI API
        content = []
        for img in Dict["images"]:
            content.append({
                "type": "image_url",
                "image_url":{
                    "url": f"data:{img['media_type']};base64,{img['content']}"
                    }
                ,
            })
        content.append({
            "type": "text",
            "text": f"###Task:{Dict['text']}\n\n"
        })
        
        # content.append({
        #     "type": "text",
        #     "text": f"###Structure Requirement:{extract_structure(task)['Answer_str']}\n\n"
        # })
        # print(f"Structure: {extract_structure(task)['Answer_str']}")
        messages = [
            {"role": "system", "content": PLANNING_PROMPT},
            {"role": "user", "content": content},
        ]
        
        return messages, Dict, Dict_for_plan

def double_check(task_dir,task_content,structure,response_text,error_message):
    """
    The function double-checks the response json format
    1. If the json in the response is not [{"Task": ..., },{"Task": ...},...]. Modify it to the required format(Manually)
    2. Remove the wrong Task name in each step(Manually)
    3. Regenerate the response json if restriction conflict (Have to let the Large Language model to identify which tool does the instruction want to use. Here are some examples: "Generate an image" means generation one image, use ImageGeneration, "Transform the image into" means edit an image, use ImageEdit,...) Then, identify is there any restriction conflict, if there is, regenerate the response json.
    """
    try:
        response_text = reconstruction(task_content,structure,response_text,error_message)
        return response_text
    except Exception as e:
        print(f"Error calling reconstruction: {str(e)}")
        er = f"Error calling reconstruction: {str(e)}"
        save_error_file(task_dir, er)
        return response_text

def reconstruction(task_content,structure,response_text,error_message):
    prompt =f"""
Error: {error_message}  
Given the Task:  

{task_content}  

Given the Structure Requirement:  

{structure}  

Given the following previous response text:  

{response_text}  

### **Instructions for Reconstruction**:  

1. **Ensure JSON Compliance**:  
   - The plan must be formatted as a valid JSON array: `[{{"Task":"", ...}}, {{"Task":"", ...}}, ...]`.  

2. **Correct Misuse of Task Names**:  
   - Replace any invalid Task Names with `"Call_tool"`. Retain the remaining content as-is.

3. **Enforce Tool Usage Restrictions**:  
   - No More than one ImageGeneration tool can be used in the plan.
   - Decide the appropriate tool for each step:  
     - Add "Use ImageGeneration tool:" to the beginning of `Input_text` when the task has no input images.  
     - Add "Use Text2Video_VideoGeneration tool:" to the beginning of `Input_text` when there is *NO* Input_images and it is not the first Call_tool step.
     - Add "Use Image2Video_VdieoGeneration tool:" to the beginning of `Input_text` when the task is Input_images.

4. **Placeholder Consistency**:  
   - Use `#image{{ID}}#` for original input images, starting from 1.  
   - Use `<GEN_img{{ID}}>` for generated images, starting from 0. 
   - Use `<GEN_vid{{ID}}>` for generated videos, starting from 0.
   - Use `<GEN_text{{ID}}>` for generated text, starting from 0.

5. **Story Consistency**:  
   - For each step, ensure characters are consistently described with unique attributes: face, costumes, age, profession, race, and nationality.  

6. **Output Plan Format**:  
   - Use `"Output": "<WAIT>"` as the placeholder in every step.  

---

### **Plan Structure Example**:  

**Example 1** (Simple story with generated images):  
```json
[
    {{
        "Step": 1,
        "Task": "Call_tool",
        "Input_text": "Use ImageGeneration tool: Alice, a 34-year-old mother with two kids, has a kind and warm face. She wears a pastel-colored apron over her casual dress and is of European descent with a soft-spoken personality. Professionally, Alice is a homemaker.",
        "Input_images": [],
        "Output": "<WAIT>"
    }},
    {{
        "Step": 2,
        "Task": "Call_tool",
        "Input_text": "Alice steps outside her cozy suburban home on a sunny morning. She wears her apron and casual dress. Modify her environment to include a vibrant garden and children playing on the lawn.",
        "Input_images": ["<GEN_img0>"],
        "Output": "<WAIT>"
    }},
    {{
        "Step": 3,
        "Task": "Call_tool",
        "Input_text": "Alice prepares a picnic basket with fresh fruits and sandwiches for her children. She smiles warmly as she watches them play.",
        "Input_images": ["<GEN_img1>"],
        "Output": "<WAIT>"
    }},
    {{
        "Step": 4,
        "Task": "Call_tool",
        "Input_text": "Alice sits on a picnic blanket with her children, enjoying a sunny afternoon together.",
        "Input_images": ["<GEN_img2>"],
        "Output": "<WAIT>"
    }},
    {{
        "Step": 5,
        "Task": "Call_tool",
        "Input_text": "Alice and her children share a meal together, laughing and enjoying the sunny day.",
        "Input_images": ["<GEN_img3>"],
        "Output": "<WAIT>"
    }},
    {{
        "Step": 6,
        "Task": "AddVideo",
        "Input_text": "Alice and her children clean up after the picnic, putting away the picnic basket and blanket.",
        "Input_images": ["<GEN_img4>"],
        "Output": "<WAIT>"
    }}
]
```
"""
    try:
        completion = OpenAIClient.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=4096,
            temperature=0.7
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling Azure API: {str(e)}")
        print("Switch to Claude API")
        try:
            completion = ClaudeClient.chat.completions.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
            )
            # response_text = response.content[0].text
            response_text = completion.choices[0].message.content
        except Exception as e:
            print(f"Error calling Claude API: {str(e)}")
            raise ValueError("Error calling API")
    return response_text


def extract_plan_from_response(response_text, plan_file):
    """
    Extracts the first valid JSON object or array from the response text and saves it to a file.
    
    Args:
        response_text (str): The text containing JSON data.
        plan_file (str): The file path where the extracted JSON will be saved.
    
    Raises:
        ValueError: If no valid JSON is found in the response text.
    """
    response_text = response_text.strip()
    start_index = response_text.find('[')
    if start_index == -1:
        raise ValueError("No valid JSON found in the response")

    # Find the position of the last closing bracket ']'
    open_brackets = 0
    end_index = start_index
    for idx in range(start_index, len(response_text)):
        if response_text[idx] == '[':
            open_brackets += 1
        elif response_text[idx] == ']':
            open_brackets -= 1
            if open_brackets == 0:
                end_index = idx + 1  # include the closing bracket
                break
    
    json_text = response_text[start_index:end_index]

    try:
        extracted_json = json.loads(json_text)
        save_plan_json(extracted_json, plan_file)
        print(f"JSON extracted and saved to {plan_file}")
    except json.JSONDecodeError as e:
        raise ValueError(f"No valid JSON found in the response: {str(e)}")


def decode_result_into_jsonl(result, task_dir, benchmark_file):
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    with open(benchmark_file, 'r') as f:
        benchmark = json.load(f)

    # Get the last 4 characters from the task_dir and convert them to an integer
    ID = int(task_dir[-4:])
    print(ID)

    # Retrieve the corresponding benchmark entry based on the ID
    benchmark = benchmark[ID]

    # Regular expression pattern to find <boi>...<eoi> and capture the content inside
    image_pattern = re.compile(r'<boi>(.*?)<eoi>', re.DOTALL)

    # Find all matches of images in the result string
    image_matches = list(image_pattern.finditer(result))

    last_end = 0
    image_counter = 1

    # Extract text and images from the result
    decoded_output = []
    for match in image_matches:
        start, end = match.span()

        # Extract text content before this image
        if start > last_end:
            text_content = result[last_end:start].strip()
            if text_content:
                decoded_output.append({"type": "text", "content": text_content})

        # Extract the image path inside <boi> and <eoi>
        image_path = match.group(1).strip()
        if image_path and os.path.exists(image_path):
            decoded_output.append({
                "type": "video",
                "caption": "",  # Add a caption if needed
                "content": os.path.abspath(image_path)
            })
            image_counter += 1

        # Update last_end to the end of the current match
        last_end = end

    # Check for any remaining text after the last image
    if last_end < len(result):
        text_content = result[last_end:].strip()
        if text_content:
            decoded_output.append({"type": "text", "content": text_content})

    # Construct the final JSON object in the desired format
    jsonl_entry = {
        "id": f"{ID:04}",  # Ensures ID is formatted as a zero-padded 4-character string
        "Category": benchmark.get("Category"),
        "original_query_path": benchmark.get("original_query_path"),
        "original_golden_path": benchmark.get("original_golden_path"),
        "Query": benchmark.get("Query"),
        "Golden": benchmark.get("Golden", []),
        "output": decoded_output
    }

    # Write the final JSON object to the JSONL file
    jsonl_file_path = os.path.join(task_dir, "result.jsonl")
    with open(jsonl_file_path, 'a') as jsonl_file:
        # Write the whole JSON object as a single line in the file
        jsonl_file.write(json.dumps(jsonl_entry) + '\n')

    return jsonl_entry

def replace_image_placeholders_in_plan(plan, Dict):
    """
    This function replaces image placeholders in the plan with the actual image content from Dict['images'].
    """
    image_placeholder_pattern = r"#image(\d+)#"  # Matches the image placeholder with a number
    for step in plan:
        
        # Function to replace placeholders with actual image content
        def replace_image_placeholder(match):
            image_index = int(match.group(1)) - 1  # Get the image index from #image{i}#
            if 0 <= image_index < len(Dict['images']):
                return f"{Dict['images'][image_index]}"
            else:
                return match.group(0)  # Return the original placeholder if index is out of range

        # Replace image placeholders in Input_images
        print(step)
        input_images = step.get('Input_images', [])
        new_input_images = []
        for img in input_images:
            new_img = re.sub(image_placeholder_pattern, replace_image_placeholder, img)
            new_input_images.append(new_img)
        step['Input_images'] = new_input_images

    return plan



def handle_caption_step(task_dir,step,result):
    # Caption step: input all the required images in Input_images list, and generate the caption
    
    input_text = step.get('Input_text', '')
    input_images = step.get('Input_images', [])
    
    tool_input = {
        "Input_text": input_text,
        "Input_images": input_images
    }
    
    tool_input_json = json.dumps(tool_input)
    
    try:
        response = tool_agent(tool_input_json, "Caption")
        # Process response
        response_text = response.get('text', '')
        if not response_text:
            print(f"Expected 'text' in response for task 'Caption', got {list(response.keys())}")
        result += response_text
        step['Output'] = response_text
    except Exception as e:
        print(f"Error calling tool_agent in handle_caption_step: {e}")
        error_message = f"Error calling tool_agent in handle_caption_step: {e}"
        save_error_file(task_dir, error_message)
        step['Output'] = None  # Ensure Output is set even on error
        result += error_message
    return result

def upload_video(video_path, upload_url):
    """
    Upload a video file to a hosting service and return the URL of the uploaded file.
    """

    endpoint = 'http://oss-cn-shanghai.aliyuncs.com' # Suppose that your bucket is in the Hangzhou region.

    auth = oss2.Auth(os.getenv('ALIYUN_ACCESS_KEY_ID'), os.getenv('ALIYUN_ACCESS_KEY_SECRET'))
    bucket = oss2.Bucket(auth, endpoint, 'video-storage-6999')
    key = video_path.split("/")[-1]

    rs = bucket.put_object_from_file(key, video_path)

    if rs.status == 200:
        print(f"Uploaded video to OSS: {key}")
    
        return f"https://video-storage-6999.oss-cn-shanghai.aliyuncs.com/{key}"
    else:
        print(f"Error uploading video to OSS: {rs.status}")
        return None

def handle_audio_addition(video_path, task_dir, step, result):
    """
    Handles video-to-audio generation by:
    - Uploading the video to a hosting service.
    - Running the Replicate API for video-to-audio generation.
    - Saving the returned video to disk.
    """
    # Upload the video to a hosting service
    uploaded_video_url = upload_video(video_path, "")
    print(f"Uploaded video URL: {uploaded_video_url}")

    # Prepare input for Replicate API
    input_data = {
        "video": uploaded_video_url,
        "prompt": step.get("Input_text", ""),
        "duration": 5,
        "negative_prompt": "conversation"
    }

    # Call Replicate API
    output_url = replicate.run(
        "zsxkib/mmaudio:4b9f801a167b1f6cc2db6ba7ffdeb307630bf411841d4e8300e63ca992de0be9",
        input=input_data
    )
    
    # Download and save the output video
    key = video_path.split("/")[-1]
    output_video_path = os.path.join(task_dir, f"{key.split('.')[0]}_with_audio.mp4")

    response = requests.get(output_url, stream=True)
    response.raise_for_status()  # Ensure the request was successful
    with open(output_video_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Generated video with audio saved to: {output_video_path}")
    return output_video_path

def extract_structure(benchmark):

    golden_list = []
    text_i = 1
    img_i = 1
    video_i = 1
    for item in benchmark["Golden"]:
        if item["type"] == "text":
            golden_list.append(f"<gen_text{text_i}>")
            text_i += 1
        elif item["type"] == "image":
            golden_list.append(f"<gen_img{img_i}>")
            img_i += 1
        elif item["type"] == "video":
            golden_list.append(f"<gen_vid{video_i}>")
            video_i += 1

    return {
    "Answer": golden_list,
    "Answer_str": " ".join(golden_list)
    }
    
def Execute_plan(plan,task, task_dir):
    if os.path.exists(f"{task_dir}/error.log"):
        return
    result = ""
    Call_tool_times = 0
    VDGen = 0
    generated_image_list = []
    generated_video_list = []
    generated_text_list = []
    for i,step in enumerate(plan):
        Task = step.get("Task", "")
        try:
            if Task == "Call_tool":
                Call_tool_times += 1
                input_images = step.get('Input_images', [])
                processed_input_images = []
                for img in input_images:
                    if img is None:
                        print(f"Warning: None image encountered in step {i+1}.")
                        processed_input_images.append(None)
                        continue
                    if '<GEN_' in img:
                        # Extract the id from the placeholder
                        match = re.match(r'<GEN_img(\d+)>', img)
                        if match:
                            id = int(match.group(1))
                            if id < len(generated_image_list):
                                processed_input_images.append(generated_image_list[id])
                                match_gen = re.match(r'<GEN_img(\d+)>', generated_image_list[id])
                                if match_gen:
                                    print(f"Warning: Nested <GEN> placeholder in step {i+1}.")
                            else:
                                print(f"Error: <GEN_{id}> is out of range in generated_image_list.")
                                processed_input_images.append(f"<GEN_{id}>")
                        else:
                            print(f"Error: Invalid placeholder format '{img}' in step {i+1}.")
                            processed_input_images.append(None)
                    else:
                        processed_input_images.append(img)
                input_images = processed_input_images
                step['Input_images'] = input_images
                input_text = step.get('Input_text', '')
                generated_text_list.append(input_text)

                tool_input = {
                    "Input_text": input_text,
                    "Input_images": input_images
                }
                tool_input_json = json.dumps(tool_input)
                try:
                    response = tool_agent(tool_input_json, Task)
                except Exception as e:
                    print(f"Error calling tool_agent for task '{Task}' in step {i+1}: {e}")
                    error_message = f"Error processing task in step{i+1}: {str(e)}"
                    # print(error_message)
                    save_error_file(task_dir, error_message)
                    response = {'text': '', 'images': [], 'video': ""} 
                step['Output'] = []
                images = response.get('images', [])
                video = response.get('video', "")
                if len(images) >= 2:
                    VDGen += 1
                    # screenshots generated, leave only last image.
                    images = images[-1:]
                if len(images) == 0 and video == "":
                    print(f"Warning: No images generated in step {i+1}")
                    error_message = f"Warning: No images generated in step {i+1}"
                    save_error_file(task_dir, error_message)
                    break
                for j, img_base64 in enumerate(images):
                    image_filename = f"image_step_{i+1}_{j+1}.png"
                    image_path = os.path.join(task_dir, image_filename)
                    try:
                        with open(image_path, 'wb') as f:
                            f.write(base64.b64decode(img_base64))
                        step['Output'].append(image_path)
                    except Exception as e:
                        print(f"Error saving image file: {e}")
                        error_message = f"Error saving image file: {e}"
                        save_error_file(task_dir, error_message)
                        step['Output'].append(None)
                generated_image_list.extend(step['Output'])
                
                if not video == "":
                    step['Output'].append(video)
                    generated_video_list.append(video)
                for i,img in enumerate(generated_image_list):
                    if img is None:
                        generated_image_list[i] = f"<GEN_{i}>"
                        
        except Exception as e:
            print(f"Error processing step {i+1}: {str(e)}")
            error_message = f"Error processing task in step{i+1}: {str(e)}"
            print(error_message)
            save_error_file(task_dir, error_message)
            continue
    
    for i, step in enumerate(plan):
        try:
            Task = step.get("Task", "")
            input_images = step.get('Input_images', [])
            input_text = step.get('Input_text', '')
            processed_input_images = []
            for img in input_images:
                if img is None:
                    print(f"Warning: None image encountered in step {i+1}.")
                    processed_input_images.append(None)
                    continue
                if '<GEN_' in img:
                    # Extract the id from the placeholder
                    match = re.match(r'<GEN_vid(\d+)>', img)
                    if match:
                        id = int(match.group(1))
                        if id < len(generated_video_list):
                            processed_input_images.append(generated_video_list[id])
                            match_gen = re.match(r'<GEN_vid(\d+)>', generated_video_list[id])
                            if match_gen:
                                print(f"Warning: Nested <GEN> placeholder in step {i+1}.")
                                
                        else:
                            print(f"Error: <GEN_vid{id}> is out of range in generated_image_list.")
                            processed_input_images.append(f"<GEN_vid{id}>")
                    else:
                        print(f"Error: Invalid placeholder format '{img}' in step {i+1}.")
                        processed_input_images.append(None)
                else:
                    processed_input_images.append(img)
            #Handle text input placeholders
            processed_text_input = ""
            if 'GEN_' in input_text:
                match = re.match(r'<GEN_text(\d+)>', input_text)
                if match:
                    id = int(match.group(1))
                    if id < len(generated_text_list):
                        processed_text_input = generated_text_list[id]
                        match_gen = re.match(r'<GEN_(\d+)>', generated_text_list[id])
                        if match_gen:
                            print(f"Warning: Nested <GEN> placeholder in step {i+1} for text.")
                            
                    else:
                        print(f"Error: <GEN_{id}> is out of range in generated_text_list.")
                        processed_text_input = f"<GEN_{id}>"
                else:
                    print(f"Error: Invalid placeholder format '{input_text}' in step {i+1}.")
                    processed_text_input = ""
            input_images = processed_input_images
            step['Input_images'] = input_images
            step['Input_text'] = processed_text_input

            if Task == "AddVideo":
                # add audio to video input
                for img in input_images:
                    # call v2a model for audio addition
                    img = handle_audio_addition(img, task_dir, step, result)
                    result += f"<boi>{img}<eoi>"
                continue
            
            if Task == "Caption":
                print(step['Input_images'])
                print(step['Input_text'])

                result = handle_caption_step(task_dir,step,result)
                if step["Output"] is None:
                    error_message = f"Error processing task in step{i+1}: Caption output is None"
                    print(error_message)
                    save_error_file(task_dir, error_message)
                continue
            elif Task == "Call_tool":
                continue
            else:
                # print(f"Unknown task '{Task}' in step {i+1}. Skipping this step.")
                error_message = f"Unknown task '{Task}' in step {i+1}. Skipping this step."
                print(error_message)
                save_error_file(task_dir, error_message)
                break
        except Exception as e:
            print(f"Error processing step {i+1}: {str(e)}")
            error_message = f"Error processing task in step{i+1}: {str(e)}"
            print(error_message)
            save_error_file(task_dir, error_message)
            continue
    plan_file_final = f"{task_dir}/plan_{task.get('id', '0000')}_final.json"
    result_json = decode_result_into_jsonl(result, task_dir,benchmark_file)
    # Post-process

    
    save_result_json(result_json, task_dir)
    save_plan_json(plan, plan_file_final)
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, help="Input json file")
    parser.add_argument("--outdir", type=str, help="Output directory")
    args = parser.parse_args()
    
    input_json = args.input_json
    outdir = args.outdir
    
    data = load_input_json(input_json)
    for task in tqdm(data):

        print(f"Processing Task: {task.get('id', '0000')}")
        task_dir = f"{outdir}/Task_{task.get('id', '0000')}"
        plan_file = f"{task_dir}/plan_{task.get('id', '0000')}.json"
        result_file = f"{task_dir}/result.json"
        error_file = f"{task_dir}/error.log"
        os.makedirs(task_dir, exist_ok=True)
        if os.path.exists(result_file) and not os.path.exists(error_file):
            print(f"Skipping task {task.get('id', '0000')}, result file already exists")
            continue
        if os.path.exists(plan_file):
            print(f"Skipping task {task.get('id', '0000')} plan generation, plan file already exists, directly extract from it")
            plan = load_input_json(plan_file)
            response_text = json.dumps(plan)
            if not isinstance(plan, list):
                error_message = f"Error: Plan file is not a list: {plan_file}"
                print(error_message)
                save_error_file(task_dir, error_message)
        else:
            try: 
                messages, Dict, Dict_for_plan = preprocess_task(task, task_dir, plan_model="openai")

                completion = OpenAIClient.chat.completions.create(
                    model='gpt-4o-2024-08-06',
                    messages=messages,
                    max_completion_tokens=4096,
                    temperature=0.7
                )
                response_text = completion.choices[0].message.content
                try:
                    extract_plan_from_response(response_text, plan_file)
                except Exception as e:
                    print(f"Error extracting plan: {str(e)}")
                    print(completion)
                    raise ValueError("Error extracting plan")
            except Exception as e:
                print(f"Error in Azure API, switch to Claude API: {str(e)}") 
                print("Switch to Claude API")
                # print("Error in Azure API, switch to Claude API")
                messages,Dict,Dict_for_plan = preprocess_task(task, task_dir, plan_model="openai")
                
                completion = ClaudeClient.chat.completions.create(
                    model = "claude-3-5-sonnet-20240620",
                    max_completion_tokens=8192,
                    messages=messages,
                    temperature=0.7,
                )
                response_text = completion.choices[0].message.content
                try:
                    extract_plan_from_response(response_text, plan_file)
                except Exception as e:
                    print(f"Error extracting plan: {str(e)}")
                    print(completion)
                    continue
            plan = load_input_json(plan_file)

            if not isinstance(plan, list):
                print(f"Error: Plan file is not a list: {plan_file}")
                error_message = f"Error: Plan file is not a list: {plan_file}"
                save_error_file(task_dir, error_message)
                continue
            # Replace the image placeholders in the plan with the actual image content
            print(Dict_for_plan)
            plan = replace_image_placeholders_in_plan(plan, Dict_for_plan)
            save_plan_json(plan, plan_file)
            # Save Dict_for_plan to a file
            with open(f"{task_dir}/Dict_for_plan.json", 'w') as f:
                json.dump(Dict_for_plan, f, indent=4)
        
        Execute_plan(plan,task, task_dir)
        
        
        cnt = 0
        while os.path.exists(error_file):
            cnt += 1
            if cnt > 2:
                with open(f"{task_dir}/skip", 'w') as f:
                    f.write("skip")
                break
            with open(f"{task_dir}/error.log", 'r') as f:
                error_message = f.read()
            os.remove(f"{task_dir}/error.log")
            Query = task["Query"]
            task_content = ""
            for seg in Query:
                if seg['type'] == "text":
                    task_content += seg['content']
            structure = f"{extract_structure(task)['Answer_str']}"
            # print(response_text)
            response_text = double_check(task_dir,task_content,structure,response_text,error_message)
            try:
                extract_plan_from_response(response_text, plan_file)
            except Exception as e:
                print(f"Error extracting plan: {str(e)}")
                with open(f"{task_dir}/error.log", 'a') as f:
                    f.write(f"Error extracting plan: {str(e)}")
                    break
            plan = load_input_json(plan_file)
            if not isinstance(plan, list):
                print(f"Error: Plan file is not a list: {plan_file}")
                error_message = f"Error: Plan file is not a list: {plan_file}"
                save_error_file(task_dir, error_message)
                continue
            # Replace the image placeholders in the plan with the actual image content
            Dict_for_plan = load_input_json(f"{task_dir}/Dict_for_plan.json")
            plan = replace_image_placeholders_in_plan(plan, Dict_for_plan)
            save_plan_json(plan, plan_file)
            Execute_plan(plan,task, task_dir)
            
            
            


if __name__ == "__main__":
    main()
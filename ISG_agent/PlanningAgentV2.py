import os
import json
from argparse import ArgumentParser
from Prompt.NewUnifyPrompt import PLANNING_PROMPT
import base64
from tqdm import tqdm
import re
from ToolAgent import tool_agent
from PIL import Image, ImageFile
from openai import OpenAI
from anthropic import  Anthropic


OpenAIClient = OpenAI()
ClaudeClient = Anthropic()

benchmark_file = "path to benchmark json"

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
                image_path = seg['content']
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
                image_path = seg['content']
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
        
        content.append({
            "type": "text",
            "text": f"###Structure Requirement:{extract_structure(task)['Answer_str']}\n\n"
        })
        print(f"Structure: {extract_structure(task)['Answer_str']}")
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

Please reconstruct the plan according to the following instructions:

1. The plan should be in the json format: [{{"Task":"", ...}}, {{"Task":"", ...}}, ...].


2. Misuse of Task Name:
    - Valid Task Names: "Call_tool", "AddImage", "Caption".
    - All the misused Task Names are "Call_tool" steps, which the directly use a tool name to describe the task. You should replace the misused Task Name with "Call_tool", keep the rest of the content unchanged. 


3. Apply the following restrictions:
   - If there are more than one "Call_tool" steps in one plan, none of them should refer to Video Generation or 3D Generation or Image Morph. You have to decide whether to use ImageGeneration or ImageEdit tool based on each step's Input_text and append "Use ImageGeneration tool:" or "Use ImageEdit tool:" at the beginning of the Input_text.
   - If there are many Call_tool steps but some of them seem like they call VideoGeneration or 3D Generation or ImageMorph, you have to reconstruct the whole plan, decide whether to use one step of VideoGeneration, 3D Generation or Image Morphor or use multiple steps of ImageGeneration and ImageEdit tools. You also have to directly append "Use {{Toolname}} tool:" at the beginning of the Input_text.

Remember:
- "generate an image", "generate one image" means to use ImageGeneration tool.
- "style transfer", "modify object", "add/remove attribute", "transform ..." refers to use ImageEdit.
- "Create a sequence/series of continuous images" refers to use VideoGeneration.
- "Create multiple views of images" refers to use 3D Generation.
- "Morph between two images" refers to use ImageMorph.
- ImageGeneration's Input_images list should be empty (no input image).
- ImageEdit's Input_images list should contain only one image.
- VideoGeneration/3D Generation's Input_images list should contain only one image.
- ImageMorph's Input_images list should contain two images.
- If there is any conflict of restrictions, you should reconstruct the whole plan.

3. Placeholder Problem:
- For the original input image, use #image{{ID}}# with ID starting from 1 as a placeholder.
- For the generated input image, use <GEN_{{ID}}> with ID starting from 0 as a placeholder.

Please provide the corrected plan in required JSON format.
"""
    try:
        response = OpenAIClient.chat.completions.create(
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
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Azure API: {str(e)}")
        print("Switch to Claude API")
        try:
            response = ClaudeClient.messages.create(
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
            response_text = response.content[0].text
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
    except json.JSONDecodeError:
        raise ValueError("No valid JSON found in the response")


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
                "type": "image",
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


def extract_structure(benchmark):

    golden_list = []
    text_i = 1
    img_i = 1
    for item in benchmark["Golden"]:
        if item["type"] == "text":
            golden_list.append(f"<gen_text{text_i}>")
            text_i += 1
        else:
            golden_list.append(f"<gen_img{img_i}>")
            img_i += 1

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
                        match = re.match(r'<GEN_(\d+)>', img)
                        if match:
                            id = int(match.group(1))
                            if id < len(generated_image_list):
                                processed_input_images.append(generated_image_list[id])
                                match_gen = re.match(r'<GEN_(\d+)>', generated_image_list[id])
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
                    response = {'text': '', 'images': []} 
                step['Output'] = []
                images = response.get('images', [])
                if len(images) >= 2:
                    VDGen += 1
                    if VDGen > 1 or Call_tool_times > 1:
                        error_message = f"Error: Too many Video Generation or 3D Generation tasks in task {task.get('id', '0000')}"
                        print(error_message)
                        save_error_file(task_dir, error_message)
                        break
                if len(images) == 0:
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
            processed_input_images = []
            for img in input_images:
                if img is None:
                    print(f"Warning: None image encountered in step {i+1}.")
                    processed_input_images.append(None)
                    continue
                if '<GEN_' in img:
                    # Extract the id from the placeholder
                    match = re.match(r'<GEN_(\d+)>', img)
                    if match:
                        id = int(match.group(1))
                        if id < len(generated_image_list):
                            processed_input_images.append(generated_image_list[id])
                            match_gen = re.match(r'<GEN_(\d+)>', generated_image_list[id])
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
            
            if Task == "AddImage":
                for img in input_images:
                    result += f"<boi>{img}<eoi>"
                continue
            input_text = step.get('Input_text', '')
            if Task == "Caption":
                print(step['Input_images'])
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
                response = OpenAIClient.chat.completions.create(
                    model='gpt-4o',
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.5
                )
                response_text = response.choices[0].message.content
                try:
                    extract_plan_from_response(response_text, plan_file)
                except Exception as e:
                    print(f"Error extracting plan: {str(e)}")
                    print(response)
                    raise ValueError("Error extracting plan")
            except Exception as e:
                print(f"Error in Azure API, switch to Claude API: {str(e)}") 
                print("Switch to Claude API")
                # print("Error in Azure API, switch to Claude API")
                messages,Dict,Dict_for_plan = preprocess_task(task, task_dir, plan_model="claude")
            
                response = ClaudeClient.messages.create(
                    model = "claude-3-5-sonnet-20240620",
                    max_tokens=8192,
                    system=PLANNING_PROMPT,
                    messages=messages,
                    temperature=0.7,
                )
                response_text = response.content[0].text
                try:
                    extract_plan_from_response(response.content[0].text, plan_file)
                except Exception as e:
                    print(f"Error extracting plan: {str(e)}")
                    print(response)
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
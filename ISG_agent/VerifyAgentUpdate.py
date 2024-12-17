import os
import json
import re
from collections import defaultdict
from openai import OpenAI
from anthropic import Anthropic
from ToolAgent import tool_agent
from PlanningAgentV2 import load_input_json, extract_plan_from_response,handle_caption_step,save_error_file,decode_result_into_jsonl,save_result_json,save_plan_json
from typing import List, Dict, Tuple, Union
from PIL import Image
import base64
import sys
from time import sleep
import shutil
import dotenv

dotenv.load_dotenv()

benchmark_file = "../ISG_eval/ISG-Bench.jsonl" # <path to benchmark.jsonl>

OpenAIClient = OpenAI(
   api_key=os.getenv("OPENAI_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)
ClaudeClient = OpenAI(
   api_key=os.getenv("OPENAI_API_KEY"), # KEY
   base_url=os.getenv("OPENAI_BASE_URL")
)

def load_task_prompt(original_tasks_file: str, task_id: str) -> Dict:
    """
    Load the original task prompt from a JSON file based on the task ID.
    Since tasks are sequential, directly access the task by index.
    """
    with open(original_tasks_file, 'r') as f:
        all_tasks = json.load(f)
    
    task_index = int(task_id)  # Convert task_id to an integer for indexing
    
    # Access the task at the specific index
    if 0 <= task_index < len(all_tasks):
        Query = all_tasks[task_index]["Query"]
        query_text = ""
        for q in Query:
            if q["type"] == "text":
                query_text += q["content"] + "\n"
        return query_text
    else:
        print(f"Task ID {task_id} is out of bounds.")
        return None


def smooth_result(task_dir, task_id, original_tasks_file, result_json_path):
    print(f"Task {task_id} is successfully verified.")
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
    print("Start to smooth the result")

    # Paths
    smooth_path = os.path.join(task_dir, "smooth_result.json")
    wrong_result_path = os.path.join(task_dir, "Wrong_result.json")

    # Load the result JSON
    with open(result_json_path, 'r') as f:
        result_json = json.load(f)

    # Extract the 'output' field
    output_segments = result_json.get('output', [])

    # Build the combined text with image placeholders including image paths
    combined_text_with_paths = ""
    image_placeholders = []
    for segment in output_segments:
        if segment['type'] == 'text':
            combined_text_with_paths += segment['content'] + "\n\n"
        elif segment['type'] == 'image':
            image_path = segment['content']
            placeholder = f"<boi>{image_path}<eoi>"
            combined_text_with_paths += placeholder + "\n\n"
            image_placeholders.append(placeholder)

    # For the language model, replace placeholders with <boi><eoi>
    combined_text_for_lm = combined_text_with_paths
    for placeholder in image_placeholders:
        combined_text_for_lm = combined_text_for_lm.replace(placeholder, "<boi><eoi>")

    # Load the task description
    task = load_task_prompt(original_tasks_file, task_id)

    # System prompt for the assistant
    sys_p = """
### Task Instructions for Refining Text Segments in Multimodal Content

1. **Sequence and Placeholder Preservation**:
   - Maintain the exact sequence and number of text segments and image placeholders (`<boi><eoi>`) as in the input. No new segments should be added between existing placeholders.

2. **Text Flow Improvements**:
   - Rephrase text segments for improved fluency and coherence. Remove redundancy and ensure smooth transitions between segments.

3. **Image-Text Consistency**:
   - Integrate references to the images naturally within the text without direct statements about the images. Describe or hint at image content.

4. **Error Handling**:
   - If any references to missing or problematic images occur, remove apologies or explanations to keep the narrative smooth.

### Key Instructions:
1. **Strict Preservation**: Output must match input in sequence and number of placeholders.
2. **Fluency Focus**: Ensure coherent and engaging text.
3. **Refinement**: Eliminate redundancy and fix fragmented sentences.

### Few-Shot Examples:

#### Example 1 (Correct):
**Input**:
- The sun rises over the mountains. <boi><eoi> A small cabin rests peacefully in the valley. <boi><eoi> Birds fly through the clear sky.

**Output**: (Correct position and number of placeholders)
- As the sun ascends over the mountains, its golden light bathes the landscape. <boi><eoi> A cozy cabin nestles in the serene valley below. <boi><eoi> Birds glide effortlessly across the bright blue sky.

#### Example 2 (Correct with Repeated Placeholders):
**Input**:
- The forest is quiet. <boi><eoi> <boi><eoi> <boi><eoi> A gentle wind moves through the trees.

**Output**:
- The forest remains still and peaceful. <boi><eoi> <boi><eoi> <boi><eoi> A soft breeze whispers through the leaves.

#### Example 3 (Incorrect):
**Input**:
- The dessert table is full of treats. <boi><eoi> <boi><eoi> <boi><eoi>

**Output** (Incorrect position of placeholders, additional text involved):
- The dessert table overflows with delicious treats. <boi><eoi> The man begins organizing his gear. <boi><eoi> The table is fulfilled <boi><eoi>
**Error**: New text was introduced between the image placeholders.
"""

    # Prepare messages for the assistant
    message_smooth = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": f"###Task Description:\n{task}\n###Result:\n{combined_text_for_lm}"}
    ]

    # Check if the smoothed result already exists
    if os.path.exists(smooth_path):
        print(f"Smoothed result already exists for task: {task_dir}")
    else:
        for attempt in range(3):
            try:
                # Initialize smoothed_text to None
                smoothed_text = None

                # Call the language model (replace with your actual API call)
                response = OpenAIClient.chat.completions.create(
                    model='gpt-4o',  # Replace with your model name
                    messages=message_smooth,
                    max_tokens=4096,
                    temperature=0,
                )

                # Extract the smoothed text from the response
                smoothed_text = response.choices[0].message.content.strip()
                def extract_smoothed_text(smoothed_text):
                    # if there is 'Result:', extract the text after 'Result:'
                    if 'Result:' in smoothed_text:
                        smoothed_text = smoothed_text.split('Result:')[1].strip()
                    return smoothed_text
                smoothed_text = extract_smoothed_text(smoothed_text)
                # Verify that the smoothed text has the correct number of placeholders
                num_placeholders_original = combined_text_for_lm.count("<boi><eoi>")
                num_placeholders_smoothed = smoothed_text.count("<boi><eoi>")

                if num_placeholders_original != num_placeholders_smoothed:
                    raise Exception("The number of image placeholders in the smoothed text does not match the original.")

                # Split the smoothed text back into segments
                smoothed_parts = re.split(r'(<boi><eoi>)', smoothed_text)

                # Remove empty strings resulting from splitting
                smoothed_parts = [part for part in smoothed_parts if part.strip() != '']

                # Extract the sequence of types from the smoothed parts
                smoothed_sequence = []
                for part in smoothed_parts:
                    if part.strip() == "<boi><eoi>":
                        smoothed_sequence.append('image')
                    else:
                        smoothed_sequence.append('text')

                # Extract the sequence of types from the original output segments
                original_sequence = []
                for segment in output_segments:
                    original_sequence.append(segment['type'])

                # Compare the sequences
                if smoothed_sequence != original_sequence:
                    print("The sequence of segments in the smoothed text does not match the original.")
                    print("Ending smoothing and writing NoChange file.")
                    # Fallback: Copy the original result to smooth_path
                    shutil.copyfile(result_json_path, smooth_path)
                    with open(os.path.join(task_dir, "NoChange"), "w") as f:
                        f.write("NoChange")
                    # Save the wrong output
                    with open(wrong_result_path, 'w') as f:
                        f.write(smoothed_text)
                    raise Exception("sequence of segments mismatch")
                else:
                    # Proceed to reconstruct the smoothed output
                    smoothed_segments = []
                    image_placeholder_iter = iter(image_placeholders)

                    for part in smoothed_parts:
                        part = part.strip()
                        if part == "<boi><eoi>":
                            # Get the next image placeholder with the image path
                            image_placeholder = next(image_placeholder_iter, None)
                            if image_placeholder is None:
                                raise Exception("More image placeholders in smoothed text than in original.")
                            # Extract the image path from the placeholder
                            match = re.match(r'<boi>(.*?)<eoi>', image_placeholder)
                            if match:
                                image_path = match.group(1)
                                smoothed_segments.append({
                                    "type": "image",
                                    "caption": "",
                                    "content": image_path
                                })
                            else:
                                raise Exception("Invalid image placeholder format.")
                        elif part:
                            # It's a text segment
                            smoothed_segments.append({
                                "type": "text",
                                "content": part
                            })

                    # Ensure we have processed all image placeholders
                    if next(image_placeholder_iter, None) is not None:
                        raise Exception("Not all image placeholders were used in smoothed text.")

                    # Save the smoothed result
                    result_json['output'] = smoothed_segments
                    with open(smooth_path, 'w') as f:
                        json.dump(result_json, f, indent=4)

                    print(f"Smoothing successful for task: {task_dir}")
                    break
            except Exception as e:
                print(f"Error during result smoothing: {e}")
                # If the error is due to mismatched sequences, we have already broken out of the loop
                if "sequence of segments" in str(e) or attempt == 2:
                    print("Failed to smooth the result.")
                    # Fallback: Copy the original result to smooth_path
                    shutil.copyfile(result_json_path, smooth_path)
                    with open(os.path.join(task_dir, "NoChange"), "w") as f:
                        f.write("NoChange")
                    # Save the wrong output if it exists
                    if smoothed_text is not None:
                        with open(wrong_result_path, 'w') as f:
                            f.write(smoothed_text)
                    break
                else:
                    print(f"Retrying result smoothing (attempt {attempt+1}/3)...")
                    sleep(2 ** attempt)


from tqdm import tqdm

def main(input_dir: str, original_tasks_file: str):
    # List all directories under input_dir (assuming task directories follow sequential names like 0000, 0001, etc.)
    task_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for index, task_dir_name in tqdm(enumerate(task_dirs)):
        # Generate the task ID based on the directory name (assumed to be in the form '0000', '0001', etc.)
        count = 0
        task_id = task_dir_name[-4:]

        task_dir = os.path.join(input_dir, task_dir_name)
        plan_json_path = os.path.join(task_dir, f"plan_{task_id}.json")
        final_plan_json_path = os.path.join(task_dir, f"plan_{task_id}_final.json")
        result_json_path = os.path.join(task_dir, f"result.json")
        error_log_path = os.path.join(task_dir, "error.log")
                    
            
        if os.path.exists(error_log_path):
            print("SmoothMode")
            continue
        if os.path.exists(f"{task_dir}/skip"):
            continue
        if not os.path.exists(result_json_path):
            print(f"Result missing for task: {task_dir}")
            continue
        if not os.path.exists(final_plan_json_path):
            print(f"Final plan missing for task: {task_dir}")
            continue
        print(f"Task {task_id} is successfully verified.")
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")
        print("Start to smooth the result")
        # In this part, we will smooth the result's text segments, make it interleaved with image, make the final result more fluent.
        smooth_result(task_dir, task_id, original_tasks_file, result_json_path)
            
                
                

if __name__ == "__main__":
    # Ensure input_dir and original_tasks_file are passed as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python VerifyAgentUpdate.py <input_dir> <original_tasks_file>")
        sys.exit(1)

    input_dir = sys.argv[1]
    original_tasks_file = sys.argv[2]
    
    main(input_dir, original_tasks_file)
    
    
    


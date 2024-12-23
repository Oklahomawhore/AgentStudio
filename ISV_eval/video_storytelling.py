import openai
import os
import json
import random
from PIL import Image
from collections import defaultdict
from datasets import load_dataset
import dotenv
import base64
from openai import OpenAI
from typing import List
from tqdm import tqdm

dotenv.load_dotenv()

VIDEO_ROOT = "VideoStoryTelling"

OpenAIClient = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# A sample dataset (replace with your actual data)

ds = load_dataset("flwrlabs/ucf101")["train"]

# Function to get image URL or path
def get_image_path(image_obj, idx, video_id):
    # Save image to file and return path
    image_path = f"images/{video_id}_{idx:04d}.jpg"
    if not os.path.exists(os.path.join(VIDEO_ROOT,"images")):
        os.makedirs(os.path.join(VIDEO_ROOT,"images"))
    image_obj.save(os.path.join(VIDEO_ROOT,image_path))
    return image_path

# Function to call GPT API to generate a description
def generate_description(frames):
    # Combine frames to create a prompt
    prompt = """
    I will give you a sequence of images from a video showing a person engaged in some activities. 
    Please generate a detailed story based on these frames, USE YOUR IMAGINATION to create a background 
    stroy about the why the how and the what (make your ideas go wild but PROVIDE *SAFE* CONTENT only). 
    NOTE:the story generated should be closed related to each frame to be used as voiceovers. 
    Response in a json string format.
    
    Considerations:
    OUTPUT PLACEHOLDER: use #image{ID}# as key for each trunk of your story, ID starts from 1.

    Example Output:

    {
        "#image1#" : "A little dog tries to get into the door, but fails as the was closed",
        "#image2#" : "The master of the dog comes after hearing the noice, walking towards the door",
        "#image3#" : "The master opens the door for the dog, dog comes in",
        ...
    }
     
    """
    
    images_paths = [get_image_path(frame["image"], frame["frame"], frame["video_id"]) for idx, frame in enumerate(frames)]

    content = []
    content.append({
        "type" : "text",
        "text"  : prompt
    })
    for image in images_paths:
        # Encode the image to Base64
        with open(os.path.join(VIDEO_ROOT,image), "rb") as f:
            image_data = f.read()
            # Encode the image to Base64
            img_base64 = base64.b64encode(image_data).decode("utf-8")

            # Append the image data to the content list
            content.append({
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image;jpeg;base64,{img_base64}"
                },
            })
    messages = [
        {
            "role" : "user",
            "content" : content
        }
    ]
    try:
        # Call OpenAI API to generate the description
        response = OpenAIClient.chat.completions.create(
            model="gpt-4o-mini",  # You can also use gpt-3.5-turbo depending on your requirement
            messages=messages,
            max_tokens=8192,
            temperature=0.7
        )
    except openai.BadRequestError as e:
        print(f"exception {e.code} caught in openai api calling, retrying!")
        if e.code == "400":
            # Call OpenAI API to generate the description
            messages.append({
                "role" : "assistant",
                "refusal" : e.message
            })
            messages.append({
                "role" : "user",
                "content" : "Improve the original prompt and execute accordingly, DO return final json format only."
            })
            response = OpenAIClient.chat.completions.create(
                model="gpt-4o-mini",  # You can also use gpt-3.5-turbo depending on your requirement
                messages=messages,
                max_tokens=8192,
                temperature=0.7
            )
    except Exception as e:
        print(e)
        return "#<ANSWER_NULL>#" , images_paths
    story_text = response.choices[0].message.content
    print(f"{frames[0]['video_id']} : {story_text}")
    return story_text, images_paths

# Create the dataset for video storytelling
def create_video_storytelling_entry(idx, frames):
    story_text, images = generate_description(frames)
    
    entry = {
        "id": frames[0]["video_id"],
        "Category": "Video Storytelling",
        "Query": [
            {
                "type": "text",
                "content": f"I will give you a picture of a person in a scenario. Generate a video according to the images and a story according it, make the story go crazy."
            },
            {
                "type": "image",
                "content": images[0]  # First image
            }
        ],
        "Golden": [
            {
                "type": "text",
                "content": story_text
            },
            {
                "type": "image",
                "content": images[1]  # Second image
            },
            {
                "type": "image",
                "content": images[2]  # Third image
            },
            {
                "type": "image",
                "content": images[3]  # Fourth image
            },
            {
                "type": "image",
                "content": images[4]  # Fifth image
            },
            {
                "type" : "video",
                "content" : frames[0]["video_id"]
            }
        ],
        "predict": {
            "input": f"I will give you a picture of a person in a scenario. <image1> Please use a combination of 4 images and text to show what will happen next. Please generate an overall description first, then directly generate adjacent image blocks.\n (the caption of the <query_img1> is: \"{story_text}\") ",
            "structural": {
                "Query": [
                    f"<query_text1>",
                    f"<query_img1>"
                ],
                "Answer": [
                    f"<gen_text1>",
                    f"<gen_img1>",
                    f"<gen_img2>",
                    f"<gen_img3>",
                    f"<gen_img4>"
                ]
            },
            "block_tuple": {
                "Thought": "Analyze the given prompt to understand the relationships between the elements mentioned, focusing on the sequence and the description provided.",
                "relation": [
                    [
                        "<gen_text1>",
                        "<query_img1>",
                        "describes the overall action of"
                    ],
                    [
                        "<gen_img1>",
                        "<query_img1>",
                        "is the starting point of the sequence"
                    ],
                    [
                        "<gen_img2>",
                        "<gen_img1>",
                        "is the next image in the sequence"
                    ],
                    [
                        "<gen_img3>",
                        "<gen_img2>",
                        "is semantically consistent to the next step"
                    ],
                    [
                        "<gen_img4>",
                        "<gen_img3>",
                        "is the final image in the sequence"
                    ]
                ]
            }
        }
    }

    return entry

# Process the dataset and create video storytelling entries
def process_dataset(dataset):
    curated_entries = []
    idx = 0

    if os.path.exists(os.path.join(VIDEO_ROOT, "video_storytelling.json")):
        with open(os.path.join(VIDEO_ROOT, "video_storytelling.json"), "r") as f:
            curated_entries = json.load(f)

    # For each unique video/clip, generate a video storytelling entry
    video_id = dataset[0]["video_id"]
    video_frames = []
    video_frames.append(dataset[0])
    for index, data in tqdm(enumerate(dataset), total=len(dataset)):
        if index == 0:
            continue

        # print(data)
        if data["video_id"] == video_id:
            video_frames.append(data)
            video_id = data["video_id"]
        else:
            def select_equally_spaced_elements(lst, n):
                if len(lst) <= n:
                    return lst  # If the list is smaller than or equal to n, just return the whole list
                step = (len(lst) - 1) // (n - 1)  # Adjusted for better spacing
                return [lst[i * step] for i in range(n)]

            
            selected_elements = select_equally_spaced_elements(video_frames, 5)
            # print(selected_elements)

            # Create the entry
            if idx > len(curated_entries) - 1:
                # print("Exceeds previoius record, creating new!")
                
                entry = create_video_storytelling_entry(idx, selected_elements)
                curated_entries.append(entry)
                save_to_json(curated_entries, filename=os.path.join(VIDEO_ROOT, "video_storytelling.json"))
            
            idx += 1
            video_frames.clear()
            video_id = data["video_id"]

    return curated_entries

# Save the curated dataset to a JSON file
def save_to_json(data, filename="curated_video_storytelling_dataset.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Dataset saved to {filename}")

# Main processing
if __name__ == "__main__":
    curated_data = process_dataset(ds)  # Replace dataset with your actual data
    save_to_json(curated_data,filename=os.path.join(VIDEO_ROOT, "video_storytelling.json"))
from datasets import load_dataset
import json
import os

def curate_text_to_video_dataset(dataset_name, output_file, max_entries=10):
    # Load the dataset
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    curated_data = []

    for idx, entry in enumerate(dataset['train'][:max_entries]):
        # Extract the text content
        story_content = entry.get('text', '')
        
        # Generate a unique ID
        entry_id = f"{idx:04d}"
        
        # Construct a sample entry
        curated_entry = {
            "id": entry_id,
            "Category": "Video Storytelling",
            "Query": [
                {
                    "type": "text",
                    "content": "I will give you a picture of a person in a scenario. Generate a video according to the images and a story according it, make the story go crazy."
                },
                {
                    "type": "image",
                    "content": f"images/v_SampleImage_{entry_id}.jpg"
                }
            ],
            "Golden": [
                {
                    "type": "text",
                    "content": f"```json\n{{\n    \"#image1#\": \"{story_content[:100]}...\",\n    \"#image2#\": \"Continues with the story...\",\n    \"#image3#\": \"More storytelling here...\"\n}}\n```"
                },
                {
                    "type": "video",
                    "content": f"v_SampleVideo_{entry_id}"
                }
            ],
            "predict": {
                "input": "I will give you a picture of a person in a scenario. <image1> Please use a combination of 4 images and text to show what will happen next. Please generate an overall description first, then directly generate adjacent image blocks.",
                "structural": {
                    "Query": ["<query_text1>", "<query_img1>"],
                    "Answer": [
                        "<gen_text1>",
                        "<gen_img1>",
                        "<gen_img2>",
                        "<gen_img3>",
                        "<gen_img4>"
                    ]
                },
                "block_tuple": {
                    "Thought": "Analyze the given prompt to understand the relationships between the elements mentioned, focusing on the sequence and the description provided.",
                    "relation": [
                        ["<gen_text1>", "<query_img1>", "describes the overall action of"],
                        ["<gen_img1>", "<query_img1>", "is the starting point of the sequence"],
                        ["<gen_img2>", "<gen_img1>", "is the next image in the sequence"],
                        ["<gen_img3>", "<gen_img2>", "is semantically consistent to the next step"],
                        ["<gen_img4>", "<gen_img3>", "is the final image in the sequence"]
                    ]
                }
            }
        }
        curated_data.append(curated_entry)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(curated_data, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to {output_file}")

# Usage
curate_text_to_video_dataset(
    dataset_name="/data/wangshu/wangshu_code/ISG/ISV_eval/NovelConditionedVGen/Chinese-H-Novels",
    output_file="NovelConditionedVGen/novel_t2v.json",
    max_entries=10  # Adjust the number of entries as needed
)
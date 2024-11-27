_TEXT_TEXT_JUDGE_PROMPT = """
You are given two texts and a question. Please judge whether the question is a correct within the two texts.

Judge Requirement:
{requirement}

Output Requirement:
Please output in JSON format, directly output your judgment in key "Judge" and your reason in key "Reason". Do not write an introduction or summary. Do not output other irrelevant information.

Here is the input:
Text 1:
[Start of Text 1]
{text1}
[End of Text 1]
Text 2:
[Start of Text 2]
{text2}
[End of Text 2]
Question: 
[Start of Question]
{question}
[End of Question]

Now please judge the question.
"""

_TEXT_IMAGE_JUDGE_PROMPT = """
You are given a text, an image and a question between them. Please judge whether the question is a correct.

Judge Requirement:
{requirement}

Output Requirement:
Please output in JSON format, directly output your judgment in key "Judge" and your reason in key "Reason". Do not write an introduction or summary. Do not output other irrelevant information.

Here is the input:
Text: [Start of Text] {text} [End of Text]
Image: Please see the image.
Question: [Start of Question] {question} [End of Question]

Now please judge the question.
"""

_IMAGE_IMAGE_JUDGE_PROMPT = """
You are given two images and a question between them. Please judge whether the question is a correct.

Judge Requirement:
{requirement}

Output Requirement:
Please output in JSON format, directly output a key "Judge" and your reason in key "Reason". Do not write an introduction or summary. Do not output other irrelevant information.

Here is the Question: [Start of Question] {question} [End of Question]

Now please judge the question.
"""

# Evaluate the whole answer without ground truth
_EVALUATE_WHOLE_WO_GT_PROMPT = ["""
You are a helpful and impartial assistant. You are given a multimodal query with one or several images and an multimodal answer with interleaved text and images. Please judge whether the answer is correct and relevant to the query in several dimensions.

Judge Requirement: Evaluate the answer based on the following dimensions:
1. Coherence: How well the text and images work together to convey a unified message or story.
2. Content Accuracy: The factual correctness of both textual information and visual elements.
3. Relevance and Responsiveness: How well the generated content addresses the given query.
4. Visual-Textual Alignment: The degree to which generated images match and support the accompanying text.
5. Creativity and Originality: The model's ability to generate novel and imaginative content across both text and images.

Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

{{
  "coherence": {{
    "score": 0,
    "explanation": ""
  }},
  "content_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "relevance_and_responsiveness": {{
    "score": 0,
    "explanation": ""
  }},
  "visual_textual_alignment": {{
    "score": 0,
    "explanation": ""
  }},
  "creativity_and_originality": {{
    "score": 0,
    "explanation": ""
  }},
  "overall_score": 0,
  "final_thoughts": ""
}}

Here is the Query: 
""",
"""
Here is the Answer: 
""",
"""
Now please judge the answer. Remember to output in JSON format with scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score.
"""]


_DSG_PROMPT = """You are a helpful assistant capable of analyzing images and answering questions about them. Your task is to examine the provided image and answer the given question.

## Input
- An image
- A question about the image (e.g., "Is a dog in this image?" or "Is the dog blue in this image?")

## Output
Provide your response in JSON format with the following structure:
{{
  "Judge": "Yes" or "No",
  "Reason": "Your explanation here"
}}

## Instructions
1. Analyze the provided image.
2. Consider the question asked about the image.
3. Determine whether the answer to the question is "Yes" or "No".
4. Provide a brief but clear explanation for your judgment.
5. Format your response in the required JSON structure.

Here is the question:
{question}
Now please judge the question. Remember to output in JSON format with "Judge" and "Reason".
"""


_EVALUATE_WHOLE_W_GT_PROMPT = ["""
You are a helpful and impartial assistant. You are given a multimodal query with one or several images and a multimodal answer with interleaved text and images. I will also provide a golden answer to the query. Please judge whether the answer is correct and relevant to the query in several dimensions.

Judge Requirement: Evaluate the answer based on the following dimensions:
1. Coherence: How well the text and images work together to convey a unified message or story.
2. Content Accuracy: The factual correctness of both textual information and visual elements.
3. Relevance and Responsiveness: How well the generated content addresses the given query.
4. Visual-Textual Alignment: The degree to which generated images match and support the accompanying text.
5. Creativity and Originality: The model's ability to generate novel and imaginative content across both text and images.

Output Requirement: Please output in JSON format, including scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score. The JSON should follow this structure:

{{
  "coherence": {{
    "score": 0,
    "explanation": ""
  }},
  "content_accuracy": {{
    "score": 0,
    "explanation": ""
  }},
  "relevance_and_responsiveness": {{
    "score": 0,
    "explanation": ""
  }},
  "visual_textual_alignment": {{
    "score": 0,
    "explanation": ""
  }},
  "creativity_and_originality": {{
    "score": 0,
    "explanation": ""
  }},
  "overall_score": 0,
  "final_thoughts": ""
}}

Here is the Query: 
""",
"""
Here is the Answer: 
""",
"""
Here is the Golden Answer: 
""",
"""
Now please judge the answer. Remember to output in JSON format with scores for each dimension (on a scale of 1-10) and a final overall score (on a scale of 1-10). Also provide brief explanations for each score.
"""]


VQA_dict = {
    "Yes_No": "You should output directly 'Yes' or 'No' and your reason. 'Yes' means the question is correct corresponding to the given text and image. 'No' means the question is not correct corresponding to the given text and image.",
    "Score": "You should output a score on a scale of 1-10 and your reason. The score should be a numerical value that reflects how well the question is answered by the given text and image. 10 means the question is answered perfectly. 1 means the question is not answered at all."
}

def get_judge_prompt():
    return {
        "Text-Text": _TEXT_TEXT_JUDGE_PROMPT,
        "Text-Image": _TEXT_IMAGE_JUDGE_PROMPT,
        "Image-Image": _IMAGE_IMAGE_JUDGE_PROMPT,
        "WO_GT": _EVALUATE_WHOLE_WO_GT_PROMPT,
        "W_GT": _EVALUATE_WHOLE_W_GT_PROMPT,
        "Requirement": VQA_dict,
        "DSG": _DSG_PROMPT
    }
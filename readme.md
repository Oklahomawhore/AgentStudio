<div align="center">
<h1>Interleaved Scene Graph for Interleaved Text-and-Image Generation Assessment</h1>

[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=for-the-badge&logoWidth=40)](https://interleaved-eval.github.io/TrustLLM-Website/)
[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=for-the-badge&logoWidth=40)](https://arxiv.org/abs/2402.04788)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=for-the-badge&logoWidth=40)](https://huggingface.co/datasets/shuaishuaicdp/ISG)


<img src="https://img.shields.io/github/last-commit/Dongping-Chen/ISG?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/Dongping-Chen/ISG?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/Dongping-Chen/ISG?style=flat-square&color=5D6D7E" alt="GitHub top language" />

<img src="figures/evaluation.png">
<p align="center">

</p>
</div>

## Updates & News
- [27/11/2024] :page_facing_up: We release our paper on [Arxiv](http://arxiv.org/abs/2402.04788) and our [Dataset](http://huggingface//) today!
  
## Contents
- [Updates \& News](#updates--news)
- [Contents](#contents)
- [Evaluation Method: Interleaved Scene Graph (ISG)](#interleaved-scene-graph)
- [Evaluating Your Own Model](#evaluating-your-own-model)
- [Agent: ISG-Agent](#isg-agent)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

## Interleaved Scene Graph

<img src="figures/benchmark.png">

This evaluation method and benchmark is designed for evaluating interleaved generation in four levels: Structural, Block, Image, and Holistic. It is an well established testbed for model can perform both multimodal understanding and generation such as [Show-o](https://github.com/showlab/Show-o) and [Anole](https://github.com/GAIR-NLP/anole).

```markdown
/ISG_eval
├── images (You should download it from huggingface and place here)
├── ISG-Bench.jsonl
├── ...
```

1. **images**: Contains images in queries and golden answer. You can download it from [here](https://) and place them under ISG_eval.

2. **ISG-Bench.jsonl**: Contains ground truth compiled previously by ISG. One data sample is as follows. It contains `Query` for question and `Golden` for human-annotated golden answer.

```json
{
    "id": "0000",
    "Category": "Prediction",
    "Query": [
        {
            "type": "text",
            "content": "I will give you a picture of a person washing their hands. Please use a combination of 4 images and text to show what will happen next. Please generate an overall description first, then directly generate adjacent image blocks. For example, [whole description] <object1 image> <object2 image> <object3 image> <object4 image>."
        },
        {
            "type": "image",
            "content": "images/0000_q1.jpg"
        }
    ],
    "Golden": [
        {
            "type": "text",
            "content": "The person continues to scrub their hands thoroughly, with the soap lathering up. The hands are cleaned under running water, and the lather is rinsed away."
        },
        {
            "type": "image",
            "content": "images/0000_g1.jpg"
        },
        {
            "type": "image",
            "content": "images/0000_g2.jpg"
        },
        {
            "type": "image",
            "content": "images/0000_g3.jpg"
        },
        {
            "type": "image",
            "content": "images/0000_g4.jpg"
        }
    ],
    "predict": {
        "structural": {
            "Query": [
                "<query_text1>",
                "<query_img1>"
            ],
            "Answer": [
                "<gen_text1>",
                "<gen_img1>",
                "<gen_img2>",
                "<gen_img3>",
                "<gen_img4>"
            ]
        },
        "block_tuple": {
            "relation": [
                [
                    "<gen_text1>",
                    "<query_img1>",
                    "is an overall description of"
                ],
                ...
            ]
        },
        "block_qa": {
            "questions": [
                {
                    "subject": "<gen_text1>",
                    "object": "<query_img1>",
                    "relation": "is an overall description of",
                    "Question": "Does <gen_text1> describe this image?"
                },
                ...
            ]
        },
        "image_tuple": [
            [
                "entity",
                "hands",
                "<gen_img1>"
            ],
            ...
        ],
        "image_qa": {
            "questions": [
                {
                    "image": "<gen_img1>",
                    "Question": "Are there hands in this image?",
                    "id": 0,
                    "Preliminary": []
                },
                ...
            ]
        }
    }
}
```


## Evaluating Your Own Model

<img src="figures/case_study.png">
Once you get your model's output, manage them as a JsonLine file, where each the answer for each id is under key `output`:

```json
{
    "id": "0000",
    "Category": "Prediction",
    "output": [
        {
            "type": "text",
            "content": "<text-content>"
        },
        {
            "type": "image",
            "content": "<path_of_the_input_image>"
        }
    ]
}
```

Then, run the following script:

```shell
python ISG-eval.py --input_file <your file>
python summarize_performance.py --input_file <output of ISG-eval.py>
```

## ISG-Agent: Exploring the Upper Bound for Interleaved Generation

<img src="figures/agent.png">
ISG-Agent is a compositional framework that leverage tools to generate high-quality interleaved content while strictly follows user's query. See `ISG_agent/README.md` for enviroment setup and how to use.
## Acknowledgments

This project is a follow-up of [MLLM-as-a-Judge](https://arxiv.org/pdf/2402.04788). This work is partially funded by Toyota Motor Corporation. We’d also like to extend a thank you to [Jieyu Zhang](https://jieyuz2.github.io/), [Weikai Huang](https://weikaih04.github.io/), and [Zixian Ma](https://zixianma.github.io/) for their insightful feedback and support.

## Citation

```
@misc{chen2024mllmasajudge,
      title={MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark}, 
      author={Dongping Chen and Ruoxi Chen and Shilin Zhang and Yinuo Liu and Yaochen Wang and Huichi Zhou and Qihui Zhang and Pan Zhou and Yao Wan and Lichao Sun},
      year={2024},
      eprint={2402.04788},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

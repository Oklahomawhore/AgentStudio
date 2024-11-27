# Requirements:
## Environments:
- Since there are multiple tools used in this project, we have to prepare three different environments
- For Most of the tools: `pip install -r requirements.txt`
- For Flux: You can refer to Flux repo `https://github.com/black-forest-labs/flux` to download the `diffusers` environment for Flux.
- For DreamMover: You can refer to DreamMover repo `https://github.com/leoShen917/DreamMover` to download the environment for the DreamMover.
## Model Checkpoints:
- You have to refer to different model's repo for checkpoints download.
- Tools are: `Flux.1-dev`, `Ultra-Edit`, `DynamiCrafter`, `SV3D`, `DreamMover`

## Model Setup:
```bash
conda activate TransAgent
python app_service_formost.py

conda activate Flux
python app_service_forflux.py

conda activate mover
python app_service_forMorph.py
```

## VRAM:
- Deploy all the models require at least 90GiB VRAM, you can set device in each model's script.

# Test on ISG
- For ISG-benchmark run `python PlanningAgentV2.py --input_json benchmark_path --outdir output_dir`, you may also need to set benchmark path in the script.
- For Smoothing the output, run `python VerifyAgentUpdate.py benchmark_path output_dir` 


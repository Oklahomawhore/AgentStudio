import os 
import torch
import numpy as np
import imageio
import torchvision
import base64
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'DreamMover'))
from diffusers import DDIMScheduler, AutoencoderKL
from pipeline import MovePipeline
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from utils.predict import predict_z0, splat_flowmax
from diffusers.utils.import_utils import is_xformers_available
from flask import Flask, request, jsonify
from threading import Lock
from io import BytesIO
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
import gc

class ImageMorph:
    def __init__(self, model_path, vae_path, lora_path):
        self.model_path = model_path
        self.vae_path = vae_path
        self.lora_path = lora_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.save_dir = ".cache"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Fixed hyperparameters
        self.guidance_scale = 1.0
        self.n_inference_step = 50
        self.n_actual_inference_step = 30
        self.Time = 3
        self.save_inter = True
        self.feature_inversion = 14
        self.unet_feature_idx = [2]
        


    def _load_model(self):
        # Load the model
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                   beta_schedule="scaled_linear", clip_sample=False,
                                   set_alpha_to_one=False, steps_offset=1)
        model = MovePipeline.from_pretrained(self.model_path, scheduler=scheduler).to(self.device)
        model.modify_unet_forward()
        if self.vae_path != "default":
            model.vae = AutoencoderKL.from_pretrained(
                self.vae_path
            ).to(model.vae.device, model.vae.dtype)

        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise RuntimeError("Xformers not available.")

        # Set LoRA
        weight_name = "lora.ckpt" if os.path.exists(os.path.join(self.lora_path, "lora.ckpt")) else None
        if self.lora_path == "":
            print("Applying default parameters")
            model.unet.set_default_attn_processor()
        else:
            print("Applying LoRA: " + self.lora_path)
            model.unet.load_attn_procs(self.lora_path, weight_name=weight_name)

        return model

    def generate_frames(self, img_path1, img_path2, prompt):
        # Load images
        gc.collect()
        torch.cuda.empty_cache()
        self.model = self._load_model()
        img = [imageio.imread(img_path1), imageio.imread(img_path2)]
        image = np.stack((img[0], img[1]))
        source_image = torch.from_numpy(image).float() / 127.5 - 1
        source_image = source_image.permute(0, 3, 1, 2)
        self.source_image = source_image.to(self.device)
        self.prompt = prompt
        print("source_image shape: ", source_image.shape)
        full_h, full_w = source_image.shape[2:4]
        sup_res_h = int(full_h / 8)
        sup_res_w = int(full_w / 8)
        print("sup_res_h: ", sup_res_h)
        print("sup_res_w: ", sup_res_w)
        # Predict high-level space z_T -> 0
        try:
            flow1to2, flow2to1 = predict_z0(self.model, self, sup_res_h, sup_res_w)
        except Exception as e:
            print(f"Error predicting high-level space z_T -> 0: {e}")
            raise e
        try:
            with torch.no_grad():
                invert_code, pred_x0_list = self.model.invert(self.source_image,
                                                                self.prompt,
                                                                guidance_scale=self.guidance_scale,
                                                                num_inference_steps=self.n_inference_step,
                                                                num_actual_inference_steps=self.n_actual_inference_step,
                                                                return_intermediates=True)
                init_code = invert_code.clone()
                pred_code = pred_x0_list[self.n_actual_inference_step].clone()

                src_mask = torch.ones(2, 1, init_code.shape[2], init_code.shape[3]).cuda()
                input_code = torch.cat([init_code, src_mask], 1)

                # Inject self-attention
                editor = MutualSelfAttentionControl(start_step=0,
                                                    start_layer=10,
                                                    total_steps=self.n_inference_step,
                                                    guidance_scale=self.guidance_scale)
                attn_processor = 'lora_attn_proc' if self.lora_path else 'attn_proc'
                register_attention_editor_diffusers(self.model, editor, attn_processor=attn_processor)

                # Inter frames attention
                input_latents = [init_code[:1]]
                input_pred = [pred_code[:1]]

                for i in range(1, self.Time):
                    time = i / self.Time
                    metric1 = splat_flowmax(init_code[:1], init_code[1:], flow1to2, 1 - time)
                    metric2 = splat_flowmax(init_code[1:], init_code[:1], flow2to1, time)
                    init = torch.where(metric1 >= metric2, init_code[:1], init_code[1:])
                    input_latents.append(init)

                input_pred.append(torch.load(os.path.join(self.save_dir, 'pred_list.pt')))
                input_latents.append(init_code[1:])
                input_pred.append(pred_code[1:])
                input_latents = torch.cat(input_latents, dim=0)
                input_pred = torch.cat(input_pred, dim=0)

                gen_image = self.model(
                    prompt=prompt,
                    batch_size=input_latents.shape[0],
                    latents=input_latents,
                    pred_x0=input_pred,
                    guidance_scale=self.guidance_scale,
                    num_inference_steps=self.n_inference_step,
                    num_actual_inference_steps=self.n_actual_inference_step,
                    save_dir=self.save_dir
                )

                # Convert generated images to base64
                base64_images = []
                for i in range(gen_image.shape[0]):
                    img = gen_image[i].cpu().permute(1, 2, 0).numpy() * 255
                    img = img.astype(np.uint8)
                    buffered = BytesIO()
                    imageio.imwrite(buffered, img, format='PNG')
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    base64_images.append(img_str)

                return base64_images
        except Exception as e:
            print(f"Error generating frames: {e}")
            raise e
        
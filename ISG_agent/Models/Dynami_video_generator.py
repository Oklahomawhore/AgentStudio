import os
import sys
import time
import io
import base64
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange, repeat
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "./DynamiCrafter/")))
from utils.utils import instantiate_from_config
import requests
from urllib.parse import urlparse


class VideoGenerator:
    def __init__(self,device="cuda"):
        # Initialize models and other parameters
        # Since `interp` controls different hyperparameters and models, we need to prepare for both cases
        self.models = {}
        self.device= device
        self.load_models()

    def load_models(self):
        # Load models for `interp=False` and `interp=True` cases

        # For interp=False
        ckpt_path_no_interp = 'Models/DynamiCrafter/model.ckpt'
        config_path_no_interp = 'Models/DynamiCrafter/configs/inference_256_v1.0.yaml'
        self.models['no_interp'] = self.load_model(ckpt_path_no_interp, config_path_no_interp)

        # For interp=True
        # ckpt_path_interp = 'DynamiCrafter/checkpoints/dynamicrafter_interp_512_v1/model.ckpt'
        # config_path_interp = 'DynamiCrafter/configs/inference_512_v1.0.yaml'
        # self.models['interp'] = self.load_model(ckpt_path_interp, config_path_interp)

    def load_model(self, ckpt_path, config_path):
        config = OmegaConf.load(config_path)
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.perframe_ae = True  # Assuming `perframe_ae` is True by default
        assert os.path.exists(ckpt_path), f"Error: checkpoint not found at {ckpt_path}!"
        model = self.load_model_checkpoint(model, ckpt_path)
        model.eval()
        return model

    def load_model_checkpoint(self, model, ckpt):
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            try:
                model.load_state_dict(state_dict, strict=True)
            except:
                # Rename keys if necessary
                new_pl_sd = OrderedDict()
                for k, v in state_dict.items():
                    new_pl_sd[k] = v
                for k in list(new_pl_sd.keys()):
                    if "framestride_embed" in k:
                        new_key = k.replace("framestride_embed", "fps_embedding")
                        new_pl_sd[new_key] = new_pl_sd[k]
                        del new_pl_sd[k]
                model.load_state_dict(new_pl_sd, strict=True)
        else:
            # Handle deepspeed case
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]] = state_dict['module'][key]
            model.load_state_dict(new_pl_sd)
        print('>>> Model checkpoint loaded.')
        return model

    def generate_video(self, prompt_list, seconds_per_screenshot=1):
        # Determine which model and parameters to use
        # if not interp:
            # Parameters for `interp == False`
        model = self.models['no_interp']
        n_samples = 1
        bs = 1
        height = 256
        width = 256
        unconditional_guidance_scale = 7.5
        ddim_steps = 50
        ddim_eta = 1.0
        text_input = True
        video_length = 32
        frame_stride = 3
        timestep_spacing = 'uniform_trailing'
        guidance_rescale = 0.7
        cfg_img = None
        multiple_cond_cfg = False
        seconds_per_screenshot = seconds_per_screenshot
        # else:
        #     # Parameters for `interp == True`
        #     model = self.models['interp']
        #     n_samples = 1
        #     bs = 1
        #     height = 320
        #     width = 512
        #     unconditional_guidance_scale = 7.5
        #     ddim_steps = 50
        #     ddim_eta = 1.0
        #     text_input = False  # For interpolation, text input is False
        #     video_length = 32
        #     frame_stride = 5
        #     timestep_spacing = 'uniform_trailing'
        #     guidance_rescale = 0.7
        #     cfg_img = None
        #     multiple_cond_cfg = False
        #     seconds_per_screenshot = 1

        # Validate dimensions
        assert (height % 16 == 0) and (width % 16 == 0), "Error: image size [h, w] should be multiples of 16!"
        assert bs == 1, "Current implementation only supports batch size of 1!"

        # Latent noise shape
        h, w = height // 8, width // 8
        channels = model.model.diffusion_model.out_channels
        n_frames = video_length
        print(f'Inference with {n_frames} frames')
        noise_shape = [bs, channels, n_frames, h, w]

        # Load prompts and data
        assert prompt_list is not None, "Error: `prompt_list` not found!"
        data_list, prompt_text = self.load_data_prompts(prompt_list, video_size=(height, width), video_frames=n_frames, interp=False)

        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Prepare prompts and videos
            prompts = [prompt_text] if prompt_text else [""]  # Handle empty prompt if image-to-video
            videos = data_list[0].unsqueeze(0).to("cuda")  # Single video for this batch

            # Generate samples
            samples = self.image_guided_synthesis(
                model, prompts, videos, noise_shape, n_samples, ddim_steps, ddim_eta,
                unconditional_guidance_scale, cfg_img=cfg_img, fs=frame_stride, text_input=text_input,
                multiple_cond_cfg=multiple_cond_cfg, interp=False,
                timestep_spacing=timestep_spacing, guidance_rescale=guidance_rescale
            )

            # Extract base64-encoded screenshots
            generated_video = samples[0, 0]  # Shape: [c, t, h, w]
            base64_screenshots = self.process_video_and_get_base64_screenshots(
                generated_video, fps=8, seconds_per_screenshot=seconds_per_screenshot
            )

        print(f"Time used: {(time.time() - start):.2f} seconds")
        torch.cuda.empty_cache()
        return base64_screenshots

    def load_data_prompts(self, prompt_list, video_size=(256, 256), video_frames=32, interp=False):
        transform = transforms.Compose([
            transforms.Resize(min(video_size)),
            transforms.CenterCrop(video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        data_list = []
        prompt_text = None

        if interp:  # Interpolation case, prompt_list contains two images
            assert prompt_list[0]["type"] == "image" and prompt_list[1]["type"] == "image", \
                "Error: for interpolation, both inputs must be images!"
            # Process the two images for interpolation
            image1 = Image.open(prompt_list[0]["content"]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1)  # [C, 1, H, W]
            image2 = Image.open(prompt_list[1]["content"]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1)  # [C, 1, H, W]
            # Create frame tensors by repeating images to match video_frames
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            data_list.append(frame_tensor)
        else:
            assert prompt_list[0]["type"] == "text" and prompt_list[1]["type"] == "image", \
                "Error: first input must be text, and second must be an image!"
            # Extract text prompt
            prompt_text = prompt_list[0]["content"]
            # Process the image
            image = Image.open(prompt_list[1]["content"]).convert('RGB')
            image_tensor = transform(image).unsqueeze(1)  # [C, 1, H, W]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            data_list.append(frame_tensor)
        return data_list, prompt_text

    def image_guided_synthesis(self, model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.,
                               unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False,
                               multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform',
                               guidance_rescale=0.0, **kwargs):
        from lvdm.models.samplers.ddim import DDIMSampler
        from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
        ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
        batch_size = noise_shape[0]
        fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

        if not text_input:
            prompts = [""] * batch_size

        img = videos[:, :, 0]  # [b, c, h, w]
        img_emb = model.embedder(img)  # [b, l, c]
        img_emb = model.image_proj_model(img_emb)

        cond_emb = model.get_learned_conditioning(prompts)
        cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            z = self.get_latent_z(model, videos)  # [b, c, t, h, w]
            if loop or interp:
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                img_cat_cond = z[:, :, :1, :, :]
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            cond["c_concat"] = [img_cat_cond]  # [b, c, t, h, w]

        if unconditional_guidance_scale != 1.0:
            if model.uncond_type == "empty_seq":
                prompts = batch_size * [""]
                uc_emb = model.get_learned_conditioning(prompts)
            elif model.uncond_type == "zero_embed":
                uc_emb = torch.zeros_like(cond_emb)
            uc_img_emb = model.embedder(torch.zeros_like(img))  # [b, l, c]
            uc_img_emb = model.image_proj_model(uc_img_emb)
            uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
            if model.model.conditioning_key == 'hybrid':
                uc["c_concat"] = [img_cat_cond]
        else:
            uc = None

        # Additional unconditioning if using multiple condition CFG
        if multiple_cond_cfg and cfg_img != 1.0:
            uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
            if model.model.conditioning_key == 'hybrid':
                uc_2["c_concat"] = [img_cat_cond]
            kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
        else:
            kwargs.update({"unconditional_conditioning_img_nonetext": None})

        z0 = None
        cond_mask = None

        batch_variants = []
        for _ in range(n_samples):
            if z0 is not None:
                cond_z0 = z0.clone()
                kwargs.update({"clean_cond": True})
            else:
                cond_z0 = None

            samples, _ = ddim_sampler.sample(
                S=ddim_steps,
                conditioning=cond,
                batch_size=batch_size,
                shape=noise_shape[1:],
                verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                cfg_img=cfg_img,
                mask=cond_mask,
                x0=cond_z0,
                fs=fs,
                timestep_spacing=timestep_spacing,
                guidance_rescale=guidance_rescale,
                **kwargs
            )

            # Reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        # [variants, batch, c, t, h, w]
        batch_variants = torch.stack(batch_variants)
        return batch_variants.permute(1, 0, 2, 3, 4, 5)

    def get_latent_z(self, model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, 'b c t h w -> (b t) c h w')
        z = model.encode_first_stage(x)
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        return z

    def process_video_and_get_base64_screenshots(self, samples, fps=8, seconds_per_screenshot=1):
        """
        Processes a single generated video and captures screenshots at specified intervals,
        returning them as base64-encoded strings.

        Args:
            samples (torch.Tensor): The generated video frames as a tensor of shape [c, t, h, w].
            fps (int): The frames per second for the output video.
            seconds/screenshots (float): Number of screenshots to capture per second from the video.

        Returns:
            base64_screenshots (list): A list of base64-encoded screenshots.
        """
        print(f"Initial tensor shape: {samples.shape}")

        # Ensure the video is detached and on the CPU
        video = samples.detach().cpu()
        print(f"Tensor after detach and move to CPU: {video.shape}")

        # Clamp values to be in the correct range
        video = torch.clamp(video.float(), -1.0, 1.0)

        # Prepare the video for processing
        video = (video + 1.0) / 2.0  # Normalize to [0, 1]
        video = (video * 255).to(torch.uint8)  # Convert to [0, 255]
        print(f"Tensor shape after normalization and conversion: {video.shape}")

        # Permute dimensions to match the expected format (c, t, h, w -> t, h, w, c)
        video = video.permute(1, 2, 3, 0)  # [t, h, w, c]
        print(f"Tensor shape after permutation: {video.shape}")

        # Capture screenshots
        total_frames = video.shape[0]
        frame_interval = max(1, int(seconds_per_screenshot * fps))  # Frames to skip between screenshots
        print(f"Total frames: {total_frames}, Frame interval for screenshots: {frame_interval}")

        # Capture and encode screenshots to base64
        base64_screenshots = []
        for i in range(frame_interval-1, total_frames, frame_interval):
            frame = video[i]  # Extract the ith frame [h, w, c]
            frame_pil = Image.fromarray(frame.numpy(), 'RGB')  # Convert to PIL image

            # Encode the screenshot to base64
            buffered = io.BytesIO()
            frame_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_screenshots.append(img_str)

        print(f"Total {len(base64_screenshots)} screenshots captured.")

        return base64_screenshots

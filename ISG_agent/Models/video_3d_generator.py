import os
import sys
import io
import base64
import math
import numpy as np
import torch
import imageio
import cv2
from glob import glob
from PIL import Image
from rembg import remove
from einops import rearrange, repeat
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor
from urllib.parse import urlparse

# Ensure that the necessary modules are accessible
# Dynamically add the path for generative models and necessary modules
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "./generative_models/")))
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering

class Video3DGenerator:
    def __init__(self, device="cuda"):
        # Initialize model parameters
        self.version = "sv3d_u"
        self.num_frames = 21
        self.num_steps = 50
        self.fps_id = 6
        self.motion_bucket_id = 127
        self.cond_aug = 1e-5
        self.seed = 23
        self.decoding_t = 14
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_config = os.path.join(
            os.path.dirname(__file__),
            "generative_models",
            "scripts",
            "sampling",
            "configs",
            "sv3d_u.yaml",
        )
        self.output_folder = os.path.join(
            os.path.dirname(__file__), "outputs", "AgentWork", "sv3d_u"
        )
        self.verbose = False

        # Set seeds
        torch.manual_seed(self.seed)
        if "cuda" in self.device:
            torch.cuda.manual_seed_all(self.seed)

        # Load the model and filter
        self.model, self.filter = self.load_model(
            self.model_config,
            self.device,
            self.num_frames,
            self.num_steps,
            self.verbose,
        )

        # Debugging: print model device
        print("Model is on device:", next(self.model.parameters()).device)


    def generate_video(self, input_list, screenshots_per_second=1,proportions=[10/12, 11/12,1/12, 2/12]):
        """
        Generate a single sample conditioned on images provided in input_list and return base64-encoded screenshots.

        Args:
            input_list (list): List of dictionaries with 'type' and 'content' keys.
            screenshots_per_second (int): Number of screenshots to capture per second.

        Returns:
            List[str]: Base64-encoded screenshots extracted from the video.
        """
        base64_screenshots = []  # To store screenshots to be returned to the agent

        for input_item in input_list:
            assert (
                input_item["type"] == "image"
            ), "Currently, only image type inputs are supported."

            image = self.load_image(input_item["content"])
            image = self.process_image(image)

            # Convert image to tensor and prepare it for model input
            image_tensor = ToTensor()(image) * 2.0 - 1.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            H, W = image_tensor.shape[2:]
            assert image_tensor.shape[1] == 3
            F = 8
            C = 4
            shape = (self.num_frames, C, H // F, W // F)

            value_dict = {
                "cond_frames_without_noise": image_tensor,
                "motion_bucket_id": self.motion_bucket_id,
                "fps_id": self.fps_id,
                "cond_aug": self.cond_aug,
                "cond_frames": image_tensor
                + self.cond_aug * torch.randn_like(image_tensor).to(self.device),
            }

            # Generate and process the sample
            with torch.no_grad():
                try:
                    if "cuda" in self.device:
                        with torch.autocast(device_type='cuda'):
                            batch, batch_uc = self.get_batch(
                                self.get_unique_embedder_keys_from_conditioner(
                                    self.model.conditioner
                                ),
                                value_dict,
                                [1, self.num_frames],
                                T=self.num_frames,
                                device=self.device,
                            )
                            c, uc = self.model.conditioner.get_unconditional_conditioning(
                                batch,
                                batch_uc=batch_uc,
                                force_uc_zero_embeddings=[
                                    "cond_frames",
                                    "cond_frames_without_noise",
                                ],
                            )

                            for k in ["crossattn", "concat"]:
                                uc[k] = repeat(
                                    uc[k], "b ... -> b t ...", t=self.num_frames
                                )
                                uc[k] = rearrange(
                                    uc[k], "b t ... -> (b t) ...", t=self.num_frames
                                )
                                c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_frames)
                                c[k] = rearrange(
                                    c[k], "b t ... -> (b t) ...", t=self.num_frames
                                )

                            randn = torch.randn(shape, device=self.device)
                            additional_model_inputs = {
                                "image_only_indicator": torch.zeros(2, self.num_frames, device=self.device),
                                "num_video_frames": batch["num_video_frames"],
                            }

                            def denoiser(input, sigma, c):
                                return self.model.denoiser(
                                    self.model.model,
                                    input,
                                    sigma,
                                    c,
                                    **additional_model_inputs
                                )

                            samples_z = self.model.sampler(
                                denoiser, randn, cond=c, uc=uc
                            )
                            samples_x = self.model.decode_first_stage(samples_z)
                            samples_x[-1:] = value_dict["cond_frames_without_noise"]
                            samples = torch.clamp(
                                (samples_x + 1.0) / 2.0, min=0.0, max=1.0
                            )
                    else:
                        # For CPU, we don't use autocast
                        batch, batch_uc = self.get_batch(
                            self.get_unique_embedder_keys_from_conditioner(
                                self.model.conditioner
                            ),
                            value_dict,
                            [1, self.num_frames],
                            T=self.num_frames,
                            device=self.device,
                        )
                        c, uc = self.model.conditioner.get_unconditional_conditioning(
                            batch,
                            batch_uc=batch_uc,
                            force_uc_zero_embeddings=[
                                "cond_frames",
                                "cond_frames_without_noise",
                            ],
                        )

                        for k in ["crossattn", "concat"]:
                            uc[k] = repeat(
                                uc[k], "b ... -> b t ...", t=self.num_frames
                            )
                            uc[k] = rearrange(
                                uc[k], "b t ... -> (b t) ...", t=self.num_frames
                            )
                            c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_frames)
                            c[k] = rearrange(
                                c[k], "b t ... -> (b t) ...", t=self.num_frames
                            )

                        randn = torch.randn(shape, device=self.device)
                        additional_model_inputs = {
                            "image_only_indicator": torch.zeros(2, self.num_frames, device=self.device),
                            "num_video_frames": batch["num_video_frames"],
                        }

                        def denoiser(input, sigma, c):
                            return self.model.denoiser(
                                self.model.model,
                                input,
                                sigma,
                                c,
                                **additional_model_inputs
                            )

                        samples_z = self.model.sampler(
                            denoiser, randn, cond=c, uc=uc
                        )
                        samples_x = self.model.decode_first_stage(samples_z)
                        samples_x[-1:] = value_dict["cond_frames_without_noise"]
                        samples = torch.clamp(
                            (samples_x + 1.0) / 2.0, min=0.0, max=1.0
                        )

                    os.makedirs(self.output_folder, exist_ok=True)
                    base_count = len(glob(os.path.join(self.output_folder, "*.mp4")))

                    # Save the video
                    video_path = os.path.join(
                        self.output_folder, f"{base_count:06d}.mp4"
                    )
                    samples = rearrange(samples, "t c h w -> t h w c") * 255
                    samples_np = samples.cpu().numpy().astype(np.uint8)
                    imageio.mimwrite(video_path, samples_np, fps=30)
                    
                    # Capture screenshots at specified intervals
                    total_frames = samples_np.shape[0]
                    
                    sequential_proportions = sorted(proportions)
                    frame_indices = [int(p * total_frames) for p in sequential_proportions]

                    
                    for i in frame_indices:
                        frame = samples_np[i]  # Extract the frame at the given index
                        screenshot_image = Image.fromarray(frame)

                        # Encode the image to base64
                        buffered = io.BytesIO()
                        screenshot_image.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        base64_screenshots.append(img_str)

                    # Rearrange the screenshots back to the original proportions order
                    base64_screenshots_org = [None] * len(proportions)
                    for idx, prop in enumerate(sequential_proportions):
                        original_index = proportions.index(prop)
                        base64_screenshots_org[original_index] = base64_screenshots[idx]
                    del c, uc, samples_z, samples_x, samples, samples_np
                    torch.cuda.empty_cache()
                except Exception as e:
                    if batch is not None:
                        del batch
                    if batch_uc is not None:
                        del batch_uc
                    if c is not None:
                        del c
                    if uc is not None:
                        del uc
                    if samples_z is not None:
                        del samples_z
                    if samples_x is not None:
                        del samples_x
                    if samples is not None:
                        del samples
                    if samples_np is not None:
                        del samples_np
                    torch.cuda.empty_cache()
                    raise e
        return base64_screenshots  # Return screenshots to the agent

    def load_image(self, content):
        """
        Loads an image from a file path or base64-encoded data.

        Args:
            content (str): The image content, either as a file path or base64 string.

        Returns:
            PIL.Image.Image: The loaded image.

        Raises:
            ValueError: If the image cannot be loaded.
        """
        # Check if content is a file path
        if os.path.exists(content):
            image = Image.open(content).convert("RGBA")
            return image
        else:
            # Try to decode base64-encoded data
            try:
                image_bytes = base64.b64decode(content)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                return image
            except Exception as e:
                raise ValueError(
                    f"Failed to load image from both file path and base64 data: {e}"
                )

    def process_image(self, image):
        """
        Processes the image: removes background, resizes, and centers the object.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The processed image.
        """
        # Remove background and process the image
        if image.mode != "RGBA":
            image.thumbnail([768, 768], Image.Resampling.LANCZOS)
            image = remove(image.convert("RGBA"), alpha_matting=True)

        # Resize object in frame
        image_arr = np.array(image)
        in_h, in_w = image_arr.shape[:2]
        ret, mask = cv2.threshold(
            np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
        )
        x, y, w, h = cv2.boundingRect(mask)

        # Ensure side_len can hold the image slice
        max_size = max(w, h)
        side_len = max_size

        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)

        # Compute the center of the padded image
        center_x = side_len // 2
        center_y = side_len // 2

        # Adjust the slice to fit within the padded image
        padded_image[
            center_y - h // 2 : center_y - h // 2 + h,
            center_x - w // 2 : center_x - w // 2 + w,
        ] = image_arr[y : y + h, x : x + w]

        # Resize frame to 576x576
        rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
        rgba_arr = np.array(rgba) / 255.0
        rgb = (
            rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        )
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))

        return input_image

    def get_unique_embedder_keys_from_conditioner(self, conditioner):
        return list(set([x.input_key for x in conditioner.embedders]))

    def get_batch(self, keys, value_dict, N, T, device):
        batch = {}
        batch_uc = {}

        for key in keys:
            if key == "fps_id":
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]], device=device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "motion_bucket_id":
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]], device=device)
                    .repeat(int(math.prod(N)))
                )
            elif key == "cond_aug":
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]], device=device),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == "cond_frames" or key == "cond_frames_without_noise":
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            elif key == "polars_rad" or key == "azimuths_rad":
                batch[key] = (
                    torch.tensor(value_dict[key], device=device)
                    .repeat(N[0])
                )
            else:
                batch[key] = value_dict[key]

        if T is not None:
            batch["num_video_frames"] = T

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], torch.Tensor):
                batch_uc[key] = torch.clone(batch[key])
        return batch, batch_uc

    def load_model(self, config, device, num_frames, num_steps, verbose=False):
        config = OmegaConf.load(config)
        if "cuda" in device:
            config.model.params.conditioner_config.params.emb_models[
                0
            ].params.open_clip_embedding_config.params.init_device = device

        config.model.params.sampler_config.params.verbose = verbose
        config.model.params.sampler_config.params.num_steps = num_steps
        config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )

        # Instantiate the model
        model = instantiate_from_config(config.model).eval()

        # Move the entire model to the specified device, including submodules
        model = model.to(device)

        # Ensure all submodules are on the correct device
        for module in model.modules():
            for param in module.parameters(recurse=False):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            if hasattr(module, 'buffers'):
                for buf in module.buffers(recurse=False):
                    buf.data = buf.data.to(device)

        filter = DeepFloydDataFiltering(verbose=False, device=device)
        return model, filter

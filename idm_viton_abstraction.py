import io
import gradio as gr
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer

# import spaces
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from torchvision.transforms.functional import to_pil_image


class idm_viton_abstraction:
    def __init__(self) -> None:

        self.base_path = "yisol/IDM-VTON"
        self.example_path = os.path.join(os.path.dirname(__file__), "example")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        self.unet.requires_grad_(False)

        self.tokenizer_one = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )

        self.tokenizer_two = AutoTokenizer.from_pretrained(
            self.base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.base_path, subfolder="scheduler"
        )

        self.text_encoder_one = CLIPTextModel.from_pretrained(
            self.base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.base_path,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
        # "stabilityai/stable-diffusion-xl-base-1.0",
        self.UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            self.base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )

        self.parsing_model = Parsing(0)
        self.openpose_model = OpenPose(0)

        self.UNet_Encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder_one.requires_grad_(False)
        self.text_encoder_two.requires_grad_(False)

        self.tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.pipe = TryonPipeline.from_pretrained(
            self.base_path,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            torch_dtype=torch.float16,
        )
        self.pipe.unet_encoder = self.UNet_Encoder

    def pil_to_binary_mask(self, pil_image, threshold=0):
        np_image = np.array(pil_image)
        grayscale_image = Image.fromarray(np_image).convert("L")
        binary_mask = np.array(grayscale_image) > threshold
        mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if binary_mask[i, j] == True:
                    mask[i, j] = 1
        mask = (mask * 255).astype(np.uint8)
        output_mask = Image.fromarray(mask)
        return output_mask

    # @spaces.GPU
    def start_tryon(self, dict, garm_img, garment_des, is_checked, denoise_steps, seed,garment_type="upper_body"):
        device = "cuda"

        self.openpose_model.preprocessor.body_estimation.model.to(device)
        self.pipe.to(device)
        self.pipe.unet_encoder.to(device)

        garm_img = garm_img.convert("RGB").resize((768, 1024))
        human_img = dict["background"].resize((768, 1024)).convert("RGB")

        if is_checked:
            keypoints = self.openpose_model(human_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location(
                "hd", garment_type, model_parse, keypoints
            )
            mask = mask.resize((768, 1024))
        else:
            mask = self.pil_to_binary_mask(
                dict["layers"][0].convert("RGB").resize((768, 1024))
            )
            # mask = transforms.ToTensor()(mask)
            # mask = mask.unsqueeze(0)
        mask_gray = (1 - transforms.ToTensor()(mask)) * self.tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

        human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

        args = apply_net.create_argument_parser().parse_args(
            (
                "show",
                "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                "./ckpt/densepose/model_final_162be9.pkl",
                "dp_segm",
                "-v",
                "--opts",
                "MODEL.DEVICE",
                "cuda",
            )
        )
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        with torch.no_grad():
            # Extract the images
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    prompt = "model is wearing " + garment_des
                    negative_prompt = (
                        "monochrome, lowres, bad anatomy, worst quality, low quality"
                    )
                    with torch.inference_mode():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds,
                            negative_pooled_prompt_embeds,
                        ) = self.pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            negative_prompt=negative_prompt,
                        )

                        prompt = "a photo of " + garment_des
                        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                        if not isinstance(prompt, List):
                            prompt = [prompt] * 1
                        if not isinstance(negative_prompt, List):
                            negative_prompt = [negative_prompt] * 1
                        with torch.inference_mode():
                            (
                                prompt_embeds_c,
                                _,
                                _,
                                _,
                            ) = self.pipe.encode_prompt(
                                prompt,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                negative_prompt=negative_prompt,
                            )

                        pose_img = (
                            self.tensor_transfrom(pose_img)
                            .unsqueeze(0)
                            .to(device, torch.float16)
                        )
                        garm_tensor = (
                            self.tensor_transfrom(garm_img)
                            .unsqueeze(0)
                            .to(device, torch.float16)
                        )
                        generator = (
                            torch.Generator(device).manual_seed(seed)
                            if seed is not None
                            else None
                        )
                        images = self.pipe(
                            prompt_embeds=prompt_embeds.to(device, torch.float16),
                            negative_prompt_embeds=negative_prompt_embeds.to(
                                device, torch.float16
                            ),
                            pooled_prompt_embeds=pooled_prompt_embeds.to(
                                device, torch.float16
                            ),
                            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                                device, torch.float16
                            ),
                            num_inference_steps=denoise_steps,
                            generator=generator,
                            strength=1.0,
                            pose_img=pose_img.to(device, torch.float16),
                            text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                            cloth=garm_tensor.to(device, torch.float16),
                            mask_image=mask,
                            image=human_img,
                            height=1024,
                            width=768,
                            ip_adapter_image=garm_img.resize((768, 1024)),
                            guidance_scale=2.0,
                        )[0]
        return images[0], mask_gray

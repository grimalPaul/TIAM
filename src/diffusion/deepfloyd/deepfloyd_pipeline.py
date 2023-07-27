from typing import List, Union

import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


class IF_pipeline_all:
    def __init__(self, stage_1_path, stage_2_path, stage_3_path=None) -> None:
        safety_modules = {
            "feature_extractor": None,
            "safety_checker": None,
            "watermarker": None,
        }

        self.stage_1 = DiffusionPipeline.from_pretrained(
            stage_1_path, **safety_modules, variant="fp16", torch_dtype=torch.float16
        )

        self.stage_2 = DiffusionPipeline.from_pretrained(
            stage_2_path,
            text_encoder=None,
            **safety_modules,
            variant="fp16",
            torch_dtype=torch.float16,
        )

        if stage_3_path is not None:
            self.stage_3 = DiffusionPipeline.from_pretrained(
                stage_3_path, **safety_modules, torch_dtype=torch.float16
            )

        else:
            self.stage_3 = None

    def __call__(
        self,
        prompt: Union[str, List[str]],
        generator: Union[torch.Generator, List[torch.Generator]],
        num_images_per_prompt: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            generator (`torch.Generator` or `List[torch.Generator]`):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
        """

        # text embeds
        # do it once for all stages
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(
            prompt=prompt, num_images_per_prompt=num_images_per_prompt
        )

        # stage 1
        image = self.stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt",
        ).images

        # stage 2
        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            output_type="pt" if self.stage_3 is not None else "pil",
        ).images

        if self.stage_3 is not None:
            # stage 3
            image = self.stage_3(
                prompt=prompt, image=image, noise_level=100, generator=generator
            ).images

        return ImagePipelineOutput(images=image)

    def to(self, rank: int):
        self.stage_1.enable_model_cpu_offload(gpu_id=rank)
        self.stage_2.enable_model_cpu_offload(gpu_id=rank)
        if self.stage_3 is not None:
            self.stage_3.enable_model_cpu_offload(gpu_id=rank)

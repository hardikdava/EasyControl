import os
import time
import subprocess
from typing import List, Tuple

from cog import BasePredictor, Input, Path
import torch
from PIL import Image

from core import EasyControl, MODEL_URL, MODEL_CACHE, ASPECT_RATIOS


def download_weights(url, dest, file=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if not file:
        subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class CogPredictor(BasePredictor):

    def setup(self) -> None:
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, ".")
        self.predictor = EasyControl()

    @staticmethod
    def handle_infer_size(aspect_ratio) -> Tuple[int, int]:
        """
        Resolve aspect ratio to image resolution.
        """
        image_resolution = ASPECT_RATIOS[aspect_ratio]
        print(f"Aspect ratio {aspect_ratio} resolved to {image_resolution}")
        return image_resolution

    @torch.inference_mode()
    def predict(
            self,
            prompt: str = Input(
                description="Prompt",
                default="A person wearing a dress",
            ),
            subject_image: Path = Input(
                description="Subject image",
                default=None
            ),
            control_image: Path = Input(
                description="Control image",
                default=None
            ),
            aspect_ratio: str = Input(
                description="Aspect ratio of the output image",
                choices=list(ASPECT_RATIOS.keys()),
                default="1:1",
            ),
            guidance_scale: float = Input(
                description="Guidance scale",
                default=3.5,
                ge=0,
                le=50
            ),
            lora_weights: List[str] = Input(
                description="Huggingface path, or URL to the LoRA weights. Ex: alvdansen/frosting_lane_flux",
                default=[],
            ),
            lora_scales: List[float] = Input(
                description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
                default=[],
            ),
            seed: int = Input(
                description="Random seed. Set for reproducible generation",
                default=42
            ),
            num_inference_steps: int = Input(
                description="Number of inference steps",
                ge=1, le=50, default=25,
            ),
            num_outputs: int = Input(
                description="Number of output images",
                ge=1, le=4, default=1,
            ),
            output_format: str = Input(
                description="Format of the output images",
                choices=["webp", "jpg", "png"],
                default="webp",
            ),
            output_quality: int = Input(
                description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
                default=80,
                ge=0,
                le=100,
            ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        print("seed:", seed)
        print("guidance scale:", guidance_scale)
        print("num inference steps:", num_inference_steps)
        width, height = self.handle_infer_size(aspect_ratio=aspect_ratio)

        outputs = self.predictor.predict(
            prompt=prompt,
            control_image_path=control_image,
            lora_scales=lora_scales,
            lora_weights=lora_weights,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            seed=seed,
            num_outputs=num_outputs,
        )
        return self.post_process(outputs, output_format, output_quality)

    def post_process(
            self, images: List[Image.Image], output_format="webp", output_quality=80
    ):
        output_paths = []
        for i, image in enumerate(images):
            # TODOs: Add safety checker here
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))
        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )
        return output_paths

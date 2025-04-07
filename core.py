import time
from pathlib import Path
from typing import Optional, Union, Tuple

from diffusers.utils import load_image
from PIL import Image
import supervision as sv
import torch

from base import BasePredictor
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
MODEL_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()


control_models = {
    "canny": "canny.safetensors",
    "depth": "depth.safetensors",
    "hedsketch": "hedsketch.safetensors",
    "pose": "pose.safetensors",
    "seg": "seg.safetensors",
    "inpainting": "inpainting.safetensors",
    "subject": "subject.safetensors",
}

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


def load_image(
    image_path: Union[Path, str, Image.Image],
    resolution_wh: Tuple[int, int] = [1024, 1024],
    letterbox: bool = True,
) -> Image:
    """
    Load image with letterbox functionality
    Returns:
        object:
    """

    if isinstance(image_path, (str, Path)):
        img_pil = Image.open(image_path)
    else:
        img_pil = image_path
    # apply letterbox to image
    if letterbox:
        img_cv = sv.pillow_to_cv2(img_pil)
        img_cv = sv.letterbox_image(img_cv, resolution_wh)
        print(f"letterbox applied to image with resolution {resolution_wh}")
        img_pil = sv.cv2_to_pillow(img_cv)
    return img_pil


class EasyControl(BasePredictor):

    def __init__(self):
        """
        Initialize the In-Context Multi-LoRA model.
        """
        super().__init__()

        self.pipes = {}
        self.pipes["pose"] = FluxPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_CACHE,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
        self.pipes["pose"].transformer = transformer
        self.pipes["pose"].to("cuda")

        path = control_models["pose"]
        set_single_lora(self.pipes["pose"].transformer, path, lora_weights=[1], cond_size=512)

    def predict(self,
                control_image_path: Union[str, Path],
                prompt: str,
                guidance_scale: float = 3.5,
                num_inference_steps: int = 30,
                seed: Optional[int] = None,
                num_outputs: int = 1,
                width: int = 1024,
                height: int = 1024,
                ):
        start_time = time.time()
        spatial_image = load_image(control_image_path, letterbox=False)
        results = self.pipes["pose"](
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(5),
            spatial_images=[spatial_image],
            subject_images=[],
            cond_size=512,
        ).images

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        # Clear cache after generation
        clear_cache(self.pipe["pose"].transformer)
        return results

import time
from pathlib import Path
from typing import Optional, Union

from PIL import Image
import torch

from base import BasePredictor
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora

MODEL_CACHE = "FLUX.1-dev"
MODEL_NAME = "black-forest-labs/FLUX.1-dev"
MODEL_URL = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)

control_models = {
    "pose": "pose.safetensors",
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


class EasyControl(BasePredictor):

    def __init__(self):
        """
        Initialize the In-Context Multi-LoRA model.
        """
        super().__init__()

        self.pipe = FluxPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.bfloat16
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_CACHE,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
        self.pipe.transformer = transformer
        self.pipe.to("cuda")

        path = control_models["pose"]
        set_single_lora(self.pipe.transformer, path, lora_weights=[1], cond_size=512)

    def clear_cache(self):
        for name, attn_processor in self.pipe.transformer.attn_processors.items():
            attn_processor.bank_kv.clear()

    def predict(self,
                control_image_path: Union[str, Path],
                prompt: str,
                guidance_scale: float = 3.5,
                num_inference_steps: int = 30,
                seed: Optional[int] = None,
                lora_weights: Optional[Union[str, list]] = None,
                lora_scales: Optional[Union[float, list]] = None,
                num_outputs: int = 1,
                width: int = 1024,
                height: int = 1024,
                ):
        start_time = time.time()
        # load dev lora
        self.load_loras(
            hf_loras=lora_weights, lora_scales=lora_scales
        )
        spatial_image = Image.open(control_image_path)
        results = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            spatial_images=[spatial_image],
            subject_images=[],
            cond_size=512,
        ).images

        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        # Clear cache after generation
        self.clear_cache()
        return results

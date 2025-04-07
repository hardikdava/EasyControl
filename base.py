from abc import abstractmethod
import time
import os
import re

from weights import WeightsDownloadCache


class BasePredictor:

    def __init__(self):
        self.pipe = None
        self.weights_cache = WeightsDownloadCache()

    @abstractmethod
    def setup_pipeline(self):
        """
        Setup the pipeline
        :return:
        """
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass

    def load_loras(self, hf_loras, lora_scales):
        # handle inputs
        lora_local_paths = []
        lora_weights = []
       
        names = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]
        count = 0

        if not hf_loras:
            return

        if isinstance(hf_loras, str):
            hf_loras = [hf_loras]

        if isinstance(lora_scales, float) or isinstance(lora_scales, int):
            lora_scales = len(hf_loras) * [lora_scales]
        elif not lora_scales:
            lora_scales = len(hf_loras) * [0.8]
        elif len(lora_scales) == 1 and len(hf_loras) > 1:
            lora_scales = len(hf_loras) * [lora_scales[0]]
        # loop through each lora
        for hf_lora in hf_loras:
            t1 = time.time()
            if re.match(r"^https?://civitai.com", hf_lora):
                print(f"Downloading LoRA weights from - Civitai URL: {hf_lora}")
                local_weights_cache = self.weights_cache.ensure(hf_lora, file=True)
                lora_path = os.path.join(
                    local_weights_cache, "output/flux_train_civitai/lora.safetensors"
                )
                adapter_name = names[count]
                lora_local_paths.append(lora_path)
                lora_weights.append(lora_scales[count])
                count += 1

            elif hf_lora.endswith(".safetensors"):
                print(f"Downloading LoRA weights from - safetensor URL: {hf_lora}")
                try:
                    lora_path = self.weights_cache.ensure(hf_lora, file=True)
                except Exception as e:
                    print(f"Error downloading LoRA weights: {e}")
                    continue
                adapter_name = names[count]
                lora_local_paths.append(lora_path)
                lora_weights.append(lora_scales[count])
                count += 1
            else:
                print(f"Downloading LoRA weights from - Replicate URL: {hf_lora}")
                local_weights_cache = self.weights_cache.ensure(hf_lora)
                lora_path = os.path.join(
                    local_weights_cache, "output/flux_train_replicate/lora.safetensors"
                )
                adapter_name = names[count]
                lora_local_paths.append(lora_path)
                lora_weights.append(lora_scales[count])
                count += 1

            t2 = time.time()
            print(f"Loading LoRA took: {t2 - t1:.2f} seconds")
        adapter_names = names[:count]
        adapter_weights = lora_scales[:count]
        print(f"adapter_names: {adapter_names}")
        print(f"adapter_weights: {adapter_weights}")

        return lora_local_paths, lora_weights





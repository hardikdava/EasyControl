from abc import abstractmethod
import time
import os

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

        self.pipe.unload_lora_weights()
        names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                 'u', 'v', 'w', 'x', 'y', 'z']
        count = 0

        if not hf_loras:
            return

        if isinstance(hf_loras, str):
            hf_loras = [hf_loras]

        if isinstance(lora_scales, float) or isinstance(lora_scales, int):
            lora_scales = len(hf_loras) * [lora_scales]
        elif lora_scales is None:
            lora_scales = len(hf_loras) * [1.0]
        # loop through each lora
        for hf_lora in hf_loras:
            t1 = time.time()
            print(f"Downloading LoRA weights from - Replicate URL: {hf_lora}")
            local_weights_cache = self.weights_cache.ensure(hf_lora)
            lora_path = os.path.join(local_weights_cache, "output/flux_train_replicate/lora.safetensors")
            adapter_name = names[count]
            count += 1
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)

            t2 = time.time()
            print(f"Loading LoRA took: {t2 - t1:.2f} seconds")
        adapter_names = names[:count]
        adapter_weights = lora_scales[:count]
        print(f"adapter_names: {adapter_names}")
        print(f"adapter_weights: {adapter_weights}")
        self.last_loaded_loras = hf_loras
        self.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)




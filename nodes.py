import os
try:
    from modelscope.pipelines import pipeline
    from modelscope.outputs import OutputKeys
except Exception as e:
    os.system('pip install "modelscope" --upgrade -f https://pypi.org/project/modelscope/')
    from modelscope.pipelines import pipeline
    from modelscope.outputs import OutputKeys

from PIL import Image
import numpy as np
import folder_paths

class ModelscopePipelineLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": ("STRING", {"default":"image-to-video"}),
                "model": ("STRING", {"default":"damo/i2vgen-xl"}),
                "model_revision": ("STRING", {"default":"v1.1.3"}),
                "device": ("STRING", {"default":"cuda:0"}),
            }
        }
        
    RETURN_TYPES = ("ModelscopePipeline",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "Modelscope"

    def load_checkpoint(self,task,model,model_revision,device):
        pipe = pipeline(task=task, model=model, model_revision=model_revision, device=device)
        return (pipe,)

class I2VGEN_XL_Simple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ModelscopePipeline",),
                "image": ("IMAGE",),
                "text": ("STRING", {"default":""}),
            }
        }

    RETURN_TYPES = ({},)
    RETURN_NAMES = ("video",)
    FUNCTION = "run_inference"
    CATEGORY = "Modelscope"

    def run_inference(self,pipe,image,text):
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        comfy_path = os.path.dirname(folder_paths.__file__)
        image_path=f'{comfy_path}/output/i2vgen_xl.png'
        image.save(image_path)
        output_video_path = pipe(image_path, caption=text)[OutputKeys.OUTPUT_VIDEO]
        return (output_video_path,)


NODE_CLASS_MAPPINGS = {
    "Modelscope Pipeline Loader":ModelscopePipelineLoader,
    "I2VGEN-XL Simple":I2VGEN_XL_Simple,
}


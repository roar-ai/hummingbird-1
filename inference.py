import os
import numpy as np
import torch
import argparse
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


parser = argparse.ArgumentParser()
parser.add_argument("--reference_image", type=str)
parser.add_argument("--attribute", type=str)
parser.add_argument("--output_path", type=str)

args = parser.parse_args()


accelerator = Accelerator()


def generate_images(pipe, args):

    pipe = accelerator.prepare(pipe)
    pipe = pipe.to(accelerator.device)
    processor = LlavaNextProcessor.from_pretrained(
        "/mnt/minhquan/llava-v1.6-mistral-7b-hf"
    )

    vlm = LlavaNextForConditionalGeneration.from_pretrained(
        "/mnt/minhquan/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attention_2=True,
    )
    vlm.to(accelerator.device)
    instruction_prompt = f"[INST] <image>\nProvide a concise description of key elements and context of the image directly related to the attribute: {args.attribute}. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]"
    generator_s = torch.Generator("cuda").manual_seed(93)
    input_image = [Image.open(args.reference_image).convert("RGB")]
    with torch.no_grad():

        inputs = processor(
            instruction_prompt,
            images=input_image,
            return_tensors="pt",
            padding=True,
        ).to(accelerator.device)
        output_text = vlm.generate(**inputs, max_new_tokens=165)
        output_text = processor.batch_decode(output_text, skip_special_tokens=True)
        output_text = [output.split(" [/INST]")[1].strip() for output in output_text]
        print("Context description from MLLM: ", output_text[0])

        image = pipe(
            prompt=output_text,
            image=input_image,
            num_inference_steps=50,
            generator=generator_s,
            guidance_scale=7.5,
            use_origin=False,
        ).images[0]

        image.save(args.output_path)


def main():
    from pipelines.pipeline_stable_diffusion_xl import HummingbirdPipeline

    pipe = HummingbirdPipeline.from_pretrained(
        "./hummingbird/stable-diffusion-xl-base-1.0"
    )
    from peft import PeftModel

    pipe.unet = PeftModel.from_pretrained(
        pipe.unet,
        "./hummingbird/lora_unet_65000",
    )
    
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.image_encoder.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()


    generate_images(pipe, args)


if __name__ == "__main__":
    main()

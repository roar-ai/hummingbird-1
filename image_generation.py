import os
import numpy as np
import torch
import argparse
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator

from torch.utils.data import Dataset
import random
import shutil
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


from clip_custom.hoi_dataset import BongardDatasetBLIP
from clip_custom.imagenet_dataset import ImageNetSketchDataset
from clip_custom.mme_dataset import MMEDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--save_image_gen", type=str)
parser.add_argument("--dfu_times", type=int, default=1)
parser.add_argument("--generator", type=str, default="hummingbird")

args = parser.parse_args()


accelerator = Accelerator()
os.makedirs(args.save_image_gen, exist_ok=True)


def generate_images(pipe, dataloader, args):

    if args.generator == "hummingbird" or args.generator == "sdxl":
        pipe, dataloader = accelerator.prepare(pipe, dataloader)
        pipe = pipe.to(accelerator.device)
        processor = LlavaNextProcessor.from_pretrained(
            "llava-v1.6-mistral-7b-hf"
        )

        vlm = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,
        )
        vlm.to(accelerator.device)
        instruction_prompts = {
            "hoi": "[INST] <image>\nProvide a concise description of key elements and context of the image to focus on the interaction between {} and {}. The answer is {}. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]",
            "vqav2": "[INST] <image>\nProvide a concise description of key elements and context of the image to directly answer the question: {} The answer is {}. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]",
            "imagenet": "[INST] <image>\n{} Provide a concise description of the object in the image. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]",
            "generic": "[INST] <image>\nGiven the image, generate a caption which captures all the key aspects of the image. It should not contain anything that is not present in the image. A key aspect can be a count of entities present, entities that are noteworthy, relationships between objects or informative descriptions of the presented image. [/INST]",
        }
        generator_s = torch.Generator("cuda").manual_seed(93)
    elif args.generator == "difftpt":
        pipe, dataloader = accelerator.prepare(pipe, dataloader)
        pipe = pipe.to(accelerator.device)
    elif args.generator == "boomerang":
        from diffusers import DPMSolverMultistepScheduler
        from pipelines.boomerang import (
            encode_latents,
            boomerang_forward,
            boomerang_reverse,
        )

        NUM_INFERENCE_STEPS = (
            50 if isinstance(pipe.scheduler, DPMSolverMultistepScheduler) else 100
        )
        pipe, dataloader = accelerator.prepare(pipe, dataloader)
        pipe = pipe.to(accelerator.device)

    with torch.no_grad():
        for count, batch in enumerate(dataloader):

            if args.dataset == "hoi":
                (
                    original_images,
                    answer,
                    query_prompts,
                    query_category,
                    subject_class,
                    object_class,
                    act_class,
                    file_ids,
                ) = batch[0]
                answer = [1, 0]
            else:
                (
                    original_images,
                    answer,
                    query_prompts,
                    query_category,
                    file_id,
                ) = batch[0]
                file_ids = [file_id]
                answer = [answer]

            print(f"{count} / {len(dataloader)}, {file_ids}.")
            for j in range(len(original_images)):
                file_id = file_ids[j]
                os.makedirs(
                    os.path.join(args.save_image_gen, *file_id.split("/")[:-1]),
                    exist_ok=True,
                )
                dist_path = os.path.join(args.save_image_gen, *file_id.split("/"))
                if not os.path.exists(dist_path):
                    original_images[j].save(dist_path)

                if args.generator == "hummingbird" or args.generator == "sdxl":
                    input_image = [original_images[j]]
                    batch_size = 1
                    if args.dataset == "hoi":
                        instruct_prompt = [
                            instruction_prompts[query_category]
                            .format(subject_class[j], object_class[j], "")
                            .strip()
                        ] * batch_size
                    else:
                        instruct_prompt = [
                            instruction_prompts[query_category]
                            .format(query_prompts, "")
                            .strip()
                        ] * batch_size
                    print(instruct_prompt)
                    inputs = processor(
                        instruct_prompt,
                        images=input_image,
                        return_tensors="pt",
                        padding=True,
                    ).to(accelerator.device)
                    output_text = vlm.generate(**inputs, max_new_tokens=200)
                    output_text = processor.batch_decode(
                        output_text, skip_special_tokens=True
                    )
                    output_text = [
                        output.split(" [/INST]")[1]
                        .replace("Yes, ", "")
                        .replace("No, ", "")
                        .strip()
                        for output in output_text
                    ]
                    print("Pseudo text: ", output_text[0])
                    image_id = file_id.split("/")[-1]
                    txt_path = os.path.join(
                        args.save_image_gen,
                        *file_id.split("/")[:-1],
                        str(image_id.split(".jpg")[0]) + "_" + str(answer[j]) + ".txt",
                    )
                    with open(txt_path, "w") as f:
                        f.write(output_text[0])
                    f.close()
                for time_ in range(args.dfu_times):
                    if args.generator == "hummingbird":
                        generator_s = torch.Generator("cuda").manual_seed(time_)
                        images = pipe(
                            prompt=output_text,
                            image=input_image,
                            num_inference_steps=25,
                            generator=generator_s,
                            guidance_scale=15,
                            grad_steps=5,
                            use_origin=False,
                        ).images
                    elif args.generator == "difftpt":

                        images = pipe(original_images, guidance_scale=3).images
                    elif args.generator == "randaugment":
                        images = [
                            pipe(original_image) for original_image in original_images
                        ]
                    elif args.generator == "sdxl":
                        generator_s = torch.Generator("cuda").manual_seed(time_)
                        images = pipe(
                            prompt=output_text,
                            num_inference_steps=25,
                            generator=generator_s,
                        ).images

                    elif args.generator == "boomerang":
                        percent_noise = 0.5
                        prompt = f"{query_prompts}"
                        clean_z = encode_latents(
                            pipe, original_images[0].convert("RGB")
                        )
                        noisy_z = boomerang_forward(pipe, percent_noise, clean_z)
                        images = boomerang_reverse(
                            pipe,
                            prompt,
                            percent_noise,
                            noisy_z,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            output_type="pil",
                        ).images

                    for index in range(len(images)):
                        image_id = file_id.split("/")[-1]
                        images[index].save(
                            os.path.join(
                                args.save_image_gen,
                                *file_id.split("/")[:-1],
                                str(image_id.split(".jpg")[0])
                                + "_"
                                + str(answer[j])
                                + "_"
                                + str(time_)
                                + ".jpg",
                            )
                        )


def main():
    if args.generator == "hummingbird":
        from pipelines.pipeline_stable_diffusion_xl import HummingbirdPipeline

        pipe = HummingbirdPipeline.from_pretrained(
            "hummingbird/stable-diffusion-xl-base-1.0"
        )
        from peft import PeftModel

        pipe.unet = PeftModel.from_pretrained(
            pipe.unet,
            "hummingbird/lora_unet_65000",
        )
        
        pipe.unet.eval()
        pipe.vae.eval()
        pipe.image_encoder.eval()
        pipe.text_encoder.eval()
        pipe.text_encoder_2.eval()
    elif args.generator == "difftpt":
        from diffusers import StableDiffusionImageVariationPipeline

        model_name_path = "sd-image-variations-diffusers"
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(model_name_path)
    elif args.generator == "randaugment":
        from torchvision.transforms import RandAugment

        pipe = RandAugment()
    elif args.generator == "sdxl":
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stable-diffusion-xl-base-1.0"
        )

    elif args.generator == "boomerang":
        from diffusers import StableDiffusionPipeline
        from diffusers import DPMSolverMultistepScheduler
        from pipelines.boomerang import __call__

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        type(pipe).__call__ = __call__

    if args.dataset == "mme":
        dataset = MMEDataset()
    elif args.dataset == "imagenet_sketch":
        dataset = ImageNetSketchDataset("imagenet-sketch")
    elif args.dataset == "hoi":
        dataset = BongardDatasetBLIP(
            "bongard-datasets",
            "unseen_obj_unseen_act",
            "test",
            None,
            None,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    generate_images(pipe, dataloader, args)


if __name__ == "__main__":
    main()

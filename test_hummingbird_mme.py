import argparse

from pipelines.InternVL2_8B.conversation import get_conv_template


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--synthetic_dir", type=str)
parser.add_argument("--dataset", type=str, default="mme")
parser.add_argument("--model", type=str, default="llava")

args = parser.parse_args()


def main():
    import os
    from PIL import Image
    from tqdm import tqdm
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    device = "cuda:0"
    if args.model == "llava":
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(
            "llava-v1.6-mistral-7b-hf"
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
        )
    elif args.model == "instructblip":
        from transformers import (
            InstructBlipProcessor,
            InstructBlipForConditionalGeneration,
        )
        import torch
        from PIL import Image
        import requests

        model = InstructBlipForConditionalGeneration.from_pretrained(
            "instructblip-vicuna-7b"
        )
        processor = InstructBlipProcessor.from_pretrained(
            "instructblip-vicuna-7b"
        )
        model.to(device)

    elif args.model == "internvl":
        import math
        import numpy as np
        import torch
        import torchvision.transforms as T
        from decord import VideoReader, cpu
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode
        from transformers import AutoModel, AutoTokenizer
        from pipelines.InternVL2_8B.modeling_internvl_chat import InternVLChatModel

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose(
                [
                    T.Lambda(
                        lambda img: img.convert("RGB") if img.mode != "RGB" else img
                    ),
                    T.Resize(
                        (input_size, input_size),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=MEAN, std=STD),
                ]
            )
            return transform

        def find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, image_size
        ):
            best_ratio_diff = float("inf")
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(
            image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
        ):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size
            )

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size,
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def load_image(image, input_size=448, max_num=12):
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(
                image, image_size=input_size, use_thumbnail=True, max_num=max_num
            )
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        def split_model(model_name):
            device_map = {}
            world_size = torch.cuda.device_count()
            num_layers = {
                "InternVL2-1B": 24,
                "InternVL2-2B": 24,
                "InternVL2-4B": 32,
                "InternVL2-8B": 32,
                "InternVL2-26B": 48,
                "InternVL2-40B": 60,
                "InternVL2-Llama3-76B": 80,
            }[model_name]
            # Since the first GPU will be used for ViT, treat it as half a GPU.
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f"language_model.model.layers.{layer_cnt}"] = i
                    layer_cnt += 1
            device_map["vision_model"] = 0
            device_map["mlp1"] = 0
            device_map["language_model.model.tok_embeddings"] = 0
            device_map["language_model.model.embed_tokens"] = 0
            device_map["language_model.output"] = 0
            device_map["language_model.model.norm"] = 0
            device_map["language_model.lm_head"] = 0
            device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

            return device_map

        # If you set `load_in_8bit=True`, you will need one 80GB GPUs.
        # If you set `load_in_8bit=False`, you will need at least two 80GB GPUs.
        path = "InternVL2-8B"
        device_map = split_model("InternVL2-8B")
        model = InternVLChatModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    image_dataset = load_dataset("lmms-lab/MME")[
        "test"
    ]
    dataloader = DataLoader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: batch,
    )
    accuracy = {
        "count": [],
        "color": [],
        "existence": [],
        "position": [],
        "scene": [],
        "artwork": [],
        "commonsense_reasoning": []

    }

    weights = {
        "count": 1.0,
        "color": 1.0,
        "existence": 1.0,
        "position": 1.0,
        "scene": 1.0,
        "artwork": 1.0,
        "commonsense_reasoning": 1.0
    }
    if args.model == "llava":
        question_template = "[INST] <image>\nQuestion: {}\nAnswer: [/INST]"
        yes_template = "[INST] <image>\nQuestion: {}\nAnswer: yes [/INST]"
        no_template = "[INST] <image>\nQuestion: {}\nAnswer: no [/INST]"
    elif args.model == "instructblip":
        question_template = "{} Please answer yes or no."
        yes_template = "{} Please answer yes or no. yes "
        no_template = "{} Please answer yes or no. no "
    elif args.model == "internvl":
        question_template = "<image>\n{}"
        yes_template = "<image>\n{} yes"
        no_template = "<image>\n{} no"

    idx = 0
    for data in tqdm(dataloader):
        batch = data[0]
        question_id, image, question, answer, category = (
            batch["question_id"],
            batch["image"],
            batch["question"],
            batch["answer"],
            batch["category"],
        )
        if category not in accuracy.keys():
            continue
        if args.model == "llava":
            real_inputs = processor(
                images=image,
                text=question_template.format(question),
                return_tensors="pt",
            ).to(device=device, dtype=torch.float16)

            yes_inputs = processor(
                images=image, text=yes_template.format(question), return_tensors="pt"
            ).to(device=device, dtype=torch.float16)

            no_inputs = processor(
                images=image, text=no_template.format(question), return_tensors="pt"
            ).to(device=device, dtype=torch.float16)

            yes_loss = model(
                **real_inputs, return_dict=True, labels=yes_inputs.input_ids
            ).loss
            no_loss = model(
                **real_inputs, return_dict=True, labels=no_inputs.input_ids
            ).loss
        elif args.model == "instructblip":

            real_inputs = processor(
                images=image,
                text=question_template.format(question),
                return_tensors="pt",
            ).to(device)
            yes_inputs = processor(
                images=image, text=yes_template.format(question), return_tensors="pt"
            ).to(device)
            no_inputs = processor(
                images=image, text=no_template.format(question), return_tensors="pt"
            ).to(device)
            yes_loss = model(
                **real_inputs, return_dict=True, labels=yes_inputs.input_ids
            ).loss
            no_loss = model(
                **real_inputs, return_dict=True, labels=no_inputs.input_ids
            ).loss

        elif args.model == "internvl":
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()

            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
            model_inputs = preprocess_inputs(
                model,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN,
                tokenizer,
                question_template,
                question,
                num_patches_list,
            )
            input_ids = model_inputs["input_ids"].cuda()
            attention_mask = model_inputs["attention_mask"].cuda()

            yes_inputs = preprocess_inputs(
                model,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN,
                tokenizer,
                yes_template,
                question,
                num_patches_list,
            )
            no_inputs = preprocess_inputs(
                model,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN,
                tokenizer,
                no_template,
                question,
                num_patches_list,
            )
            with torch.no_grad():
                yes_loss = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=torch.tensor(
                        [1] * pixel_values.size(0), dtype=torch.long
                    ),
                    labels=yes_inputs.input_ids.cuda(),
                    return_dict=True,
                ).loss

                no_loss = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=torch.tensor(
                        [1] * pixel_values.size(0), dtype=torch.long
                    ),
                    labels=no_inputs.input_ids.cuda(),
                    return_dict=True,
                ).loss

        real_loss = torch.tensor([yes_loss, no_loss])

        image_id = question_id.split("/")[-1]
        augment_img = Image.open(
            os.path.join(
                args.synthetic_dir,
                question_id.split("/")[0],
                image_id.split(".")[0] + f".png_{answer}_0.jpg" # + image_id.split(".")[1],
            )
        ).convert("RGB")

        if args.model == "llava":
            augment_inputs = processor(
                images=augment_img,
                text=question_template.format(question),
                return_tensors="pt",
            ).to(device=device, dtype=torch.float16)
            yes_inputs = processor(
                images=augment_img,
                text=yes_template.format(question),
                return_tensors="pt",
            ).to(device=device, dtype=torch.float16)

            no_inputs = processor(
                images=augment_img,
                text=no_template.format(question),
                return_tensors="pt",
            ).to(device=device, dtype=torch.float16)
            yes_loss = model(
                **augment_inputs, return_dict=True, labels=yes_inputs.input_ids
            ).loss
            no_loss = model(
                **augment_inputs, return_dict=True, labels=no_inputs.input_ids
            ).loss

        elif args.model == "instructblip":
            augment_inputs = processor(
                images=augment_img,
                text=question_template.format(question),
                return_tensors="pt",
            ).to(device)
            yes_inputs = processor(
                images=augment_img,
                text=yes_template.format(question),
                return_tensors="pt",
            ).to(device)

            no_inputs = processor(
                images=augment_img,
                text=no_template.format(question),
                return_tensors="pt",
            ).to(device)
            yes_loss = model(
                **augment_inputs, return_dict=True, labels=yes_inputs.input_ids
            ).loss
            no_loss = model(
                **augment_inputs, return_dict=True, labels=no_inputs.input_ids
            ).loss

        elif args.model == "internvl":
            pixel_values = load_image(augment_img, max_num=12).to(torch.bfloat16).cuda()
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
            model_inputs = preprocess_inputs(
                model,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN,
                tokenizer,
                question_template,
                question,
                num_patches_list,
            )
            input_ids = model_inputs["input_ids"].cuda()
            attention_mask = model_inputs["attention_mask"].cuda()

            yes_inputs = preprocess_inputs(
                model,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN,
                tokenizer,
                yes_template,
                question,
                num_patches_list,
            )
            no_inputs = preprocess_inputs(
                model,
                IMG_START_TOKEN,
                IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN,
                tokenizer,
                no_template,
                question,
                num_patches_list,
            )
            with torch.no_grad():
                yes_loss = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=torch.tensor(
                        [1] * pixel_values.size(0), dtype=torch.long
                    ),
                    labels=yes_inputs.input_ids.cuda(),
                    return_dict=True,
                ).loss

                no_loss = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=torch.tensor(
                        [1] * pixel_values.size(0), dtype=torch.long
                    ),
                    labels=no_inputs.input_ids.cuda(),
                    return_dict=True,
                ).loss

        augment_logits = torch.tensor([yes_loss, no_loss])

        weight = weights[category]
        logits = (real_loss * weight + augment_logits) / (weight + 1)
        predict_id = torch.argmin(logits)
        predict = "Yes" if predict_id == 0 else "No"
        # print(predict, answer)
        accuracy[category].append(int(predict == answer))

        idx += 1

    for key, value in accuracy.items():
        if len(value) == 0:
            continue
        print(f"{key}: {sum(value) / len(value)}")

    for key, value in accuracy.items():
        value_plus = []
        if len(value) == 0:
            continue
        for i in range(0, len(value), 2):
            if value[i] == True and value[i + 1] == True:
                value_plus.append(1)
            else:
                value_plus.append(0)

        print(f"{key}+: {sum(value_plus) / len(value_plus)}")


def preprocess_inputs(
    model,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    IMG_CONTEXT_TOKEN,
    tokenizer,
    question_template,
    question,
    num_patches_list,
):
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
    template.append_message(template.roles[0], question_template.format(question))
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    return model_inputs


if __name__ == "__main__":
    main()

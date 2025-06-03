import os
import argparse
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import (
    Dataset,
    DataLoader,
    SubsetRandomSampler,
    IterableDataset,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import numpy as np

from pipelines.pipeline_stable_diffusion_xl import (
    retrieve_timesteps,
    HummingbirdPipeline,
)
from pipelines.scheduling_ddim import DDIMScheduler

# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from aesthetic import load_models
import open_clip
from tqdm import tqdm
import csv
import json
from pprint import pprint
import pandas as pd
import logging
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    CLIPTokenizer,
)
from clip_custom import (
    BongardDatasetBLIP,
    ClipTestTimeTuning,
    test_time_tuning,
    VQAv2Dataset,
    MMEDataset,
    ImageNetDataset,
    VQAv2GQADataset,
)
from peft import get_peft_model, LoraConfig, TaskType


def set_processors(attentions):
    for attn in attentions:
        attn.set_processor(AttnProcessor2_0())


def set_torch_2_attn(unet):
    optim_count = 0
    for name, module in unet.named_modules():
        if "attn1" or "attn2" == name.split(".")[-1]:
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


def standard_process(image):
    output = torch.nn.functional.adaptive_avg_pool2d(image, 224)
    output = torchvision.transforms.functional.normalize(
        output,
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return output


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse():
    parser = argparse.ArgumentParser(description="PyTorch DDP Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        metavar="N",
        help="mini-batch size per process (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-6,
        type=float,
        metavar="LR",
        help="Initial learning rate",
    )
    parser.add_argument(
        "--lr_unet", default=1e-6, type=float, help="unet learning rate"
    )
    parser.add_argument(
        "--lr_image", default=1e-6, type=float, help="text learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--output-path",
        default="dummy",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--dataset",
        default="hoi",
        type=str,
        help="name of dataset including [hoi, mme, imagenet]",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--hpsv2",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--pickscore",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--clip",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--aesthetic",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--context",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--grad_steps", default=5, type=int, help="truncate backpropagation"
    )

    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", 0), type=int)
    parser.add_argument("--sync_bn", action="store_true", help="enabling sync BN.")
    args = parser.parse_args()
    return args


def main():
    args = parse()
    test_image = Image.open("./examples/image-3.jpg").convert("RGB")

    pick_reward = args.pickscore > 0
    hps_reward = args.hpsv2 > 0
    clip_reward = args.clip > 0
    aes_reward = args.aesthetic > 0
    context_reward = args.context > 0

    # Enable tensor-core
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cudnn.benchmark = True
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    # #################### data and sampler ###################################################
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    processor = LlavaNextProcessor.from_pretrained(
        "llava-v1.6-mistral-7b-hf"
    )

    vlm = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )

    instruction_prompts = {
        "hoi": "[INST] <image>\nProvide a concise description of key elements and context of the image to focus on the interaction between {} and {}. The answer is {}. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]",
        "vqav2": "[INST] <image>\nProvide a concise description of key elements and context of the image to directly answer the question: {} The answer is {}. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]",
        "imagenet": "[INST] <image>\n{} Provide a concise description of the object in the image. Ensure the description is clear, concise, in an engineering prompt style, and directly related to the question being asked. [/INST]",
    }

    if args.dataset == "hoi":
        image_dataset = BongardDatasetBLIP(
            "bongard_datasets",
            "unseen_obj_unseen_act",
            "train",
            transform,
            None,
        )
    elif args.dataset == "mme":
        image_dataset = MMEDataset()
    elif args.dataset == "vqav2":
        image_dataset = VQAv2Dataset()
    elif args.dataset == "imagenet":
        image_dataset = ImageNetDataset(
            data_root="./imagenet", transform=None
        )
    elif args.dataset == "vqav2gqa":
        image_dataset = VQAv2GQADataset()
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(image_dataset)

    dataloader = DataLoader(
        image_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=lambda batch: batch,
    )

    if args.local_rank == 0:
        os.makedirs(args.output_path, exist_ok=True)

    # ########################### model creation stuff ############################################
    generator_s = torch.Generator("cuda").manual_seed(93)
    pipe_hummingbird = HummingbirdPipeline.from_pretrained(
        "hummingbird/stable-diffusion-xl-base-1.0"
    )
    if hasattr(F, "scaled_dot_product_attention"):
        set_torch_2_attn(pipe_hummingbird.unet)

    scheduler_config = pipe_hummingbird.scheduler.config
    scheduler_config["prediction_type"] = "epsilon"
    noise_scheduler = DDIMScheduler.from_config(scheduler_config)
    noise_scheduler.set_timesteps(25)
    pipe_hummingbird.scheduler = noise_scheduler
    print("prediction type: ", noise_scheduler.config.prediction_type)

    if clip_reward:
        clip_model, *_ = open_clip.create_model_and_transforms(
            "ViT-g-14",
            pretrained="laion2b_s34b_b88k",
        )
        clip_model = clip_model.to("cuda")
    if aes_reward:
        model_aes = load_models()
    if pick_reward:
        from transformers import AutoProcessor, AutoModel

        pick_model = (
            AutoModel.from_pretrained("./pickscore/pickmodel").eval().to("cuda")
        )
    if hps_reward:
        from typing import Union
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        hps_model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device="cuda",
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        checkpoint = torch.load("hpsv2/HPS_v2_compressed.pt", map_location="cpu")
        hps_model.load_state_dict(checkpoint["state_dict"])
        tokenizer = get_tokenizer("ViT-H-14")
        hps_model = hps_model.to("cuda")
        hps_model.eval()

    if context_reward:
        from lavis.models import load_model_and_preprocess

        context_model, blip_vis_processors, blip_text_processors = (
            load_model_and_preprocess(
                "blip2_image_text_matching", "pretrain", device="cuda", is_eval=True
            )
        )

    # ########################### end model creation stuff ############################################

    if args.lr_image > 0.0:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        pipe_hummingbird.image_encoder = get_peft_model(
            pipe_hummingbird.image_encoder, lora_config
        )
        pipe_hummingbird.image_encoder.print_trainable_parameters()
        pipe_hummingbird.__setattr__(
            "image_encoder_origin", copy.deepcopy(pipe_hummingbird.image_encoder)
        )
        pipe_hummingbird.image_encoder_origin = (
            pipe_hummingbird.image_encoder_origin.to("cuda")
        )
        pipe_hummingbird.image_encoder_origin.eval()
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["to_q", "to_v", "query", "value", "ff.net.0.proj"],
            bias="none",
        )
        pipe_hummingbird.unet = get_peft_model(pipe_hummingbird.unet, lora_config)
        pipe_hummingbird.unet.print_trainable_parameters()

    pipe_hummingbird = pipe_hummingbird.to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    num_channels_latents = pipe_hummingbird.unet.config.in_channels
    height = (
        pipe_hummingbird.unet.config.sample_size * pipe_hummingbird.vae_scale_factor
    )
    width = (
        pipe_hummingbird.unet.config.sample_size * pipe_hummingbird.vae_scale_factor
    )
    guidance_scale = 15

    if args.lr_image > 0.0:
        pipe_hummingbird.image_encoder = torch.nn.parallel.DistributedDataParallel(
            pipe_hummingbird.image_encoder,
            device_ids=[args.gpu],
            broadcast_buffers=False,
        )
        pipe_hummingbird.image_encoder.train()
    else:
        pipe_hummingbird.image_encoder.eval()

    if args.lr_unet > 0.0:
        pipe_hummingbird.unet = torch.nn.parallel.DistributedDataParallel(
            pipe_hummingbird.unet, device_ids=[args.gpu], broadcast_buffers=False
        )
        pipe_hummingbird.unet.train()
    else:
        pipe_hummingbird.unet.eval()

    pipe_hummingbird.vae.eval()
    pipe_hummingbird.vae.requires_grad_(False)
    pipe_hummingbird.text_encoder.eval()
    pipe_hummingbird.text_encoder.requires_grad_(False)
    pipe_hummingbird.text_encoder_2.eval()
    pipe_hummingbird.text_encoder_2.requires_grad_(False)
    pipe_hummingbird.image_encoder.eval()
    pipe_hummingbird.image_encoder.requires_grad_(False)

    meters = {}
    if clip_reward:
        meters["loss_clip_meter"] = AverageMeter()
    if aes_reward:
        meters["loss_aes_meter"] = AverageMeter()
    if hps_reward:
        meters["loss_hps_meter"] = AverageMeter()
    if pick_reward:
        meters["loss_pick_meter"] = AverageMeter()
    if context_reward:
        meters["loss_context_meter"] = AverageMeter()

    image_encoder_params = sum(
        p.numel() for p in pipe_hummingbird.image_encoder.parameters()
    )
    unet_params = sum(p.numel() for p in pipe_hummingbird.unet.parameters())
    print(
        f"Number of params: Image encoder: {image_encoder_params}, Unet: {unet_params}"
    )
    optimizer_params = []
    if args.lr_unet > 0.0:
        optimizer_params.append(
            {"params": pipe_hummingbird.unet.parameters(), "lr": args.lr_unet}
        )
    if args.lr_image > 0.0:
        optimizer_params.append(
            {
                "params": pipe_hummingbird.image_encoder.parameters(),
                "lr": args.lr_image,
            }
        )
    optimizer = optim.AdamW(optimizer_params)

    def count_trainable_params(model):
        print(sum(p.numel() for p in model.vae.parameters() if p.requires_grad))
        print(
            sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
        )
        print(
            sum(p.numel() for p in model.text_encoder_2.parameters() if p.requires_grad)
        )
        print(sum(p.numel() for p in model.unet.parameters() if p.requires_grad))
        print(
            sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad)
        )

    scaler = torch.cuda.amp.GradScaler()
    if args.dataset == "hoi":
        inputs_test = processor(
            [instruction_prompts["hoi"].format("person", "sheep", "")],
            images=test_image,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
    else:
        inputs_test = processor(
            [
                instruction_prompts["vqav2"].format(
                    "Are there six people appear in this image?", ""
                )
            ],
            images=test_image,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

    texts_test = vlm.generate(
        **inputs_test, max_new_tokens=75, pad_token_id=processor.tokenizer.pad_token_id
    )
    texts_test = processor.batch_decode(texts_test, skip_special_tokens=True)
    texts_test = [text.split(" [/INST]")[1].strip() for text in texts_test]
    print(texts_test)

    def custom(module):
        def custom_forward(
            sample,
            timestep,
            encoder_hidden_states,
            class_labels=None,
            timestep_cond=None,
            attention_mask=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            down_block_additional_residuals=None,
            mid_block_additional_residual=None,
            down_intrablock_additional_residuals=None,
            encoder_attention_mask=None,
            return_dict=True,
        ):
            return module(
                sample,
                timestep,
                encoder_hidden_states,
                class_labels=class_labels,
                timestep_cond=timestep_cond,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                down_intrablock_additional_residuals=down_intrablock_additional_residuals,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=return_dict,
            ).sample

        return custom_forward

    # ############ Training starts !!! ###############
    best_loss = 10.0
    accumulation_steps = 16
    iteration = 0
    device = pipe_hummingbird._execution_device
    num_images_per_prompt = 1

    if args.dataset == "imagenet":
        print("Extracting text features of imagenet classes...")
        class_feature = {}
        for class_name in tqdm(image_dataset.class_dict.values()):
            class_feature[class_name] = blip_text_processors["eval"](
                f"a photo of {class_name}"
            )

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print("epoch: ", epoch)
        for batch in dataloader:
            if args.dataset == "hoi":
                (
                    query_images,
                    query_labels,
                    query_prompts,
                    query_category,
                    subject_class,
                    object_class,
                    act_class,
                    _
                ) = batch[0]
            else:
                query_images, query_labels, query_prompts, query_category, _ = batch[0]
            for image_i in range(len(query_images)):
                input_image = [query_images[image_i]]
                batch_size = 1
                if args.dataset == "hoi":
                    instruct_prompt = [
                        instruction_prompts[query_category]
                        .format(subject_class, object_class, query_prompts[image_i])
                        .strip()
                    ] * batch_size
                else:
                    instruct_prompt = [
                        instruction_prompts[query_category]
                        .format(query_prompts, query_labels)
                        .strip()
                    ] * batch_size

                inputs = processor(
                    instruct_prompt,
                    images=input_image,
                    return_tensors="pt",
                    padding=True,
                ).to("cuda")
                texts = vlm.generate(
                    **inputs,
                    max_new_tokens=75,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                texts = processor.batch_decode(texts, skip_special_tokens=True)
                texts = [text.split(" [/INST]")[1].strip() for text in texts]

                input_image = torch.stack([transform(item) for item in input_image])

                with torch.autocast(
                    device_type="cuda", dtype=torch.float32, enabled=True
                ):
                    if args.lr_image > 0.0:
                        use_origin = True
                    else:
                        use_origin = False

                    with torch.no_grad():
                        (
                            prompt_embeds,
                            negative_prompt_embeds,
                            pooled_prompt_embeds_origin,
                            negative_pooled_prompt_embeds_origin,
                        ) = pipe_hummingbird.encode_custom(
                            prompt=texts,
                            image=input_image,
                            device=args.gpu,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=True,
                            use_origin=use_origin,
                        )
                    if args.lr_image > 0.0:
                        (pooled_prompt_embeds, negative_pooled_prompt_embeds) = (
                            pipe_hummingbird.encode_image(
                                input_image, device=args.gpu, num_images_per_prompt=1
                            )
                        )
                    else:
                        pooled_prompt_embeds = pooled_prompt_embeds_origin
                        negative_pooled_prompt_embeds = (
                            negative_pooled_prompt_embeds_origin
                        )

                    generator_s = torch.Generator("cuda").manual_seed(93)
                    latents = pipe_hummingbird.prepare_latents(
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        prompt_embeds.dtype,
                        pipe_hummingbird._execution_device,
                        generator_s,
                    )

                    generator_s = torch.Generator("cuda").manual_seed(93)
                    extra_step_kwargs = pipe_hummingbird.prepare_extra_step_kwargs(
                        generator_s, eta=0.0
                    )

                    add_text_embeds = pooled_prompt_embeds
                    add_text_embeds_origin = pooled_prompt_embeds_origin

                    text_encoder_projection_dim = (
                        pipe_hummingbird.text_encoder_2.config.projection_dim
                    )

                    add_time_ids = pipe_hummingbird._get_add_time_ids(
                        (1024, 1024),
                        (0, 0),
                        (1024, 1024),
                        dtype=prompt_embeds.dtype,
                        text_encoder_projection_dim=text_encoder_projection_dim,
                    )
                    negative_add_time_ids = add_time_ids

                    prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    )
                    add_text_embeds = torch.cat(
                        [negative_pooled_prompt_embeds, add_text_embeds], dim=0
                    )
                    add_text_embeds_origin = torch.cat(
                        [negative_pooled_prompt_embeds_origin, add_text_embeds_origin],
                        dim=0,
                    )
                    add_time_ids = torch.cat(
                        [negative_add_time_ids, add_time_ids], dim=0
                    )

                    prompt_embeds = prompt_embeds.to(
                        device
                    )  # * pipe_imagecraftor.prompt_scale.exp()
                    add_text_embeds = add_text_embeds.to(device)
                    add_text_embeds_origin = add_text_embeds_origin.to(device)
                    add_time_ids = add_time_ids.to(device).repeat(
                        batch_size * num_images_per_prompt, 1
                    )

                    # 8. Denoising loop
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }
                    added_cond_kwargs_origin = {
                        "text_embeds": add_text_embeds_origin,
                        "time_ids": add_time_ids,
                    }

                    for i, t in enumerate(noise_scheduler.timesteps):
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = noise_scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        if i < 25 - args.grad_steps:
                            with torch.no_grad():
                                noise_pred = pipe_hummingbird.unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=prompt_embeds,
                                    added_cond_kwargs=added_cond_kwargs_origin,
                                ).sample

                        else:
                            noise_pred = torch.utils.checkpoint.checkpoint(
                                custom(pipe_hummingbird.unet),
                                latent_model_input,  # sample
                                t,  # timestep
                                prompt_embeds,  # encoder_hidden_states
                                None,  # class_labels (optional)
                                None,  # timestep_cond (optional)
                                None,  # attention_mask (optional)
                                None,  # cross_attention_kwargs (optional)
                                added_cond_kwargs,  # added_cond_kwargs
                                None,  # down_block_additional_residuals (optional)
                                None,  # mid_block_additional_residual (optional)
                                None,  # down_intrablock_additional_residuals (optional)
                                None,  # encoder_attention_mask (optional)
                                True,  # return_dict
                                use_reentrant=False,
                            )
                        # perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = noise_scheduler.step(
                            noise_pred, t, latents, **extra_step_kwargs
                        ).prev_sample

                    image = pipe_hummingbird.vae.decode(
                        latents / pipe_hummingbird.vae.config.scaling_factor
                    ).sample
                    image = (image / 2 + 0.5).clamp(0, 1)

                    text_input_ids = tokenizer(
                        texts,
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to("cuda")

                    input_image = input_image.to("cuda")
                    losses = []

                    # Context loss ##########################################################################################
                    if context_reward:
                        image_context = standard_process(image)
                        if args.dataset == "imagenet":
                            txt_context = class_feature[query_labels]
                        else:
                            txt_context = blip_text_processors["eval"](texts[0])
                        itc_score = context_model(
                            {"image": image_context, "text_input": txt_context},
                            match_head="itc",
                        )

                        itm_score = context_model(
                            {"image": image_context, "text_input": txt_context},
                            match_head="itm",
                        )
                        itm_score = torch.nn.functional.softmax(itm_score, dim=1)[:, 1]

                        loss_context = -100.0 * args.context * (itc_score + itm_score)
                        losses.append(loss_context)
                        meters["loss_context_meter"].update(
                            loss_context.item() / (-100.0 * args.context)
                        )

                    # clip constraint ###################################################################################
                    if clip_reward:
                        image_clip = standard_process(image)
                        image_clip_features = clip_model.encode_image(image_clip)
                        with torch.no_grad():
                            origin_image_features = clip_model.encode_image(input_image)
                        loss_clip = (
                            -100.0
                            * args.clip
                            * torch.mean(
                                torch.sum(
                                    (
                                        image_clip_features
                                        / image_clip_features.norm(dim=-1, keepdim=True)
                                    )
                                    * (
                                        origin_image_features
                                        / origin_image_features.norm(
                                            dim=-1, keepdim=True
                                        )
                                    ),
                                    dim=1,
                                )
                            )
                        )
                        losses.append(loss_clip)
                        meters["loss_clip_meter"].update(
                            loss_clip.item() / (-100.0 * args.clip)
                        )

                    # pick score loss #################################################################################
                    if pick_reward:
                        image_pick = standard_process(image)
                        image_embs = pick_model.get_image_features(
                            pixel_values=image_pick
                        )
                        image_embs = image_embs / torch.norm(
                            image_embs, dim=-1, keepdim=True
                        )
                        with torch.no_grad():
                            text_embs = pick_model.get_text_features(
                                input_ids=text_input_ids
                            )
                            text_embs = text_embs / torch.norm(
                                text_embs, dim=-1, keepdim=True
                            )
                        # score
                        scores = pick_model.logit_scale.exp() * (
                            text_embs @ image_embs.T
                        )
                        loss_pick = -1.0 * args.pickscore * torch.mean(scores)
                        losses.append(loss_pick)
                        meters["loss_pick_meter"].update(
                            loss_pick.item() / (-1.0 * args.pickscore)
                        )

                    # hpsv2 score loss ################################################################################
                    if hps_reward:
                        image_hps = standard_process(image)
                        outputs = hps_model(image_hps, text_input_ids)
                        image_features, text_features = (
                            outputs["image_features"],
                            outputs["text_features"],
                        )
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image)
                        loss_hps = -100.0 * args.hpsv2 * torch.mean(hps_score)
                        losses.append(loss_hps)
                        meters["loss_hps_meter"].update(
                            loss_hps.item() / (-100.0 * args.hpsv2)
                        )

                    # aesthetic score loss ############################################################################
                    if aes_reward:
                        image_aes = standard_process(image)
                        image_features = model_aes["clip_model"].encode_image(image_aes)
                        im_emb = image_features / torch.linalg.norm(
                            image_features, ord=2, dim=-1, keepdim=True
                        )
                        prediction = model_aes["classifier"](im_emb)
                        loss_aes = -3.0 * args.aesthetic * torch.mean(prediction)
                        losses.append(loss_aes)
                        meters["loss_aes_meter"].update(
                            loss_aes.item() / (-3.0 * args.aesthetic)
                        )

                    # #################################################################################################

                loss = sum(losses) / accumulation_steps
                scaler.scale(loss).backward()

                if (iteration + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    # sync_prompt_scale(pipe_imagecraftor)
                    optimizer.zero_grad()
                    """ >>> gradient clipping >>> """
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(pipeline.text_encoder.parameters(), 0.1)
                    """ <<< gradient clipping <<< """
                del (
                    input_image,
                    latents,
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds_origin,
                    negative_pooled_prompt_embeds_origin,
                    added_cond_kwargs,
                    added_cond_kwargs_origin,
                    noise_pred,
                    image,
                    text_input_ids,
                )
                torch.cuda.empty_cache()
                # ##################### rest is logging, validation and saving stuff ##################################
                if iteration % 10 == 0 and args.local_rank == 0:
                    print(
                        "iteration: ",
                        iteration,
                        "rewards: ",
                        [f"{item}: {meters[item].avg}" for item in meters],
                    )
                    print("Prompt to LLaVA: ", instruct_prompt)
                    print("Prompt to SDXL: ", texts)
                if iteration % 100 == 0 and args.local_rank == 0:
                    pipe_hummingbird.image_encoder.eval()
                    pipe_hummingbird.unet.eval()
                    generator_s = torch.Generator("cuda").manual_seed(93)
                    image = pipe_hummingbird(
                        prompt=texts_test,
                        image=test_image,
                        num_inference_steps=25,
                        generator=generator_s,
                        guidance_scale=15,
                        grad_steps=args.grad_steps,
                        use_origin=use_origin,
                    ).images[0]
                    image.save(
                        os.path.join(
                            args.output_path,
                            "count_mme_"
                            + str(epoch)
                            + "_iter_"
                            + str(iteration)
                            + ".png",
                        )
                    )

                    if args.lr_image > 0.0:
                        pipe_hummingbird.image_encoder.train()
                    if args.lr_unet > 0.0:
                        pipe_hummingbird.unet.train()

                if iteration % 1000 == 0 and args.local_rank == 0:
                    if args.lr_image > 0.0:
                        unwarpped_image_encoder = pipe_hummingbird.image_encoder.module
                        unwarpped_image_encoder.save_pretrained(
                            os.path.join(args.output_path, f"lora_image_{iteration}")
                        )
                    else:
                        unwarpped_unet = pipe_hummingbird.unet.module
                        unwarpped_unet.save_pretrained(
                            os.path.join(args.output_path, f"lora_unet_{iteration}")
                        )


                iteration += 1

            del query_images, query_labels
            torch.cuda.empty_cache()
        if meters["loss_context_meter"].avg < best_loss:
            best_loss = meters["loss_context_meter"].avg
            if args.lr_image > 0.0:
                unwarpped_image_encoder = pipe_hummingbird.image_encoder.module
                unwarpped_image_encoder.save_pretrained(
                    os.path.join(args.output_path, "lora_image_best")
                )
            else:
                unwarpped_unet = pipe_hummingbird.unet.module
                unwarpped_unet.save_pretrained(
                    os.path.join(args.output_path, "lora_unet_best")
                )

        for meter in meters:
            meters[meter].reset()


if __name__ == "__main__":
    main()

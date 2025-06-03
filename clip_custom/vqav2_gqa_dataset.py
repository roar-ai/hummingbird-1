import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import random


class VQAv2GQADataset(Dataset):
    def __init__(
        self,
        transform=None,
    ):
        self.transform = transform
        self.ds = load_dataset("lmms-lab/VQAv2")[
            "validation"
        ]

        self.task_list = []
        for i in tqdm(range(len(self.ds))):
            task_data = {"row": i, "dataset": "vqav2"}
            self.task_list.append(task_data)

        self.gqa_ds = load_dataset(
            "lmms-lab/GQA",
            "train_balanced_instructions",
        )["train"]

        self.gqa_images = load_dataset(
            "lmms-lab/GQA",
            "train_balanced_images",
        )["train"]
        self.image_dict = {}
        for i in tqdm(range(len(self.gqa_images))):
            image_id, image = self.gqa_images[i]["id"], self.gqa_images[i]["image"]
            self.image_dict[image_id] = image

        for i in tqdm(range(len(self.gqa_ds))):
            task_data = {"row": i, "dataset": "gqa"}
            self.task_list.append(task_data)

        random.shuffle(self.task_list)

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        task_data = self.task_list[idx]
        row, dataset = task_data["row"], task_data["dataset"]
        if dataset == "vqav2":
            image = self.ds[row]["image"].convert("RGB")
            answer = self.ds[row]["multiple_choice_answer"]
            question = self.ds[row]["question"]

        elif dataset == "gqa":
            image_id = self.gqa_ds[row]["imageId"]
            image = self.image_dict[image_id].convert("RGB")
            question = self.gqa_ds[row]["question"]
            answer = self.gqa_ds[row]["fullAnswer"].replace(".", "")

        if self.transform:
            image = self.transform(image)

        return (
            [image],
            answer,
            question,
            "vqav2",
            None,
        )

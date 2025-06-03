import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


class MMEDataset(Dataset):
    def __init__(
        self,
        transform=None,
    ):
        self.transform = transform
        self.ds = load_dataset("lmms-lab/MME")["test"]

        self.task_list = []
        for i in tqdm(range(0, len(self.ds))):

            if self.ds[i]["category"] in [
                "count",
                "color",
                "existence",
                "position",
                "scene",
                # "artwork",
                # "commonsense_reasoning",
                # "numerical_calculation",
            ]:
                task_data = {}
                task_data["question_id"] = self.ds[i]["question_id"]
                task_data["image"] = self.ds[i]["image"]
                task_data["prompts"] = self.ds[i]["question"].split("?")[0] + "?"
                task_data["category"] = self.ds[i]["category"]
                task_data["answer"] = self.ds[i]["answer"]
                self.task_list.append(task_data)

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        task = self.task_list[idx]
        image = task["image"]
        if self.transform:
            image = self.transform(image)

        return [image], task["answer"], task["prompts"], "vqav2", task["question_id"]

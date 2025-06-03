import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm


class VQAv2Dataset(Dataset):
    def __init__(
        self,
        transform=None,
    ):
        self.transform = transform
        self.ds = load_dataset("lmms-lab/VQAv2")["validation"]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        image = self.ds[idx]["image"].convert("RGB")
        prompt = self.ds[idx]["question"]
        question_type = self.ds[idx]["question_type"]
        if self.transform:
            image = self.transform(image)

        return [image], self.ds[idx]["multiple_choice_answer"], prompt, "vqav2", question_type

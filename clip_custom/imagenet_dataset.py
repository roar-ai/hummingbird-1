import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from glob2 import glob


class ImageNetDataset(Dataset):
    def __init__(
        self,
        data_root=None,
        transform=None,
    ):
        self.transform = transform
        self.classname_path = os.path.join(data_root, "classnames.txt")
        with open(self.classname_path, "r") as f:
            classnames = f.readlines()
        f.close()
        self.class_dict = {}
        self.num_shot = 16
        for classname in classnames:
            classname = classname.replace("\n", "")
            key = classname[:9]
            value = classname[10:]
            self.class_dict[key] = value
        self.ds = []
        for class_id in self.class_dict.keys():
            folder_path = os.path.join(data_root, class_id)
            list_imgs = glob(folder_path + "/*.JPEG")[: self.num_shot]
            self.ds.extend(list_imgs)

        assert len(self.ds) == 1000 * self.num_shot

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = Image.open(self.ds[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_dict[self.ds[idx].split("/")[-2]]

        return [image], label, "What is this?", "imagenet", os.path.join(*self.ds[idx].split("/")[-2:])


class ImageNetSketchDataset(Dataset):
    def __init__(
        self,
        data_root=None,
        transform=None,
    ):
        self.transform = transform
        self.classname_path = os.path.join(data_root, "classnames.txt")
        with open(self.classname_path, "r") as f:
            classnames = f.readlines()
        f.close()
        self.class_dict = {}
        for classname in classnames:
            classname = classname.replace("\n", "")
            key = classname[:9]
            value = classname[10:]
            self.class_dict[key] = value
        self.ds = []
        for class_id in self.class_dict.keys():
            folder_path = os.path.join(data_root, class_id)
            list_imgs = glob(folder_path + "/*.JPEG")
            self.ds.extend(list_imgs)
        print(len(self.ds))
        self.ds = self.ds[44000:]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = Image.open(self.ds[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_dict[self.ds[idx].split("/")[-2]]

        return [image], label, "What is this?", "imagenet", os.path.join(*self.ds[idx].split("/")[-2:])

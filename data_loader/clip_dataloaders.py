import io
import logging
import os

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger("global")
import json
import os.path as osp
import time

import requests
import torch

from .classes import *

global_root = "/nethome/bdevnani3/flash1/long_tail_lang"
freq_file = (
    "/nethome/bdevnani3/flash1/long_tail_lang/data/ImageNet_LT/freq_classification.json"
)

# Modality Agnostic Long Tail Embedding Dataset
class MALTE_Train_Dataset(Dataset):
    """
    Modality agnostic dataset that treats both text and images as the same
    It can strengthen specific categories based on their frequencies

    """

    def __init__(
        self,
        root,
        image_to_labels,
        text_prompt_indices=-1,
        strengthen_med=True,
        strengthen_low=True,
        strengthen_high=False,
    ):
        self.path = []
        self.labels = []
        self.label_names = []
        self.freq = json.load(open(freq_file))
        with open(image_to_labels, "r") as file:
            for line in file:
                line = line.strip().split(" ")
                self.path.append(line[0])
                self.labels.append(line[1])
                self.label_names.append(line[2])
        for label in range(1000):
            if strengthen_low:
                if int(label) in self.freq["low"]:
                    temp = []
                    for i in range(82):
                        temp.append(
                            f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50/labels/{label}/{i}.pt"
                        )
                    self.path.append(temp)
            if strengthen_med:
                if int(label) in self.freq["med"]:
                    temp = []
                    for i in range(82):
                        temp.append(
                            f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50/labels/{label}/{i}.pt"
                        )
                    self.path.append(temp)
            self.labels.append(label)
            self.label_names = [CLASSES[label]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.path[index]
        label = int(self.labels[index])
        label_name = self.label_names[index]

        sample = torch.load(path, map_location=torch.device("cpu"))
        sample.requires_grad = False

        return sample, label, index, label_name, -1


# Long Tail Embedding Dataset
class LTE_Train_Dataset(Dataset):
    def __init__(self, root, image_to_labels, text_prompt_indices=-1):
        self.img_path = []
        self.labels = []
        self.label_names = []
        self.text_paths = []
        with open(image_to_labels, "r") as file:
            for line in file:
                line = line.strip().split(" ")
                self.img_path.append(line[0])
                self.labels.append(line[1])
                self.label_names.append(line[2])
                if text_prompt_indices == -1:
                    temp = line[3:]
                    self.text_paths.append(line[3:])
                else:
                    self.text_paths.append[[line[3:][i] for i in text_prompt_indices]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with torch.no_grad():

            path = self.img_path[index]
            label = int(self.labels[index])
            label_name = self.label_names[index]

            text_paths = self.text_paths[index]
            le = [torch.load(f, map_location=torch.device("cpu")) for f in text_paths]
            text_embeddings = torch.stack(le).squeeze(1)

            sample = torch.load(path, map_location=torch.device("cpu"))
            sample.requires_grad = False

        return sample, label, index, label_name, text_embeddings


# Eval Embedding Dataset
class LTE_Eval_Dataset(Dataset):
    def __init__(self, root, image_to_labels):
        self.img_path = []
        self.labels = []
        self.label_names = []
        self.text_paths = []
        with open(image_to_labels, "r") as file:
            for line in file:
                line = line.split(" ")
                self.img_path.append(line[0])
                self.labels.append(line[1])
                self.label_names.append(line[2])
                temp = []
                for i in range(1000):
                    temp.append(
                        f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50/labels/{i}/{1}.pt"
                    )
                self.text_paths.append(temp)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        with torch.no_grad():

            path = self.img_path[index]
            label = int(self.labels[index])
            label_name = self.label_names[index]

            text_paths = self.text_paths[index]
            le = [torch.load(f, map_location=torch.device("cpu")) for f in text_paths]
            text_embeddings = torch.stack(le).squeeze(1)

            sample = torch.load(path, map_location=torch.device("cpu"))
            sample.requires_grad = False

        return sample, label, index, label_name, text_embeddings


# Load datasets
def load_data(
    data_root,
    phase,
    batch_size,
    num_workers=4,
    shuffle=True,
    text_prompt_indices=-1,
    var="paired",
):

    image_to_labels = f"/nethome/bdevnani3/flash1/long_tail_lang/datasets/ImageNet_emb/RN50/labels/{phase}/image_to_label.txt"

    print(f"DataLoader Configs: {phase}, {var}")

    if phase == "train":
        if var == "paired":
            set_ = LTE_Train_Dataset(
                data_root, image_to_labels, text_prompt_indices=text_prompt_indices
            )
        else:
            set_ = MALTE_Train_Dataset(
                data_root, image_to_labels, text_prompt_indices=text_prompt_indices
            )
    elif phase == "val" or phase == "test":
        set_ = LTE_Train_Dataset(data_root, image_to_labels)
    print(len(set_))

    print("No sampler.")
    print("Shuffle is %s." % (shuffle))

    out = DataLoader(
        dataset=set_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    print("Data Loaded")
    return out

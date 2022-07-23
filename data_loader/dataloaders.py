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

global_root = "/nethome/bdevnani3/flash1/long_tail_lang"


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert("RGB")
    except IOError:
        logger.info("Failed in loading {}".format(filepath))
    return img


# Image statistics
RGB_statistics = {
    "default": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key="default"):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=224, scale=(0.5, 1), interpolation=BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]
        ),
    }
    return data_transforms[split]


# Dataset
class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        # self.lt_count = {"low": 0, "medium":0, "high": 0}
        # for k,v in freqs.items():
        #     if int(v) <= 20:
        #         self.count["low"] +=1
        #     elif int(v) <= 100:
        #         self.count["medium"] +=1
        #     else:
        #         self.count["high"] +=1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        with open(path, "rb") as f:
            sample = Image.open(f).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index, path


# Dataset
class LT_Dataset_boosted(Dataset):
    def __init__(self, root, txt, transform=None):

        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        for i in range(1000):
            if freqs[i] <= 100:
                for j in range(82):
                    self.img_path.append(j)
                    self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        if isinstance(path, int):
            sample = self.totensor(np.zeros((224, 224, 3)))
        else:
            with open(path, "rb") as f:
                sample = Image.open(f).convert("RGB")
                path = 1

            if self.transform is not None:
                sample = self.transform(sample)

        return sample, label, index, path


# Dataset
class LT_Dataset_boosted_wiki(Dataset):
    def __init__(self, root, txt, transform=None):

        self.totensor = transforms.ToTensor()
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        freqs = {}
        for l in self.labels:
            l == int(l)
            if l not in freqs:
                freqs[l] = 0
            freqs[l] += 1

        self.lt_count = freqs

        # self.lt_count = {"low": 0, "medium":0, "high": 0}
        # for k,v in freqs.items():
        #     if int(v) <= 20:
        #         self.count["low"] +=1
        #     elif int(v) <= 100:
        #         self.count["medium"] +=1
        #     else:
        #         self.count["high"] +=1

        for i in range(1000):
            if freqs[i] <= 100:
                for j in range(82):
                    self.img_path.append(j)
                    self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]

        if isinstance(path, int):
            sample = self.totensor(np.zeros((224, 224, 3)))
        else:
            with open(path, "rb") as f:
                sample = Image.open(f).convert("RGB")
                path = 1

            if self.transform is not None:
                sample = self.transform(sample)

        return sample, label, index, path


# Load datasets
def load_data(
    data_root,
    dataset,
    phase,
    batch_size,
    sampler_dic=None,
    num_workers=4,
    test_open=False,
    shuffle=True,
    boosted=False,
):

    if phase == "train_plain":
        txt_split = "train"
    elif phase == "train_val":
        txt_split = "val"
        phase = "train"
    else:
        txt_split = phase
    txt = "%s/data/%s/%s_%s.txt" % (global_root, dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print("Loading data from %s" % (txt))

    key = "clip"
    rgb_mean, rgb_std = RGB_statistics[key]["mean"], RGB_statistics[key]["std"]

    if phase not in ["train", "val"]:
        transform = get_data_transform("test", rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)

    print("Use data transformation:", transform)

    set_ = LT_Dataset(data_root, txt, transform)

    if phase == "train" and boosted == True:
        set_ = LT_Dataset_boosted(data_root, txt, transform)

    lt_count = set_.lt_count

    print(len(set_))
    if phase == "test" and test_open:
        open_txt = "%s/data/%s/%s_open.txt" % (global_root, dataset, dataset)
        print("Testing with opensets from %s" % (open_txt))
        open_set_ = LT_Dataset(
            "./data/%s/%s_open" % (dataset, dataset), open_txt, transform
        )
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == "train":
        print("Using sampler: ", sampler_dic["sampler"])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print("Sampler parameters: ", sampler_dic["params"])
        return (
            DataLoader(
                dataset=set_,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler_dic["sampler"](set_, **sampler_dic["params"]),
                num_workers=num_workers,
            ),
            lt_count,
        )
    else:
        print("No sampler.")
        print("Shuffle is %s." % (shuffle))
        return (
            DataLoader(
                dataset=set_,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            lt_count,
        )

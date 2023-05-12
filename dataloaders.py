from cv2 import transform
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
import PIL
from PIL import Image
import io
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random


def positional_encoding(pos, n_dim, max_length: int = 10000):
    # batch = len(pos)
    pe = torch.zeros(n_dim)
    for i in range(0, n_dim, 2):
        pe[i] = math.sin(pos / (max_length ** ((2 * i) / n_dim)))
        pe[i + 1] = math.cos(pos / (max_length ** ((2 * (i + 1)) / n_dim)))
    return pe  # .to("cuda:0")


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        train=True,
        positional_encoding=False,
        split="train",
    ):

        self.data_dir = data_dir
        self.transform = transform
        self.positional_encoding = positional_encoding
        self.train = train
        self.split = split
        self.label_path = os.path.join(f"{self.split}.txt")
        with open(self.label_path, "r") as f:
            self.lines = f.readlines()
        if self.split == "train" or "val":
            random.shuffle(self.lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx]

        img_path = os.path.join(
            self.data_dir,
            line.split("---")[0].split("Frame")[0],
            line.split("---")[0],
        )

        image = PIL.Image.open(img_path)
        label = int(line.split("---")[1])
        video_length = int(line.split("---")[2])
        fps = int(line.split("---")[3])
        frame_num = img_path.split("/")[-1].split(".")[0].split("Frame")[1]

        # #return  {'images': self.transform(images[5]), 'label':labels[5], 'path': img_paths}
        pe = torch.zeros(1536)
        # divide pos by frame rate - add an argument for video length
        if self.positional_encoding:
            pe = positional_encoding(
                pos=int(frame_num / fps),
                n_dim=1536,
                max_length=video_length,
            )

        return self.transform(image), label, pe, img_path  # [5]


class VideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        train=True,
        positional_encoding=False,
        split="train",
        clip_length=16,
    ):

        self.data_dir = data_dir
        self.transform = transform
        self.positional_encoding = positional_encoding
        self.train = train
        self.split = split
        self.label_path = os.path.join(f"./data/phase/{self.split}.txt")
        with open(self.label_path, "r") as f:
            self.lines = f.readlines()
        if self.split == "train" or "val":
            random.shuffle(self.lines)

        self.clip_length = clip_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx]

        img_path = os.path.join(
            self.data_dir,
            line.split("---")[0].split("Frame")[0],
            line.split("---")[0],
        )

        # image = PIL.Image.open(img_path)
        label = int(line.split("---")[1])

        # fps = int(line.split("---")[3])
        frame_num = img_path.split("/")[-1].split(".")[0].split("Frame")[1]

        images = []
        for i in range(self.clip_length - 1, 0, -1):

            try:
                fps = 30
                frame_num_ = str(int(frame_num) - int((fps * i) / 5)).zfill(5)
                if int(frame_num_) < 0:
                    frame_num_ = "00000"

                images.append(
                    torchvision.io.read_image(
                        f'{img_path.split("Frame")[0]}Frame{frame_num_}.jpg'
                    )
                )
            except:
                fps = 25
                frame_num_ = str(int(frame_num) - int((fps * i) / 5)).zfill(5)
                if int(frame_num_) < 0:
                    frame_num_ = "00000"

                images.append(
                    torchvision.io.read_image(
                        f'{img_path.split("Frame")[0]}Frame{frame_num_}.jpg'
                    )
                )

        images.append(torchvision.io.read_image(img_path))

        images_tensor = torch.stack(images)  # T, C, H, W

        # images_tensor.to("cuda")

        scripted_transforms = torch.jit.script(self.transform)

        images = scripted_transforms(torch.squeeze(images_tensor))

        pe = torch.zeros(2048)
        if self.positional_encoding:
            pe = positional_encoding(
                pos=int(int(frame_num) / 25),
                n_dim=2048,
                max_length=6000,
            )

        return torch.permute(images, (1, 0, 2, 3)), label, pe, img_path

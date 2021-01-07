import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import labelme
import cv2

train_label_pathIn = "/home/mislab/LiYulin/Research/dataset/Flower/train/annotation/"
train_img_pathIn = "/home/mislab/LiYulin/Research/dataset/Flower/train/image/"


class TrainDataset():
    def __init__(self, image_dir, label_dir, transform=None):
        """self.img_files = glob.glob(osp.join(image_dir, "*.jpg"))
        self.transform = transform"""
        label_files = glob.glob(osp.join(label_dir, "*.json"))
        self.img_files = glob.glob(osp.join(image_dir, "*.jpg"))
        self.points = []
        self.transform = transform
        for image_id, filename in enumerate(label_files):
            self.points.append(labelme.LabelFile(filename=filename))

    def __getitem__(self, idx):
        """img = Image.open(self.img_files[idx])
        if self.transform:
            transform = transforms.RandomAffine(
                0, translate=(0.3, 0.3), scale=None, shear=None, resample=0, fillcolor=0)
            img = transform(img)
        img = np.array(img)
        img = img[:, :, ::-1].copy()

        sample = {'img': img}
        return sample"""
        img = cv2.imread(self.img_files[idx])
        for shape in self.points[idx].shapes:
            points = shape["points"]
        sample = {'img': img, 'pts': points}
        return sample

    def __len__(self):
        return len(self.img_files)


class TestDataset():
    def __init__(self, image_dir, label_dir, transform=None):
        label_files = glob.glob(osp.join(label_dir, "*.json"))
        self.img_files = glob.glob(osp.join(image_dir, "*.jpg"))
        self.points = []
        self.transform = transform
        for image_id, filename in enumerate(label_files):
            self.points.append(labelme.LabelFile(filename=filename))

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        for shape in self.points[idx].shapes:
            points = shape["points"]
        if self.transform:
            transform = transforms.RandomAffine(
                0, translate=(0.2, 0.2), scale=None, shear=None, resample=0, fillcolor=0)
            img = transform(img)
        img = np.array(img)
        img = img[:, :, ::-1].copy()
        sample = {'img': img, 'pts': points}
        return sample

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    dataset = FlowerDataset(train_img_pathIn, train_label_pathIn)
    i = 60
    print(dataset.len())
    img, points = dataset.get(i)
    print(img.shape)
    print(points)

from os import path, listdir
from pathlib import Path
import random
import sys
import cv2
import numpy as np
import torch
_ROOT = str(Path(__file__).resolve().parents[1])
if not _ROOT in sys.path:
    sys.path.append(_ROOT)
from utils.augment import horizontal_flip, random_crop, random_narrow, hsv_jitter

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class Dataset():
    exts = {".bmp", ".jpg", ".jpeg", ".png"}

    def __init__(self, data_path: str, imgsz: list[int], aug: bool=False):
        if not path.exists(data_path):
            raise FileNotFoundError(data_path)
        self.imgsz = imgsz
        self.aug = aug
        self.images_dir = self.labels_dir = ""
        self.images_exts = dict()
        self.data_list = []
        # darknet yolo format
        if data_path.strip().endswith(".txt"):
            with open(data_path) as f:
                self.images_dir = self.labels_dir = path.split(f.readline())[0]
                f.seek(0)
                for l in f:
                    l = l.strip()
                    if not l:
                        continue
                    n, e = path.splitext(l)
                    if e.lower() in self.exts and path.exists(l) and \
                        path.exists(f"{path.splitext(l)[0]}.txt"):
                        self.data_list.append(n)
                        self.images_exts[n] = e
        # yolo format
        else:
            self.images_dir = data_path
            base, split = path.split(data_path)
            self.labels_dir = path.join(path.split(base)[0], "labels", split)
            images_set, labels_set = set(), set()
            for l in listdir(self.images_dir):
                n, e = path.splitext(l)
                if e.lower() in self.exts:
                    images_set.add(n)
                    self.images_exts[n] = e
            for l in listdir(self.labels_dir):
                n, e = path.splitext(l)
                if e.lower() == ".txt":
                    labels_set.add(n)
            self.data_list = list(images_set & labels_set)
        self.data_list.sort()

    def __getitem__(self, index):
        name = self.data_list[index]
        img_path = path.join(self.images_dir, f"{name}{self.images_exts[name]}")
        label_path = path.join(self.labels_dir, f"{name}.txt")
        # load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = np.loadtxt(label_path, dtype=np.float32)
        if label.ndim == 1:
            label = label[None, :]
        label = np.pad(label, ((0, 0), (1, 0)), constant_values=0)
        # augmentation
        if self.aug:
            if random.random() > 0.5:
                img, label = random_narrow(img, label)
            else:
                img, label = random_crop(img, label)
            if random.random() > 0.5:
                img, label = horizontal_flip(img, label)
            img = hsv_jitter(img)
        # resize
        img = cv2.resize(img, self.imgsz, interpolation=cv2.INTER_LINEAR)
        # hwc->chw
        img = img.transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    data = Dataset("/data/datasets/coco-darknet/val2017.txt", imgsz=[352, 352])
    img, label = data.__getitem__(0)
    print(img.shape, label.shape)

    data = Dataset("/data/datasets/coco128/images/train2017", imgsz=[352, 352])
    img, label = data.__getitem__(0)
    print(img.shape, label.shape)

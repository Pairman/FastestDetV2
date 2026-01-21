import os
import cv2
import numpy as np

import torch
import random

def horizontal_flip(image, boxes):
    boxes = boxes.copy()
    boxes[:, 2] = 1 - boxes[:, 2] # flip cx
    return image[:, ::-1], boxes

def random_crop(image, boxes):
    h, w, _ = image.shape
    # random crop
    cw, ch = random.randint(int(w * 0.75), w), random.randint(int(h * 0.75), h)
    cx, cy = random.randint(0, w - cw), random.randint(0, h - ch)
    roi = image[cy:cy + ch, cx:cx + cw]
    roi_h, roi_w, _ = roi.shape
    # bbox transform
    xy = boxes[:, 2:4] * np.array([w, h])
    wh = boxes[:, 4:6] * np.array([w, h])
    xy = (xy - np.array([cx, cy])) / np.array([roi_w, roi_h])
    wh = wh / np.array([roi_w, roi_h])
    out = boxes.copy()
    out[:, 2:4], out[:, 4:6] = xy, wh
    return roi, out

def random_narrow(image, boxes):
    h, w, _ = image.shape
    # random narrow
    cw, ch = random.randint(w, int(w * 1.25)), random.randint(h, int(h * 1.25))
    cx, cy = random.randint(0, cw - w), random.randint(0, ch - h)
    bg = np.ones((ch, cw, 3), np.uint8) * 128
    bg[cy:cy + h, cx:cx + w] = image
    # bbox transform
    xy = boxes[:, 2:4] * np.array([w, h])
    wh = boxes[:, 4:6] * np.array([w, h])
    xy = (xy + np.array([cx, cy])) / np.array([cw, ch])
    wh = wh / np.array([cw, ch])
    out = boxes.copy()
    out[:, 2:4], out[:, 4:6] = xy, wh
    return bg, out

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class Dataset():
    def __init__(self, path, img_width, img_height, aug=False):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self.aug = aug
        self.img_width = img_width
        self.img_height = img_height
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.data_list = []
        for path in lines:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            ext = os.path.splitext(path)[1].lower()
            if ext not in {".bmp", ".jpg", ".jpeg", ".png"}:
                raise ValueError(ext)
            self.data_list.append(path)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label_path = f"{img_path.rsplit('.', 1)[0]}.txt"
        # load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(img_path)
        # load labels
        if not os.path.exists(label_path):
            raise FileNotFoundError(label_path)
        label = np.loadtxt(label_path, dtype=np.float32)
        if label.ndim == 1:
            label = label[None, :]
        label = np.pad(label, ((0,0),(1,0)), constant_values=0) 
        # augmentation
        if self.aug:
            if random.getrandbits(1):
                img, label = random_narrow(img, label)
            else:
                img, label = random_crop(img, label)
            if random.getrandbits(1):
                img, label = horizontal_flip(img, label)
        # resize
        img = cv2.resize(img, (self.img_width, self.img_height),
            interpolation=cv2.INTER_LINEAR) 
        # hwc->chw
        img = img.transpose(2,0,1).astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)

if __name__ == "__main__":
    data = Dataset("/data/datasets/coco-darknet/val2017.txt")
    img, label = data.__getitem__(0)
    print(img.shape, label.shape)

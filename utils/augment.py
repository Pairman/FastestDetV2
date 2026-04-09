import random
import cv2
import numpy as np

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
    # filter out boxes whose center is outside the crop
    mask = (out[:, 2] > 0) & (out[:, 2] < 1) & (out[:, 3] > 0) & (out[:, 3] < 1)
    out = out[mask]
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

def hsv_jitter(image, h_gain=0.1, s_gain=0.1, v_gain=0.15):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

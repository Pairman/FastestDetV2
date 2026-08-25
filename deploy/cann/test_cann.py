import argparse
import cv2
import numpy as np
from ais_bench.infer.interface import InferSession

names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_preds(preds):
    N, C, H, W = preds.shape
    preds = preds.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    pobj = preds[..., 0]
    preg = preds[..., 1:5]
    pcls = preds[..., 5:]
    cls_scores, cls = pcls.max(-1), pcls.argmax(-1)
    conf = (pobj ** 0.6) * (cls_scores ** 0.4)
    gy, gx = np.meshgrid(
        np.arange(H, dtype=preds.dtype),
        np.arange(W, dtype=preds.dtype),
        indexing="ij")
    bcx = (np.tanh(preg[..., 0]) + gx) / W
    bcy = (np.tanh(preg[..., 1]) + gy) / H
    bw = sigmoid(preg[..., 2])
    bh = sigmoid(preg[..., 3])
    x1 = bcx - 0.5 * bw
    y1 = bcy - 0.5 * bh
    x2 = bcx + 0.5 * bw
    y2 = bcy + 0.5 * bh
    boxes = np.stack([x1, y1, x2, y2, conf, cls.astype(preds.dtype)], axis=-1)
    return boxes.reshape(N, H * W, 6)

def nms(boxes, scores, cls, iou_thres=0.45):
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[1:][(iou <= iou_thres) | (cls[order[1:]] != cls[i])]
    return np.array(keep, dtype=int)

def apply_nms(boxes, conf_thres=0.25, iou_thres=0.45):
    out = []
    for p in boxes:
        pb = p[p[:, 4] > conf_thres]
        if len(pb) == 0:
            out.append(np.zeros((0, 6)))
            continue
        keep = nms(pb[:, :4], pb[:, 4], pb[:, 5], iou_thres)
        out.append(pb[keep])
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="fastestdetv2.om", help=".om weights")
    parser.add_argument("--image", type=str, default="input.jpg", help="input image")
    parser.add_argument("--result", type=str, default="output.jpg", help="output image")
    parser.add_argument("--device", type=str, default="npu:0", help="device. npu:0, etc.")
    parser.add_argument("--conf-thres", type=float, default=0.65, help="confidence threshold")
    opt = parser.parse_args()
    device_id = int(opt.device.split(":")[1])
    # load model
    session = InferSession(device_id=device_id, model_path=opt.weights)
    # preproc
    print(f"Processing image {opt.image}")
    img0 = cv2.imread(opt.image)
    img = cv2.resize(img0, (352, 352))
    img = img.transpose(2, 0, 1)[None]  # HWC -> NCHW
    img = img.astype(np.float32) / 255.0  # norm
    # inference
    preds = session.infer([img])[0]  # (1, 5+num_classes, H, W)
    # postproc
    preds = apply_nms(decode_preds(preds), conf_thres=opt.conf_thres)
    # visualize
    h0, w0, _ = img0.shape
    for box in preds[0]:
        conf, cls = box[4], int(box[5])
        x1, y1 = int(box[0] * w0), int(box[1] * h0)
        x2, y2 = int(box[2] * w0), int(box[3] * h0)
        cv2.rectangle(img0, (x1, y1), (x2, y2), (255, 255, 0), 2)
        label = f"{cls} {names[cls]} {conf:.2f}"
        cv2.putText(img0, label, (x1, y1 - 5), 0, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img0, label, (x1 + 1, y1 - 5), 0, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(opt.result, img0)
    print(f"Saved result to {opt.result}")

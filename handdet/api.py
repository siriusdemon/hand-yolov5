import os.path as osp
import time
import cv2
import torch
import numpy as np

from .models import yolo
from .utils.datasets import letterbox
from .utils.general import check_img_size, non_max_suppression, scale_coords

base = osp.dirname(__file__)
names = open(osp.join(base, 'hand_labels.txt'), 'r', encoding='utf-8').read().strip().split()

class opt:
    weights = osp.join(base, 'checkpoints/best_sd.pt')
    cfgfile = osp.join(base, 'models/yolov5s.yaml')
    source = 'None'
    img_size = 640
    conf_thres = 0.35
    iou_thres = 0.45
    device = 'cpu'
    view_img = True
    classes = None
    agnostic_nms = False
    augment = False


model = yolo.Model(opt.cfgfile)
model.load_state_dict(torch.load(opt.weights, map_location=opt.device))
model.float().eval()

def detect(img):
    if type(img) == str:
        img = cv2.imread(img)
    opt.source = img
    return _detect()

def _detect():
    global model
    global names
    source, weights, imgsz = opt.source, opt.weights,  opt.img_size

    # Load model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Run inference
    device = opt.device
    img = letterbox(source, new_shape=opt.img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  #
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = source
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        det = det.numpy().tolist()
        res = []
        for *box, conf, cls_id in det:
            bbox = list(map(int, box))
            cls_name = names[int(cls_id)]
            d = {'class': cls_name, 'bbox': bbox, "conf": conf}
            res.append(d)
        return res

def visualize(im, dets):
    if type(im) == str:
        im = cv2.imread(im)
    im_h, im_w = im.shape[:2]
    for d in dets:
        box = d['bbox']
        left, top = box[0], box[1]
        right, bottom = box[2], box[3]
        cv2.rectangle(im, (left,top), (right,bottom), (0,234,242), 2)
    return im

def crop(impath, dets):
    im = cv2.imread(impath)
    for (i, d) in enumerate(dets):
        box = d['bbox']
        left, top = box[0], box[1]
        right, bottom = box[2], box[3]
        crop = im[top:bottom, left:right]
        cv2.imwrite(f"test_{i}.png", crop)

def demo():
    import sys
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "zz.jpeg"
    dets = detect(name)
    print(dets)
    visualize(name, dets)

def video():
    import sys
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = "test.mp4"
    cam = cv2.VideoCapture(name)
    ret, frame = cam.read()
    while ret:
        t = time.time()
        dets = detect(frame)
        print("detect cost: ", time.time() -t)
        frame = visualize(frame, dets)
        cv2.imshow('camera', frame)
        # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cam.read()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video()
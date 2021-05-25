import os.path as osp
import time
import cv2
import torch
import numpy as np

from .models import yolo
from .utils.datasets import letterbox
from .utils.general import check_img_size, non_max_suppression, scale_coords


# labels and models
base = osp.dirname(__file__)
names = open(osp.join(base, 'hand_labels.txt'), 'r', encoding='utf-8').read().strip().split()

class opt:
    weights = osp.join(base, 'checkpoints/best_sd.pt')
    cfgfile = osp.join(base, 'models/yolov5s.yaml')
    source = 'None'
    img_size = 640
    conf_thres = 0.15
    iou_thres = 0.45
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    view_img = True
    classes = None
    agnostic_nms = False
    augment = False

model = yolo.Model(opt.cfgfile)
model.load_state_dict(torch.load(opt.weights, map_location=opt.device))
model.float().eval()
# ----------------------------------------------------------------------


def load_images(ims):
    new_ims = []
    shapes = []
    for im in ims:
        im = letterbox(im, new_shape=opt.img_size)[0]
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im = np.asarray(im, dtype=np.float32)
        im /= 255.0  #
        new_ims.append(im)
        shapes.append(im.shape[1:])
    new_ims = np.stack(new_ims)
    new_ims = torch.from_numpy(new_ims).to(opt.device)
    return new_ims, shapes

def forward(ims):
    # ims: torch.tensor
    with torch.no_grad():
        preds, _ = model(ims, augment=opt.augment)
    return preds

def postprocess(ims, shapes, preds):
    """Apply NMS and rescale bbox to original size
    Args:
        ims: original image returned by cv2.imread 
        shapes: shape after letterbox
        preds: predictions
    """
    all_results = []
    preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    im0 = ims[0]
    shape = shapes[0]
    for i, det in enumerate(preds):  # detections per image
        res = []
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            det[:, :4] = scale_coords(shape, det[:, :4], im0.shape).round()
        det = det.numpy().tolist()
        for *box, conf, cls_id in det:
            bbox = list(map(int, box))
            cls_name = names[int(cls_id)]
            d = {'class': cls_name, 'bbox': bbox, "conf": conf}
            res.append(d)
        all_results.append(res)
    return all_results


def detect_many(ims):
    # NOTE: ims should have the same shape
    new_ims, shapes = load_images(ims)
    preds = forward(new_ims)
    results = postprocess(ims, shapes, preds)
    return results

def detect(im):
    return detect_many([im])[0]

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
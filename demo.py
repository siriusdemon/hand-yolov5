import os
import os.path as osp
import sys

import cv2
import handdet


def detect_demo(imgs):
    for img in imgs:
        im = cv2.imread(img)
        res = handdet.detect(im)
        rim = handdet.visualize(im, res)
        cv2.imwrite(f"saved{osp.basename(img)}", rim)
    print("Done!")


def crop_demo(img):
    im = cv2.imread(img)
    res = handdet.detect(im)
    crops = handdet.crop(im, res)
    for (i, crop) in enumerate(crops):
        cv2.imwrite(f"crop_{i}.png", crop)

if __name__ == '__main__':
    # guard
    if len(sys.argv) != 2:
        print(
            "Usage: python demo.py path/to/image\n"
            "   or: python demo.py path/to/image-directory"
        )
        os._exit(0)

    # detect
    target = sys.argv[1]
    if osp.isdir(target):
        imgs = [osp.join(target, f) for f in sorted(os.listdir(target))]
    elif osp.isfile(target):
        imgs = [target]

    detect_demo(imgs)
    crop_demo(imgs[0])
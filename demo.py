import os
import os.path as osp
import sys

import cv2
import handdet


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

imlist = []
for img in imgs:
    im = cv2.imread(img)
    imlist.append(im)
    res = handdet.detect(im)
    rim = handdet.visualize(im, res)
    cv2.imwrite(f"saved{osp.basename(img)}", rim)
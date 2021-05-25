import os
import os.path as osp
import sys
import handdet
import decord
import cv2


VIDEO_TYPE = ['mp4', 'MP4', 'avi', "AVI", 'mpeg', "MPEG", 'mkv', "MKV", 'webm', "WEBM"]

def is_video(file):
    ext = file.split('.')[-1]
    return ext in VIDEO_TYPE

def process(video_path):
    """ Steps
    1. read the video
    2. every 2s get 1 frame
    3. create an directory with the name stem of the video
    4. crop hands from frames with a confidence threshold
    5. save hands into the directory
    """
    # Step1
    video = decord.VideoReader(video_path)
    avg_fps = video.get_avg_fps()
    print(f"Read video from {video_path}")
    print(f"Average FPS: {avg_fps}")

    # Step2
    crop_per_second = 0.3           # I mean 1 frame per 2s
    interval = int(avg_fps / crop_per_second)
    frames = video[::interval]
    frames = frames.asnumpy()[:, :, :, ::-1]

    # Step3
    base = "result_crops"
    os.makedirs(base, exist_ok=True)
    crop_dir = osp.join(base, osp.basename(video_path).split('.')[0])
    os.makedirs(crop_dir, exist_ok=True)

    # Step4 + Step5
    def extend_box(box):
        # add more background to enable better classify
        left, top, right, bottom = box
        w = right - left
        h = bottom - top
        left = max(0, left - w // 6)
        top = max(0, top - h // 6)
        right = right + w // 6
        bottom = bottom + h // 6
        return left, top, right, bottom
    # set a mini batch 
    res = []
    batch = 16
    total = len(frames)
    times = total // batch
    if total % batch > 0: times += 1
    for i in range(times):
        s = i * batch
        e = min(s + batch, total)
        frs = frames[s:e]
        rs = handdet.detect_many(frames)
        res += rs

    CONF = 0.3
    for i, (frame, det) in enumerate(zip(frames, res)):
        det = [d for d in det if d['conf'] >= CONF]
        for d in det: d['bbox'] = extend_box(d['bbox'])
        for j, crop in enumerate(handdet.crop(frame, det)):
            crop_path = osp.join(crop_dir, f"crop{j:02}_fr{i:03}.png")
            cv2.imwrite(crop_path, crop)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python demo.py path/to/image\n"
            "   or: python demo.py path/to/image-directory"
        )
        os._exit(0)

    # Step0
    target = sys.argv[1]
    if osp.isfile(target):
        video_paths = [target] 
    elif osp.isdir(target):
        video_paths = [osp.join(target, f) for f in os.listdir(target) if is_video(f)]
    else:
        print(f"{target} is not a directory or a video")
        os._exit(0)
    print(f"Process {len(video_paths)} video")

    for i, video_path in enumerate(video_paths):
        print(f"Process {i+1:2d}/{len(video_paths)}")
        process(video_path)
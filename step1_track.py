"""
STEP 1: Video -> data/trajectories.csv

Finds people (YOLO), follows each one with an ID (ByteTrack) and saves, for every processed frame:
the box around the person and the point where their FEET touch the floor.

Run:  python step1_track.py --video your_video.mp4

Accuracy tips:
  --model yolov8m.pt     bigger model = more accurate but slower (n < s < m < l)
  --imgsz 1280           better for small/far people (e.g. high cameras), slower
  --fps 8                more frames per second = smoother tracking, slower
"""
import argparse
import csv
import os
import time

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to the input video")
    ap.add_argument("--fps", type=float, default=5, help="frames per second to process")
    ap.add_argument("--model", default="yolov8s.pt", help="YOLO model: yolov8n/s/m/l.pt")
    ap.add_argument("--imgsz", type=int, default=960, help="detection image size (bigger finds smaller people)")
    ap.add_argument("--conf", type=float, default=0.3, help="minimum detection confidence")
    ap.add_argument("--tracker", default="tracker.yaml", help="tracker settings file (or botsort.yaml)")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    stride = max(1, round(src_fps / args.fps))  # process every Nth frame
    print(f"Video FPS: {src_fps:.1f} | processing every {stride} frames (~{src_fps / stride:.1f} per second)")
    print(f"Model: {args.model} | image size: {args.imgsz} | tracker: {args.tracker}")

    model = YOLO(args.model)  # downloads the model automatically the first time

    out_file = open("data/trajectories.csv", "w", newline="")
    writer = csv.writer(out_file)
    writer.writerow(["track_id", "frame", "t_sec", "x_px", "y_px", "conf", "x1", "y1", "x2", "y2"])

    frame_idx, done, saved_reference, started = 0, 0, False, time.time()
    n_frames_to_process = (total + stride - 1) // stride if total else 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            if not saved_reference:
                cv2.imwrite("data/reference_frame.jpg", frame)  # used later to draw zones
                saved_reference = True

            # classes=[0] = "person" only. persist=True keeps IDs consistent from frame to frame.
            result = model.track(frame, persist=True, classes=[0], conf=args.conf, imgsz=args.imgsz,
                                 tracker=args.tracker, verbose=False)[0]

            if result.boxes.id is not None:
                ids = result.boxes.id.int().tolist()
                boxes = result.boxes.xyxy.tolist()
                confs = result.boxes.conf.tolist()
                for track_id, (x1, y1, x2, y2), conf in zip(ids, boxes, confs):
                    foot_x = (x1 + x2) / 2   # middle of the box...
                    foot_y = y2              # ...at the bottom = where the person stands
                    writer.writerow([track_id, frame_idx, round(frame_idx / src_fps, 3),
                                     round(foot_x, 1), round(foot_y, 1), round(conf, 2),
                                     round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            done += 1
            if done % 25 == 0:
                speed = done / (time.time() - started)
                left = f" | about {(n_frames_to_process - done) / speed / 60:.1f} min left" if n_frames_to_process else ""
                print(f"  processed {done}/{n_frames_to_process or '?'} frames{left}")

        frame_idx += 1

    cap.release()
    out_file.close()
    print("Done. Saved data/trajectories.csv and data/reference_frame.jpg")


if __name__ == "__main__":
    main()

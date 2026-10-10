"""
STEP 3b: Draw the tracking video that plays on the dashboard.

Shows a box + ID + short trail for every person, your zones (coloured by their status) and a live
count of people inside each zone. Run it AFTER step3_analyze.py.

Run:  python step3b_render_video.py --video your_video.mp4
Out:  outputs/tracking_video.mp4   (H.264, plays in any browser)
"""
import argparse
import json
import os

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd

STATUS_COLOUR = {"Bottleneck": (0, 0, 255), "Dead zone": (255, 150, 0), "OK": (255, 255, 255)}  # BGR


def person_colour(pid):
    rng = np.random.default_rng(int(pid) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="the SAME video you gave to step1_track.py")
    ap.add_argument("--fps", type=float, default=15, help="frames per second of the output video")
    ap.add_argument("--width", type=int, default=960, help="output video width in pixels")
    ap.add_argument("--trail-sec", type=float, default=3.0, help="length of the walking trail")
    args = ap.parse_args()

    traj = pd.read_csv("outputs/trajectories_clean.csv")
    zones = json.load(open("data/zones.json"))
    status = {}
    if os.path.exists("outputs/zone_stats.csv"):
        status = dict(pd.read_csv("outputs/zone_stats.csv")[["zone", "status"]].values)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w, src_h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    sc = min(1.0, args.width / src_w)
    out_w, out_h = int(src_w * sc) // 2 * 2, int(src_h * sc) // 2 * 2   # H.264 needs even sizes
    out_stride = max(1, round(src_fps / args.fps))
    out_fps = src_fps / out_stride

    # per-person arrays (frame numbers, boxes, feet) so we can smoothly interpolate between processed frames
    step = max(1, int(np.median(np.diff(np.sort(traj["frame"].unique())))))   # frames between processed frames
    max_gap = 3 * step
    people = {}
    for pid, d in traj.groupby("person_id"):
        d = d.sort_values("frame")
        people[pid] = {"fr": d["frame"].to_numpy(float),
                       "box": d[["x1", "y1", "x2", "y2"]].to_numpy(float),
                       "foot": d[["x_px", "y_px"]].to_numpy(float)}

    zone_polys = [np.array(z["polygon"], np.float32).reshape(-1, 1, 2) for z in zones]
    trail_frames = args.trail_sec * src_fps

    os.makedirs("outputs", exist_ok=True)
    writer = imageio_ffmpeg.write_frames("outputs/tracking_video.mp4", (out_w, out_h), fps=out_fps,
                                         codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                                         quality=6, macro_block_size=2)
    writer.send(None)

    f, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f % out_stride == 0:
            frame = cv2.resize(frame, (out_w, out_h))

            # --- who is visible in this frame, and where? ---
            visible = []
            for pid, p in people.items():
                fr = p["fr"]
                if f < fr[0] or f > fr[-1]:
                    continue
                i = int(np.searchsorted(fr, f))
                if fr[i] != f and fr[i] - fr[i - 1] > max_gap:
                    continue  # tracker lost this person around now
                box = [np.interp(f, fr, p["box"][:, k]) for k in range(4)]
                foot = [np.interp(f, fr, p["foot"][:, k]) for k in range(2)]
                recent = p["foot"][(fr <= f) & (fr >= f - trail_frames)]
                visible.append((pid, box, foot, recent))

            # --- zones with live counts ---
            overlay = frame.copy()
            counts = []
            for poly in zone_polys:
                counts.append(sum(cv2.pointPolygonTest(poly, (float(v[2][0]), float(v[2][1])), False) >= 0
                                  for v in visible))
            for z, poly, n in zip(zones, zone_polys, counts):
                pts = (poly.reshape(-1, 2) * sc).astype(np.int32)
                colour = STATUS_COLOUR.get(status.get(z["name"], "OK"), (255, 255, 255))
                cv2.fillPoly(overlay, [pts], colour)
            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)
            for z, poly, n in zip(zones, zone_polys, counts):
                pts = (poly.reshape(-1, 2) * sc).astype(np.int32)
                colour = STATUS_COLOUR.get(status.get(z["name"], "OK"), (255, 255, 255))
                cv2.polylines(frame, [pts], True, colour, 2)
                label = f'{z["name"]}: {n}'
                x, y = int(pts[0][0]), max(18, int(pts[0][1]) - 6)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

            # --- people: trail, box, ID ---
            for pid, box, foot, recent in visible:
                c = person_colour(pid)
                if len(recent) > 1:
                    cv2.polylines(frame, [(recent * sc).astype(np.int32).reshape(-1, 1, 2)], False, c, 2)
                x1, y1, x2, y2 = [int(v * sc) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                tag = f"ID {int(pid)}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), c, -1)
                cv2.putText(frame, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.circle(frame, (int(foot[0] * sc), int(foot[1] * sc)), 4, (0, 0, 255), -1)

            banner = f"People in view: {len(visible)}   |   time {f / src_fps:5.1f}s"
            cv2.rectangle(frame, (0, 0), (330, 26), (0, 0, 0), -1)
            cv2.putText(frame, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            writer.send(np.ascontiguousarray(frame).tobytes())
            written += 1
            if written % 100 == 0:
                print(f"  rendered {f}/{total or '?'} frames")
        f += 1

    writer.close()
    cap.release()
    print(f"Done. Saved outputs/tracking_video.mp4 ({written} frames at {out_fps:.1f} fps)")


if __name__ == "__main__":
    main()

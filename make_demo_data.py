"""
OPTIONAL: creates fake data so you can test steps 3 and 4 BEFORE you have a real video.
It writes data/trajectories.csv, data/reference_frame.jpg and data/zones.json.

Run:  python make_demo_data.py
"""
import csv
import json
import os

import cv2
import numpy as np

rng = np.random.default_rng(7)
os.makedirs("data", exist_ok=True)
FPS, W, H = 5, 1280, 720

ZONES = {
    "Promo Display": [[300, 150], [520, 150], [520, 300], [300, 300]],
    "Electronics":   [[700, 150], [980, 150], [980, 300], [700, 300]],
    "Checkout":      [[900, 450], [1180, 450], [1180, 650], [900, 650]],
    "Back Corner":   [[80, 500], [300, 500], [300, 680], [80, 680]],
}

# --- a simple fake "store photo" ---
img = np.full((H, W, 3), (170, 170, 170), np.uint8)
for name, poly in ZONES.items():
    p = np.array(poly, np.int32)
    cv2.fillPoly(img, [p], (120, 90, 60))
    cv2.putText(img, name, (p[0][0] + 10, p[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite("data/reference_frame.jpg", img)
json.dump([{"name": n, "polygon": p} for n, p in ZONES.items()], open("data/zones.json", "w"), indent=2)


def simulate(start_t, waypoints):
    """walk through waypoints (x, y, pause_seconds); return list of (t, x, y)."""
    t, (x, y) = start_t, waypoints[0][:2]
    out = []
    for wx, wy, pause in waypoints[1:]:
        while np.hypot(wx - x, wy - y) > 8:           # walk at ~150 px/s
            d = np.hypot(wx - x, wy - y)
            move = min(150 / FPS, d)                       # never overshoot the waypoint
            x += (wx - x) / d * move
            y += (wy - y) / d * move
            t += 1 / FPS
            out.append((t, x, y))
        for _ in range(int(pause * FPS)):               # stand still for 'pause' seconds
            t += 1 / FPS
            out.append((t, x + rng.normal(0, 2), y + rng.normal(0, 2)))
    return out


rows = []
for pid in range(1, 71):
    to_checkout = rng.random() < 0.3
    start = rng.uniform(40, 70) if to_checkout else rng.uniform(0, 150)
    wp = [(60, 380, 0), (400, 380, 0)]
    if rng.random() < 0.10:                              # few people stop at the promo display
        wp += [(410, 250, rng.uniform(2, 6)), (410, 380, 0)]
    if not to_checkout and rng.random() < 0.38:          # many people browse electronics
        wp += [(840, 380, 0), (840, 250, rng.uniform(8, 20)), (840, 380, 0)]
    if to_checkout:
        wp += [(1040, 550, rng.uniform(15, 30)), (1040, 380, 0)]
    if rng.random() < 0.03:
        wp += [(190, 590, rng.uniform(3, 6))]
    wp += [(60, 380, 0)]
    for t, x, y in simulate(start, wp):
        fx, fy = round(x + rng.normal(0, 2), 1), round(y + rng.normal(0, 2), 1)
        rows.append([pid, int(t * FPS), round(t, 2), fx, fy, 0.9, fx - 25, fy - 120, fx + 25, fy])  # box around the feet

rows.sort(key=lambda r: (r[0], r[2]))
with open("data/trajectories.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["track_id", "frame", "t_sec", "x_px", "y_px", "conf", "x1", "y1", "x2", "y2"])
    w.writerows(rows)
print(f"Created demo data: {len(rows)} rows, 70 fake people.")

"""
STEP 2: Draw store zones on a frame of your video -> data/zones.json

Controls (click on the picture window first):
  LEFT click   add a corner point
  RIGHT click  undo last point
  N            finish this zone (you will be asked for its name in the TERMINAL)
  S            save all zones and quit
  Q            quit without saving

Run:  python step2_zones.py
"""
import json
import os

import cv2
import numpy as np

FRAME_PATH = "data/reference_frame.jpg"
ZONES_PATH = "data/zones.json"
MAX_WIDTH = 1200  # shrink big frames so they fit on your screen

frame = cv2.imread(FRAME_PATH)
if frame is None:
    raise SystemExit(f"Cannot find {FRAME_PATH}. Run step1_track.py first.")

scale = min(1.0, MAX_WIDTH / frame.shape[1])
view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()

zones = []
if os.path.exists(ZONES_PATH):
    zones = json.load(open(ZONES_PATH))
    print(f"Loaded {len(zones)} existing zone(s). Delete {ZONES_PATH} to start fresh.")

current = []  # points of the zone being drawn (in display coordinates)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        current.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and current:
        current.pop()


def redraw():
    img = view.copy()
    overlay = img.copy()
    for z in zones:
        pts = (np.array(z["polygon"]) * scale).astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    for z in zones:
        pts = (np.array(z["polygon"]) * scale).astype(np.int32)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.putText(img, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    for p in current:
        cv2.circle(img, p, 4, (0, 0, 255), -1)
    if len(current) > 1:
        cv2.polylines(img, [np.array(current, np.int32)], False, (0, 0, 255), 2)
    cv2.putText(img, "Click corners | N=finish zone | S=save | Q=quit", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


cv2.namedWindow("Draw zones")
cv2.setMouseCallback("Draw zones", on_mouse)

while True:
    cv2.imshow("Draw zones", redraw())
    key = cv2.waitKey(30) & 0xFF
    if key == ord("n"):
        if len(current) < 3:
            print("A zone needs at least 3 points.")
            continue
        name = input("Zone name (e.g. Electronics): ").strip() or f"Zone {len(zones) + 1}"
        polygon = [[round(x / scale, 1), round(y / scale, 1)] for x, y in current]  # back to real pixels
        zones.append({"name": name, "polygon": polygon})
        current.clear()
        print(f"Added '{name}'. Draw the next zone, or press S to save.")
    elif key == ord("s"):
        os.makedirs("data", exist_ok=True)
        json.dump(zones, open(ZONES_PATH, "w"), indent=2)
        print(f"Saved {len(zones)} zone(s) to {ZONES_PATH}")
        break
    elif key == ord("q"):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()

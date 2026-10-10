"""
STEP 3: trajectories + zones -> dwell times, heatmap, bottleneck / dead-zone flags

Reads : data/trajectories.csv, data/zones.json, data/reference_frame.jpg
Writes: outputs/zone_stats.csv, outputs/occupancy.csv, outputs/dwell_by_person.csv,
        outputs/summary.json, outputs/heatmap.png

Run:  python step3_analyze.py
"""
import json
import os

import cv2
import numpy as np
import pandas as pd

# ---------------- settings you can tune ----------------
MIN_TRACK_SEC = 1.0      # ignore tracks shorter than this (usually detection noise)
MIN_VISIT_SEC = 1.0      # someone must stay this long inside a zone to count as a visitor
DEAD_ZONE_SHARE = 0.10   # zone visited by less than 10% of all people = dead zone
CROWD_PEOPLE = 5         # a zone with this many people at once is "crowded"...
CROWD_SECONDS = 10       # ...if that lasts this long, flag it as a bottleneck
OCC_BIN_SEC = 1.0        # occupancy is counted once per second
STITCH = True            # re-join one person's broken tracks (see stitch_tracks below)
STITCH_MAX_GAP_SEC = 4.0 # a person may "disappear" (hidden behind a shelf) for at most this long
# --------------------------------------------------------


def stitch_tracks(df, frame_w, frame_h):
    """The tracker sometimes loses a person (hidden behind a shelf) and gives them a NEW id when
    they reappear. That splits one visit into two and ruins dwell times. Here we glue a track that
    ends to a track that starts shortly after, close by (distance measured in 'person heights')."""
    g = df.groupby("track_id")
    info = pd.DataFrame({
        "start": g["t_sec"].min(), "end": g["t_sec"].max(),
        "sx": g["x_px"].first(), "sy": g["y_px"].first(),
        "ex": g["x_px"].last(), "ey": g["y_px"].last(),
        "height": g.apply(lambda d: (d["y2"] - d["y1"]).median(), include_groups=False),
    }).sort_values("start")

    # a track that starts/ends at the edge of the picture means someone walked in/out of view.
    # Those must NOT be glued to another person (e.g. one person leaves by the door, another enters).
    mx, my = 0.04 * frame_w, 0.04 * frame_h
    def at_border(g, which):
        r = g.first() if which == "first" else g.last()
        return (r["x1"] < mx) | (r["x2"] > frame_w - mx) | (r["y1"] < my) | (r["y2"] > frame_h - my)
    info["enters_view"] = at_border(g, "first")
    info["leaves_view"] = at_border(g, "last")

    new_id, used_ends = {}, set()
    for b, rb in info.iterrows():
        new_id[b] = b
        best, best_d = None, None
        if rb["enters_view"]:
            continue   # B walked into view from the edge: it is a new person
        for a, ra in info.iterrows():
            gap = rb["start"] - ra["end"]
            if a == b or a in used_ends or ra["leaves_view"] or gap <= 0 or gap > STITCH_MAX_GAP_SEC:
                continue
            d = np.hypot(rb["sx"] - ra["ex"], rb["sy"] - ra["ey"])
            allowed = ra["height"] * (0.5 + 0.9 * gap)   # a walking person covers ~0.9 heights per second
            if d <= allowed and (best is None or d < best_d):
                best, best_d = a, d
        if best is not None:
            new_id[b] = new_id[best]
            used_ends.add(best)
    out = df.copy()
    out["track_id"] = out["track_id"].map(new_id)
    return out

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/trajectories.csv")
zones = json.load(open("data/zones.json"))
ref = cv2.imread("data/reference_frame.jpg")
if df.empty:
    raise SystemExit("trajectories.csv is empty - no people were tracked.")
if "y2" not in df.columns:
    raise SystemExit("trajectories.csv is from an older version. Please re-run step1_track.py.")
H, W = ref.shape[:2]

# ---- 1. clean: re-join broken tracks, then remove very short ones ----
df = df.sort_values(["track_id", "t_sec"]).reset_index(drop=True)
df["orig_id"] = df["track_id"]
n_before = df["track_id"].nunique()
if STITCH:
    df = stitch_tracks(df, W, H).sort_values(["track_id", "t_sec"]).reset_index(drop=True)
    print(f"Track stitching: {n_before} raw tracks -> {df['track_id'].nunique()} people")
duration = df.groupby("track_id")["t_sec"].agg(lambda s: s.max() - s.min())
df = df[df["track_id"].isin(duration[duration >= MIN_TRACK_SEC].index)].copy()
n_people = df["track_id"].nunique()
df.rename(columns={"track_id": "person_id"}).to_csv("outputs/trajectories_clean.csv", index=False)  # used by the video renderer

# ---- 2. how much time does each row represent? (gap to the person's next row) ----
step = df.groupby("track_id")["t_sec"].diff().median()
dt = df.groupby("track_id")["t_sec"].shift(-1) - df["t_sec"]
df["dt"] = dt.fillna(step).clip(upper=2 * step)  # cap so a missed detection doesn't add fake time

# ---- 3. for each zone: who was inside, for how long? ----
rows, dwell_rows, occ_frames = [], [], []
t_max = int(df["t_sec"].max() // OCC_BIN_SEC) + 1

for z in zones:
    contour = np.array(z["polygon"], dtype=np.float32).reshape(-1, 1, 2)
    inside = np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
                       for x, y in zip(df["x_px"], df["y_px"])])
    sub = df[inside]

    # dwell time per person = sum of the time slices spent inside
    dwell = sub.groupby("track_id")["dt"].sum()
    visitors = dwell[dwell >= MIN_VISIT_SEC]
    for tid, d in visitors.items():
        dwell_rows.append({"track_id": tid, "zone": z["name"], "dwell_s": round(d, 1)})

    # how many different people were inside during each second?
    bins = (sub["t_sec"] // OCC_BIN_SEC).astype(int)
    people = sub.groupby(bins)["track_id"].nunique().reindex(range(t_max), fill_value=0)
    occ_frames.append(pd.DataFrame({"time_s": people.index * OCC_BIN_SEC, "zone": z["name"], "people": people.values}))

    # longest stretch of time the zone stayed crowded
    longest = run = 0
    for crowded in (people.values >= CROWD_PEOPLE):
        run = run + 1 if crowded else 0
        longest = max(longest, run)
    crowded_seconds = longest * OCC_BIN_SEC

    share = len(visitors) / n_people
    if crowded_seconds >= CROWD_SECONDS:
        status = "Bottleneck"
    elif share < DEAD_ZONE_SHARE:
        status = "Dead zone"
    else:
        status = "OK"

    rows.append({
        "zone": z["name"],
        "visitors": len(visitors),
        "visitor_share_%": round(100 * share, 1),
        "avg_dwell_s": round(visitors.mean(), 1) if len(visitors) else 0,
        "median_dwell_s": round(visitors.median(), 1) if len(visitors) else 0,
        "max_dwell_s": round(visitors.max(), 1) if len(visitors) else 0,
        "peak_people_at_once": int(people.max()),
        "crowded_seconds": crowded_seconds,
        "status": status,
    })

stats = pd.DataFrame(rows)
stats.to_csv("outputs/zone_stats.csv", index=False)
pd.concat(occ_frames).to_csv("outputs/occupancy.csv", index=False)
pd.DataFrame(dwell_rows).to_csv("outputs/dwell_by_person.csv", index=False)
json.dump({"people_tracked": int(n_people), "duration_s": round(float(df["t_sec"].max()), 1)},
          open("outputs/summary.json", "w"))

# ---- 4. heatmap: add up time spent at every spot, blur it, colour it ----
grid = np.zeros((H, W), dtype=np.float32)
xs = df["x_px"].clip(0, W - 1).astype(int).to_numpy()
ys = df["y_px"].clip(0, H - 1).astype(int).to_numpy()
np.add.at(grid, (ys, xs), df["dt"].to_numpy())
grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=W / 60)
grid = grid / (grid.max() + 1e-9)                      # 0..1
colour = cv2.applyColorMap((grid * 255).astype(np.uint8), cv2.COLORMAP_JET)
alpha = np.clip(grid * 1.5, 0, 0.7)[..., None]          # strong spots more visible, empty spots transparent
heat = (ref * (1 - alpha) + colour * alpha).astype(np.uint8)

status_colour = {"Bottleneck": (0, 0, 255), "Dead zone": (255, 150, 0), "OK": (255, 255, 255)}  # BGR
for z, row in zip(zones, rows):
    pts = np.array(z["polygon"], dtype=np.int32)
    c = status_colour[row["status"]]
    cv2.polylines(heat, [pts], True, c, 3)
    cv2.putText(heat, f'{z["name"]} ({row["status"]})', (pts[0][0], max(20, pts[0][1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
cv2.imwrite("outputs/heatmap.png", heat)

# ---- 5. print a readable report ----
print(f"\nPeople tracked: {n_people} | video length: {df['t_sec'].max():.0f} s\n")
print(stats.to_string(index=False))
print("\nSaved everything in the 'outputs' folder.")

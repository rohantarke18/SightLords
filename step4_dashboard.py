"""
STEP 4: Simple dashboard

Run:  streamlit run step4_dashboard.py
"""
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="StoreFlow AI", layout="wide")
st.title("StoreFlow AI: Dwell-Time & Bottleneck Mapper")

if not os.path.exists("outputs/zone_stats.csv"):
    st.error("No results found. Run step3_analyze.py first.")
    st.stop()

stats = pd.read_csv("outputs/zone_stats.csv")
occ = pd.read_csv("outputs/occupancy.csv")
summary = json.load(open("outputs/summary.json"))

# ---- top numbers ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("People tracked", summary["people_tracked"])
c2.metric("Video length (s)", summary["duration_s"])
c3.metric("Bottlenecks", int((stats["status"] == "Bottleneck").sum()))
c4.metric("Dead zones", int((stats["status"] == "Dead zone").sum()))

# ---- heatmap + table ----
left, right = st.columns([3, 2])
left.subheader("Foot-traffic heatmap")
left.image("outputs/heatmap.png", width="stretch")
right.subheader("Zone statistics")
right.dataframe(stats, width="stretch", hide_index=True)

# ---- tracking video ----
if os.path.exists("outputs/tracking_video.mp4"):
    st.subheader("Tracking video")
    st.caption("Boxes and IDs follow each person. Zones are coloured by status: red = bottleneck, "
               "blue = dead zone, white = OK. The number next to a zone name is how many people are inside right now.")
    st.video("outputs/tracking_video.mp4")
else:
    st.info("No tracking video yet. Run step3b_render_video.py to create it.")

# ---- alerts ----
st.subheader("Alerts")
flagged = False
for _, r in stats.iterrows():
    if r["status"] == "Bottleneck":
        st.error(f"**{r['zone']}**: bottleneck. Up to {r['peak_people_at_once']} people at once, "
                 f"crowded for {r['crowded_seconds']:.0f} s.")
        flagged = True
    elif r["status"] == "Dead zone":
        st.warning(f"**{r['zone']}**: dead zone. Only {r['visitor_share_%']}% of people visited it.")
        flagged = True
if not flagged:
    st.success("No bottlenecks or dead zones detected.")

# ---- charts ----
a, b = st.columns(2)
a.subheader("Average dwell time per zone (s)")
a.bar_chart(stats.set_index("zone")["avg_dwell_s"])
b.subheader("Visitors per zone")
b.bar_chart(stats.set_index("zone")["visitors"])

st.subheader("People inside each zone over time")
st.line_chart(occ.pivot(index="time_s", columns="zone", values="people"))

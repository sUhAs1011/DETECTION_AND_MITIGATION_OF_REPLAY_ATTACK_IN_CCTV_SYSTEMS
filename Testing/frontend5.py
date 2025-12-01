import requests
import streamlit as st
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Replay Detection Dashboard", layout="wide")
st.title("👁️ Replay Detection Dashboard")

st.autorefresh = st_autorefresh(interval=1000, key="verification-refresh")

with st.sidebar:
    st.header("Stream Controls")
    camera_id = st.text_input("Camera ID", "CAM-001")
    default_video = r"C:\Users\Shreyas S\Documents\htm.core\capstone\data\normal_videos\1.mp4"
    video_path = st.text_input("Video File Path", default_video)
    rtsp_url = "rtsp://localhost:8554/camera"

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start Stream", type="primary"):
            try:
                r = requests.post(f"{BACKEND}/stream/start", params={"camera_id": camera_id, "rtsp_url": rtsp_url, "video_file": video_path, "max_hash_seconds": 3600})
                if r.status_code == 200: st.success("Stream Started")
                else: st.error(f"Error: {r.text}")
            except Exception as e: st.error(f"Connection Failed: {e}")

    with col2:
        if st.button("⏹ Stop Stream"):
            try:
                r = requests.post(f"{BACKEND}/stream/stop")
                if r.status_code == 200: st.info("Stream Stopped")
            except Exception as e: st.error(f"Connection Failed: {e}")

try:
    health = requests.get(f"{BACKEND}/health", timeout=0.5).json()
except: health = {}

st.markdown("### System Status")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Backend", "Online" if health else "Offline")
col2.metric("FFmpeg", "Running" if health.get("ffmpeg_running") else "Stopped")
col3.metric("HTM Engine", "Running" if health.get("htm_engine_running") else "Stopped")
col4.metric("Verification", "Active" if health.get("verification_active") else "Inactive")

st.divider()

try:
    stats = requests.get(f"{BACKEND}/stats", params={"camera_id": camera_id}, timeout=0.5).json() if camera_id else {}
except: stats = {"detected": 0, "events": []}

replay_count = stats.get("detected", 0)
events = stats.get("events", [])

if replay_count > 0:
    st.error(f"🚨 **REPLAY ATTACK CONFIRMED**\n\nHTM Panic + Hash Match Verified.")
elif any("SUSPICIOUS" in e.get("verdict", "") for e in events):
    st.warning(f"🟠 **SUSPICIOUS ACTIVITY DETECTED**\n\nMotion Anomalies Detected (Unverified).")
elif any("UNVERIFIED" in e.get("verdict", "") for e in events):
    st.warning(f"🟡 **POSSIBLE DUPLICATE**\n\nHash Match found (HTM Normal).")
else:
    st.success("✅ **SYSTEM SECURE** - No Anomalies Detected")

st.subheader("Live Decision Log (Fusion Logic)")

def color_verdict(val):
    color = "green"
    weight = "normal"
    if "CONFIRMED" in str(val): color = "#FF4B4B"; weight = "bold"
    elif "SUSPICIOUS" in str(val): color = "#FFA500"; weight = "bold"
    elif "UNVERIFIED" in str(val): color = "#FFD700"; weight = "normal"
    return f'color: {color}; font-weight: {weight}'

if events:
    table_data = []
    for e in events:
        ts_raw = e.get("timestamp", "")
        ts_display = ts_raw.split("T")[1][:8] if "T" in ts_raw else ts_raw
        
        table_data.append({
            "Time": ts_display,
            "Frame": e.get("frame"),
            "HTM Status": e.get("htm", "-"),
            "Hash Status": e.get("hash", "-"),
            "Verdict": e.get("verdict", "-")
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df.style.map(color_verdict, subset=["Verdict"]), use_container_width=True, height=400, hide_index=True)
else:
    st.info("System Monitoring... No events detected yet.")
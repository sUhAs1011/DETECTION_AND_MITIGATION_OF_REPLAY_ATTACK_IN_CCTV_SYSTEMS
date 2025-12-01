import subprocess
import threading
import time
import sqlite3
import uuid
import os
import glob
import json
import signal
import psutil
import sys
from fastapi import FastAPI, Query
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Import your existing hashing module
from hashing_module.live_hash5 import live_hash

# --- Absolute Path Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hashing_module", "db", "hash_store.db")
ALERTS_DIR = os.path.join(BASE_DIR, "htm", "alerts")
HTM_SCRIPT = os.path.join(BASE_DIR, "htm", "htm_infer_service2.py")

os.makedirs(ALERTS_DIR, exist_ok=True)

app = FastAPI(title="Unified Replay Detection Backend")

# Global Session State
current_session: Dict[str, Any] = {
    "camera_id": None,
    "rtsp_url": None,
    "video_file": None,
    "video_id": None,
    "hashing_active": False,
    "verification_active": False,
    "htm_pid": None
}

# Process Handles
ffmpeg_proc: Optional[subprocess.Popen] = None
ffplay_proc: Optional[subprocess.Popen] = None
htm_proc: Optional[subprocess.Popen] = None

# Thread Handles
manager_thread: Optional[threading.Thread] = None
continuous_hash_thread: Optional[threading.Thread] = None
verification_results = []

# ---------------- Configuration ----------------
WARMUP_FRAMES = 100
EXCLUSION_ZONE_SECONDS = 2.0
VERIFICATION_POLL_INTERVAL = 1.0
CORRELATION_WINDOW_SEC = 30.0 

# ---------------- Database Initialization ----------------
def init_db():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_number INTEGER,
                timestamp TEXT,
                camera_id TEXT,
                video_id TEXT,
                message TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_hashes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                video_id TEXT,
                frame_number INTEGER,
                timestamp TEXT, 
                hash TEXT,
                phash TEXT,
                hash_algo TEXT,
                gray INTEGER,
                width INTEGER,
                height INTEGER,
                serialize_mode TEXT,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()
        print(f"[Backend] Database initialized at {DB_PATH}")
    except Exception as e:
        print(f"[Backend] ❌ DB Init Error: {e}")

init_db()

# ---------------- Helper: Process Management ----------------
def start_process(cmd):
    if os.name == 'posix':
        return subprocess.Popen(cmd, preexec_fn=os.setsid)
    return subprocess.Popen(cmd)

def stop_process(proc: Optional[subprocess.Popen]):
    if proc and proc.poll() is None:
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            proc.wait(timeout=2)
        except Exception:
            try: proc.kill()
            except: pass
    return None

def stop_ffmpeg():
    global ffmpeg_proc
    ffmpeg_proc = stop_process(ffmpeg_proc)

def stop_ffplay():
    global ffplay_proc
    ffplay_proc = stop_process(ffplay_proc)

# ---------------- Stream Management ----------------
def start_ffmpeg_stream(video_file: str, rtsp_url: str):
    stop_ffmpeg()
    # Low-latency settings
    cmd = [
        "ffmpeg", "-re", "-stream_loop", "-1", "-i", video_file,
        "-s", "640x480", "-r", "20", "-b:v", "500k",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-f", "rtsp", rtsp_url
    ]
    global ffmpeg_proc
    print(f"[Backend] Starting FFmpeg Producer...")
    ffmpeg_proc = start_process(cmd)

def start_ffplay(rtsp_url: str):
    stop_ffplay()
    cmd = ["ffplay", "-fflags", "nobuffer", "-flags", "low_delay", "-x", "640", "-y", "480", rtsp_url]
    global ffplay_proc
    print(f"[Backend] Starting FFplay Viewer...")
    ffplay_proc = start_process(cmd)

def start_htm_service(rtsp_url: str, video_id: str):
    global htm_proc
    htm_proc = stop_process(htm_proc)
    
    python_exe = sys.executable
    # Pass video_id to the script so it can log it
    cmd = [python_exe, "-u", HTM_SCRIPT, "--video", rtsp_url, "--video-id", video_id]
    
    print(f"[Backend] Starting HTM Service: {' '.join(cmd)}")
    htm_proc = start_process(cmd)
    current_session["htm_pid"] = htm_proc.pid

# ---------------- Verification Logic (Hashing) ----------------
def verify_latest_frame(camera_id: str, video_id: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get latest frame for CURRENT SESSION ONLY
    cursor.execute("""
        SELECT frame_number, timestamp, phash 
        FROM frame_hashes 
        WHERE camera_id=? AND video_id=? 
        ORDER BY frame_number DESC LIMIT 1
    """, (camera_id, video_id))
    
    latest = cursor.fetchone()
    
    if not latest:
        conn.close()
        return {"status": "waiting"}

    curr_frame, curr_ts_str, curr_phash = latest
    
    try:
        curr_ts = datetime.fromisoformat(curr_ts_str)
    except ValueError:
        conn.close()
        return {"status": "error_ts"}

    if curr_frame < WARMUP_FRAMES:
        conn.close()
        return {"status": "warming_up"}

    time_thresh = (curr_ts - timedelta(seconds=EXCLUSION_ZONE_SECONDS)).isoformat()
    
    # Check history for CURRENT SESSION ONLY
    cursor.execute("""
        SELECT frame_number, timestamp 
        FROM frame_hashes 
        WHERE camera_id=? 
          AND video_id=? 
          AND phash=? 
          AND timestamp < ? 
        ORDER BY timestamp ASC LIMIT 1
    """, (camera_id, video_id, curr_phash, time_thresh))
    
    match = cursor.fetchone()
    
    flagged = False
    if match:
        flagged = True
        orig_frame, orig_ts_str = match
        orig_ts = datetime.fromisoformat(orig_ts_str)
        delta = (curr_ts - orig_ts).total_seconds()
        
        msg = f"[HASH] Visual Match: Frame {curr_frame} matches Frame {orig_frame} ({delta:.1f}s ago)"
        print(f"🚨 {msg}")
        
        # Debounce (Current Session Only)
        cursor.execute("""
            SELECT count(*) FROM alerts 
            WHERE frame_number=? AND camera_id=? AND video_id=?
        """, (curr_frame, camera_id, video_id))
        
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO alerts (frame_number, timestamp, camera_id, video_id, message) 
                VALUES (?, ?, ?, ?, ?)
            """, (curr_frame, curr_ts_str, camera_id, video_id, msg))
            conn.commit()

    conn.close()
    return {"status": "verified", "flagged": flagged}

def verification_loop(camera_id: str, video_id: str):
    current_session["verification_active"] = True
    print(f"[Backend] Verification Loop Started for {video_id}")
    
    print("[Backend] Verification waiting for HTM signal...")
    wait_start = time.time()
    flag_path = os.path.join(BASE_DIR, "htm_ready.flag")
    
    htm_ready = False
    while current_session["hashing_active"]:
        if os.path.exists(flag_path):
            htm_ready = True
            print(f"[Backend] ✅ HTM Signal received. Starting verification.")
            break
        if time.time() - wait_start > 30:
            print("[Backend] ❌ Timeout waiting for HTM. Starting anyway.")
            htm_ready = True
            break
        time.sleep(1.0)

    try:
        while current_session["hashing_active"]:
            try:
                verify_latest_frame(camera_id, video_id)
            except Exception as e:
                print(f"[Backend] Verify Error: {e}")
            time.sleep(VERIFICATION_POLL_INTERVAL)
    finally:
        current_session["verification_active"] = False
        print(f"[Backend] Verification Loop Stopped")

# ---------------- Thread Manager ----------------
def hash_and_verify_manager(rtsp_url: str, camera_id: str, video_id: str, max_hash_seconds: int):
    current_session["hashing_active"] = True
    
    global continuous_hash_thread
    continuous_hash_thread = threading.Thread(target=live_hash, kwargs={"stream_url": rtsp_url, "camera_id": camera_id, "video_id": video_id, "max_seconds": max_hash_seconds}, daemon=True)
    continuous_hash_thread.start()
    
    threading.Thread(target=verification_loop, args=(camera_id, video_id), daemon=True).start()
    
    continuous_hash_thread.join()
    current_session["hashing_active"] = False

# ---------------- Unified Alert Aggregator ----------------
def get_htm_alerts(target_video_id):
    alerts = []
    try:
        search_pattern = os.path.join(ALERTS_DIR, "alerts_events_*.jsonl")
        files = glob.glob(search_pattern)
        if not files: return []
        
        # Get latest file
        latest_file = max(files, key=os.path.getctime)
        
        with open(latest_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # STRICT SESSION FILTER: 
                        # Only accept alerts that match the current video_id
                        if data.get("video_id") == target_video_id:
                            if data.get("event") in ["ATTACK_START", "ATTACK_UPDATE"]:
                                fmt_alert = {
                                    "source": "HTM",
                                    "frame_index": data.get("frame"),
                                    "timestamp": data.get("timestamp"),
                                    "status": data.get("status"), 
                                    "reason": data.get("reason")
                                }
                                alerts.append(fmt_alert)
                    except: pass
    except Exception as e:
        print(f"[Backend] HTM Alerts Read Error: {e}")
    return alerts

def get_hash_alerts(camera_id, target_video_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # STRICT SESSION FILTER: Add video_id=?
        cursor.execute("""
            SELECT frame_number, timestamp, message 
            FROM alerts 
            WHERE camera_id=? AND video_id=? 
            ORDER BY timestamp ASC
        """, (camera_id, target_video_id))
        
        db_rows = cursor.fetchall()
        conn.close()
        return [{"source": "HASH", "frame_index": r[0], "timestamp": str(r[1]), "reason": r[2]} for r in db_rows]
    except Exception: return []

# ---------------- API Endpoints ----------------
@app.post("/stream/start")
def stream_start(camera_id: str = Query(...), rtsp_url: str = Query(...), video_file: str = Query(...), max_hash_seconds: int = Query(3600)):
    video_id = f"VID-{uuid.uuid4().hex[:8]}"
    current_session.update({"camera_id": camera_id, "rtsp_url": rtsp_url, "video_file": video_file, "video_id": video_id})
    
    # Clean old flag
    flag_path = os.path.join(BASE_DIR, "htm_ready.flag")
    if os.path.exists(flag_path):
        os.remove(flag_path)

    print(f"[Backend] Initializing Session {video_id}...")
    start_ffmpeg_stream(video_file, rtsp_url)
    
    print("[Backend] Waiting 5s for RTSP stabilization...")
    time.sleep(5) 
    
    # 1. Start HTM (Passed video_id)
    start_htm_service(rtsp_url, video_id)
    
    # 2. Wait to avoid connection collision
    time.sleep(2)
    
    # 3. Start Hashing
    print("[Backend] Launching Hashing Engine...")
    global manager_thread
    manager_thread = threading.Thread(target=hash_and_verify_manager, args=(rtsp_url, camera_id, video_id, max_hash_seconds), daemon=True)
    manager_thread.start()

    # 4. Start Viewer
    threading.Thread(target=start_ffplay, args=(rtsp_url,), daemon=True).start()
    
    return {"status": "started", "video_id": video_id}

@app.post("/stream/stop")
def stream_stop():
    global ffmpeg_proc, ffplay_proc, htm_proc
    stop_ffmpeg()
    stop_ffplay()
    htm_proc = stop_process(htm_proc)
    current_session["hashing_active"] = False
    current_session["htm_pid"] = None
    print("[Backend] All processes stopped.")
    return {"status": "stopped"}

@app.get("/health")
def health():
    ff_alive = ffmpeg_proc is not None and ffmpeg_proc.poll() is None
    htm_alive = htm_proc is not None and htm_proc.poll() is None
    return {**current_session, "ffmpeg_running": ff_alive, "htm_engine_running": htm_alive}

@app.get("/stats")
def stats(camera_id: str = Query(...)):
    try:
        current_vid = current_session.get("video_id")
        if not current_vid:
            return {"detected": 0, "events": []}

        # PASS CURRENT VIDEO ID
        htm_alerts = get_htm_alerts(current_vid)
        hash_alerts = get_hash_alerts(camera_id, current_vid)
        
        timeline = {}

        for h in hash_alerts:
            key = h["timestamp"]
            if key not in timeline:
                timeline[key] = {"timestamp": key, "frame": h["frame_index"], "htm": "NORMAL", "hash": "NORMAL", "verdict": "SAFE"}
            timeline[key]["hash"] = "⚠️ MATCH"

        for htm in htm_alerts:
            matched_key = None
            try:
                htm_time = datetime.fromisoformat(htm["timestamp"])
                for key in timeline:
                    try:
                        hash_time = datetime.fromisoformat(key)
                        if abs((htm_time - hash_time).total_seconds()) <= CORRELATION_WINDOW_SEC:
                            matched_key = key
                            break
                    except: continue
            except: continue
            
            status_str = "🚨 PANIC" if "CRITICAL" in htm.get("status", "") else "⚠️ WARN"
            if matched_key:
                timeline[matched_key]["htm"] = status_str
            else:
                key = htm["timestamp"]
                if key not in timeline:
                    timeline[key] = {"timestamp": key, "frame": htm["frame_index"], "htm": "NORMAL", "hash": "NORMAL", "verdict": "SAFE"}
                timeline[key]["htm"] = status_str

        final_log = []
        confirmed_count = 0
        for key, event in timeline.items():
            htm_bad = event["htm"] != "NORMAL"
            hash_bad = event["hash"] != "NORMAL"
            
            if htm_bad and hash_bad:
                event["verdict"] = "🔴 CONFIRMED"
                confirmed_count += 1
            elif htm_bad:
                event["verdict"] = "🟠 SUSPICIOUS"
            elif hash_bad:
                event["verdict"] = "🟡 UNVERIFIED"
            
            final_log.append(event)

        final_log.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"detected": confirmed_count, "events": final_log}
    except Exception as e:
        print(f"[Stats] Error: {e}")
        return {"detected": 0, "events": []}

@app.get("/verification_logs")
def get_verification_logs():
    return {"logs": verification_results}
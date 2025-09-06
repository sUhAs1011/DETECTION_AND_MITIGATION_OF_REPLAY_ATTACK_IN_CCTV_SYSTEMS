import cv2
import numpy as np
import subprocess
import os
import time
import sys

# ===== USER CONFIG =====
VIDEO_PATH = r"C:\Users\NITRO\OneDrive\Desktop\Capstone\Capstone_Code\Crime Footages\73.mp4"
RTSP_URL = "rtsp://localhost:8554/camera"

# Orchestration
AUTO_START_ORIGINAL = True          # Let the script start the original RTSP stream
AUTO_LAUNCH_FFPLAY = True           # Let the script launch ffplay to view the stream
RESTART_FFPLAY_ON_SWITCH = True     # Restart ffplay after switch for reliable reconnection

# Detection parameters
GRID_ROWS, GRID_COLS = 5, 5
BASELINE_SEC = 0.5                  # seconds of initial "normal" frames to build baseline
K_MAD = 1.0                         # sensitivity (lower = more sensitive)
MIN_CLUSTER_FRAC = 0.05             # fraction of tiles required as a connected cluster
PERSIST_START_SEC = 0.1             # seconds above threshold to confirm start
PERSIST_END_SEC = 0.8               # seconds below threshold to confirm end
SAFETY_MARGIN_SEC = 0.5             # extend cut on both sides (seconds)

# Encoding parameters
ENC_V = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-g", "30"]
ENC_A = ["-c:a", "aac", "-ar", "48000", "-b:a", "128k"]
# =======================


def tile_motion(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    H, W = diff.shape
    h, w = max(1, H // GRID_ROWS), max(1, W // GRID_COLS)
    out = np.zeros((GRID_ROWS, GRID_COLS), np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0, y1 = r * h, min((r + 1) * h, H)
            x0, x1 = c * w, min((c + 1) * w, W)
            tile = diff[y0:y1, x0:x1]
            out[r, c] = float(np.mean(tile)) if tile.size else 0.0
    return out


def largest_cluster(mask_bool):
    H, W = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    best = 0
    for r in range(H):
        for c in range(W):
            if mask_bool[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                size = 0
                while stack:
                    y, x = stack.pop()
                    size += 1
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < H and 0 <= nx < W and mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                best = max(best, size)
    return best


def detect_crime_window(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {path}")
        return None, None, 25.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total_duration = (total_frames / fps) if total_frames > 0 else float("inf")

    pstart_frames = max(1, int(PERSIST_START_SEC * fps))
    pend_frames = max(1, int(PERSIST_END_SEC * fps))
    baseline_frames = max(1, int(BASELINE_SEC * fps))

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return None, None, fps
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # Build baseline from initial normal frames
    baselines = []
    for _ in range(baseline_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        baselines.append(tile_motion(prev_gray, gray))
        prev_gray = gray

    if not baselines:
        cap.release()
        return None, None, fps

    base_arr = np.stack(baselines, axis=0)
    base_med = np.median(base_arr, axis=0)
    base_mad = np.median(np.abs(base_arr - base_med), axis=0) + 1e-6
    thresh = base_med + K_MAD * base_mad
    min_cluster = max(1, int(MIN_CLUSTER_FRAC * GRID_ROWS * GRID_COLS))

    start_frame = None
    end_frame = None
    above = 0
    below = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = tile_motion(prev_gray, gray)
        prev_gray = gray

        hot = (motion > thresh)
        cluster = largest_cluster(hot)

        if cluster >= min_cluster:
            above += 1
            below = 0
        else:
            below += 1
            above = 0

        if start_frame is None and above >= pstart_frames:
            # backdate to include persistence frames
            start_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - pstart_frames
        if start_frame is not None and end_frame is None and below >= pend_frames:
            end_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - pend_frames
            break

    cap.release()

    if start_frame is None:
        return None, None, fps
    if end_frame is None:
        end_frame = max(start_frame + 1, total_frames - 1) if total_frames > 0 else (start_frame + int(0.5 * fps))

    start_time = max(0.0, (start_frame / fps) - SAFETY_MARGIN_SEC)
    end_time = min(total_duration, (end_frame / fps) + SAFETY_MARGIN_SEC)
    return start_time, end_time, fps


def create_replay_file(path, t0, t1, fps, out_path="output_no_crime.mp4"):
    dur = max(0.04, t1 - t0)
    print(f"[INFO] Building replay file: replace {t0:.3f}s → {t1:.3f}s (dur {dur:.3f}s)")

    # Head: 0 -> t0
    subprocess.run([
        "ffmpeg", "-y", "-i", path, "-ss", "0", "-to", f"{t0:.3f}",
        *ENC_V, *ENC_A, "head.mp4"
    ], check=True)

    # Loop source: 0 -> t0 (pre-crime)
    subprocess.run([
        "ffmpeg", "-y", "-i", path, "-ss", "0", "-to", f"{t0:.3f}",
        *ENC_V, *ENC_A, "loop_src.mp4"
    ], check=True)

    # Filler: loop pre-crime to match duration
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", "loop_src.mp4",
        "-t", f"{dur:.3f}", *ENC_V, *ENC_A, "filler.mp4"
    ], check=True)

    # Tail: t1 -> end
    subprocess.run([
        "ffmpeg", "-y", "-i", path, "-ss", f"{t1:.3f}",
        *ENC_V, *ENC_A, "tail.mp4"
    ], check=True)

    # Concat parts (all consistently encoded)
    with open("concat.txt", "w") as f:
        f.write("file 'head.mp4'\n")
        f.write("file 'filler.mp4'\n")
        f.write("file 'tail.mp4'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
        *ENC_V, *ENC_A, "-r", f"{fps:.3f}", out_path
    ], check=True)

    # Cleanup temp parts
    for f in ["head.mp4", "loop_src.mp4", "filler.mp4", "tail.mp4", "concat.txt"]:
        try:
            os.remove(f)
        except OSError:
            pass

    print(f"[INFO] Replay file ready: {out_path}")
    return out_path


def start_original_stream(video_path, rtsp_url):
    # Use encoding for stable parameters
    cmd = [
        "ffmpeg", "-re", "-stream_loop", "-1", "-i", video_path,
        "-c:v", "libx264", "-preset", "veryfast", "-g", "30",
        "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
        "-f", "rtsp", rtsp_url
    ]
    return subprocess.Popen(cmd)


def start_replay_stream(replay_path, rtsp_url):
    cmd = [
        "ffmpeg", "-re", "-stream_loop", "-1", "-i", replay_path,
        "-c:v", "libx264", "-preset", "veryfast", "-g", "30",
        "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
        "-f", "rtsp", rtsp_url
    ]
    return subprocess.Popen(cmd)


def start_ffplay(rtsp_url):
    # Low-latency flags to help reconnection
    return subprocess.Popen(["ffplay", "-fflags", "nobuffer", "-flags", "low_delay", rtsp_url])


def terminate_process(proc, timeout=3):
    if not proc:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        pass


if __name__ == "__main__":
    # 1) Detect crime window
    t0, t1, fps = detect_crime_window(VIDEO_PATH)
    if t0 is None:
        print("[INFO] No crime detected with current thresholds. Aborting.")
        sys.exit(0)

    print(f"[INFO] Crime window (with safety): {t0:.2f}s → {t1:.2f}s")

    # 2) Build the replay file
    replay_path = create_replay_file(VIDEO_PATH, t0, t1, fps)

    # Sanity-check replay file
    if not os.path.exists(replay_path) or os.path.getsize(replay_path) == 0:
        print("[ERROR] Replay file missing or empty. Aborting.")
        sys.exit(1)

    # 3) Start original RTSP stream and viewer
    orig_ffmpeg = None
    viewer = None
    stream_start_time = time.time()

    if AUTO_START_ORIGINAL:
        print("[INFO] Starting original RTSP stream...")
        orig_ffmpeg = start_original_stream(VIDEO_PATH, RTSP_URL)
        stream_start_time = time.time()
    else:
        print("[INFO] Assuming original RTSP stream is already running...")

    if AUTO_LAUNCH_FFPLAY:
        print("[INFO] Launching ffplay viewer...")
        viewer = start_ffplay(RTSP_URL)

    # 4) Wait until crime start time relative to when streaming began
    elapsed = time.time() - stream_start_time
    wait_secs = max(0.0, t0 - elapsed)
    if wait_secs > 0:
        print(f"[INFO] Waiting {wait_secs:.2f}s before switching to replay...")
        time.sleep(wait_secs)

    # 5) Switch to replay: stop original, brief pause, start replay, restart viewer
    print("[INFO] Switching to replay stream...")
    if orig_ffmpeg is not None:
        terminate_process(orig_ffmpeg)
    else:
        # If original not spawned by us (manual), try to kill any ffmpeg (Windows)
        if os.name == "nt":
            os.system("taskkill /IM ffmpeg.exe /F")

    # Give RTSP server time to free the mount
    time.sleep(1.5)

    print("[INFO] Starting replay RTSP stream...")
    replay_ffmpeg = start_replay_stream(replay_path, RTSP_URL)

    # Restart viewer to ensure it connects to the new publisher
    if viewer is not None and RESTART_FFPLAY_ON_SWITCH:
        terminate_process(viewer, timeout=2)
        time.sleep(1.0)  # let replay publisher come up
        viewer = start_ffplay(RTSP_URL)

    print("[INFO] Replay stream running. Press Ctrl+C to stop.")

    # 6) Keep running until user interrupts
    try:
        while True:
            # Optionally, monitor child processes
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping replay stream and cleaning up...")
        terminate_process(replay_ffmpeg)
        terminate_process(viewer)

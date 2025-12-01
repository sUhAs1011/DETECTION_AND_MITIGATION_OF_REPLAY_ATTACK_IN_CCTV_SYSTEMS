import cv2
import sqlite3
import hashlib
from PIL import Image
import imagehash
from hashing_module.config.config import DB_PATH, FRAME_SIZE, TO_GRAY, HASH_ALGO
from hashing_module.db.schema import init_db

def normalize_frame(frame):
    if TO_GRAY:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return frame

def crypto_hash(frame):
    frame_bytes = frame.tobytes()
    if HASH_ALGO == "sha256":
        return hashlib.sha256(frame_bytes).hexdigest()
    elif HASH_ALGO == "blake2b":
        return hashlib.blake2b(frame_bytes, digest_size=32).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm")

def perceptual_hash(frame):
    pil_img = Image.fromarray(frame)
    return str(imagehash.phash(pil_img))  # store as hex string

def live_hash(stream_url="rtsp://localhost:8554/camera",
              camera_id="CAM10", video_id="VID010",
              max_frames=None, max_seconds=None):
    """
    Live hashing:
    - Stores both cryptographic and perceptual hashes for each frame.
    - Video files stop automatically when they end.
    - RTSP streams run until 'q' is pressed or max_frames/max_seconds reached.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {stream_url}")

    frame_idx = 0
    print(f"[INFO] Starting live hashing on {stream_url}")

    start_time = cv2.getTickCount() / cv2.getTickFrequency()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended or cannot grab frame. Stopping.")
            break

        frame_idx += 1
        ts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Normalize + hash
        norm = normalize_frame(frame)
        crypto_h = crypto_hash(norm)
        phash_h = perceptual_hash(norm)

        # Store both hashes in DB
        cursor.execute("""
            INSERT INTO frame_hashes (camera_id, video_id, frame_number, timestamp,
                                      hash, phash, hash_algo,
                                      gray, width, height, serialize_mode, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (camera_id, video_id, frame_idx, ts_sec,
              crypto_h, phash_h, HASH_ALGO,
              int(TO_GRAY), FRAME_SIZE[0], FRAME_SIZE[1], "raw", "live"))
        conn.commit()

        # Show live feed
        cv2.imshow("Live Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User stopped live hashing.")
            break

        # Stop after max_frames
        if max_frames and frame_idx >= max_frames:
            print(f"[INFO] Reached max_frames={max_frames}. Stopping.")
            break

        # Stop after max_seconds
        elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - start_time
        if max_seconds and elapsed >= max_seconds:
            print(f"[INFO] Reached max_seconds={max_seconds}. Stopping.")
            break

    cap.release()
    conn.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Live hashing stopped after {frame_idx} frames.")

if __name__ == "__main__":
    live_hash("rtsp://localhost:8554/camera",
              camera_id="CAM10",
              video_id="VID010",
              max_seconds=60)

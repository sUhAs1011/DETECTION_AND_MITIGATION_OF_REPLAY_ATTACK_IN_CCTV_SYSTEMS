import cv2
import sqlite3
import hashlib
from PIL import Image
import imagehash
from tqdm import tqdm
from hashing_module.config.config import DB_PATH, FRAME_SIZE, TO_GRAY, HASH_ALGO
from hashing_module.db.schema import init_db

def normalize_frame(frame):
    if TO_GRAY:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if FRAME_SIZE:
        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return frame

def crypto_hash(frame):
    """Cryptographic hash for forensic integrity (exact match)."""
    frame_bytes = frame.tobytes()
    if HASH_ALGO == "sha256":
        return hashlib.sha256(frame_bytes).hexdigest()
    elif HASH_ALGO == "blake2b":
        return hashlib.blake2b(frame_bytes, digest_size=32).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm")

def perceptual_hash(frame):
    """Perceptual hash for similarity detection (robust to encoding changes)."""
    pil_img = Image.fromarray(frame)
    return imagehash.phash(pil_img)

def live_verify(stream_url="rtsp://localhost:8554/camera",
                camera_id="CAM05", video_id="LIVE",
                tolerance=0.05, min_run=20, phash_threshold=5):
    """
    Live verification:
    - Hash incoming frames with both cryptographic and perceptual hashes.
    - Compare perceptual hashes against baseline for replay detection.
    - Raise alert if replay detected.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {stream_url}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    consecutive = 0
    mismatches = 0
    replay_detected = False

    print(f"[INFO] Starting live verification on {stream_url}")

    # Progress bar
    if total_frames > 0:
        pbar = tqdm(total=total_frames, desc="Verifying frames", unit="frame")
    else:
        pbar = tqdm(desc="Verifying frames", unit="frame")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended. Stopping verification.")
                break

            frame_idx += 1
            ts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Normalize frame
            norm = normalize_frame(frame)

            # Compute hashes
            crypto_h = crypto_hash(norm)
            phash_h = perceptual_hash(norm)

            # Check perceptual similarity against baseline
            cursor.execute("SELECT hash FROM frame_hashes WHERE video_id!=?", (video_id,))
            baseline_hashes = cursor.fetchall()

            match_found = False
            for (baseline_hash,) in baseline_hashes:
                # Convert baseline hash string back to imagehash object if stored
                try:
                    baseline_phash = imagehash.hex_to_hash(baseline_hash)
                    if abs(phash_h - baseline_phash) <= phash_threshold:
                        match_found = True
                        break
                except Exception:
                    # Skip if baseline hash is cryptographic
                    continue

            if match_found:
                consecutive += 1
            else:
                mismatches += 1
                if mismatches > tolerance * consecutive:
                    consecutive = 0
                    mismatches = 0

            if consecutive >= min_run:
                print(f"[ALERT] Replay detected at frame {frame_idx} (≥{min_run} consecutive perceptual matches).")
                replay_detected = True
                consecutive = 0
                mismatches = 0

            # Update progress bar
            pbar.update(1)

            # Optional: show live feed
            cv2.imshow("Verification Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] User stopped verification.")
                break

    except KeyboardInterrupt:
        print("[INFO] Verification interrupted by user.")

    cap.release()
    conn.close()
    cv2.destroyAllWindows()
    pbar.close()

    print(f"[INFO] Verification stopped after {frame_idx} frames.")
    if replay_detected:
        print("[RESULT] Replay attack detected in the footage.")
    else:
        print("[RESULT] No replay attack detected in the footage.")

if __name__ == "__main__":
    live_verify("rtsp://localhost:8554/camera",
                camera_id="CAM10",
                video_id="VID010",
                tolerance=0.05,
                min_run=20,
                phash_threshold=5)

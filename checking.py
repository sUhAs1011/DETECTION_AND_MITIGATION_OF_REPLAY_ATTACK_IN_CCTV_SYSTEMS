import sqlite3
from hashing_module.src.hasher import hash_video_frames
from hashing_module.db.schema import init_db
from hashing_module.config.config import DB_PATH

def verify_video(video_path, camera_id="CAM01", min_contiguous=30):
    """
    Hashes a new video and checks for matches against stored frame hashes.
    Returns a list of matches: (new_frame_no, new_hash, old_video_id, old_frame_no).
    """
    # Step 1: Hash the new video
    new_records = hash_video_frames(video_path, camera_id, video_id="VERIFY_RUN")

    # Step 2: Compare against DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    matches = []
    for rec in new_records:
        frame_no, h = rec[2], rec[4]  # frame_number, hash
        cur.execute("SELECT video_id, frame_number FROM frame_hashes WHERE hash=?", (h,))
        rows = cur.fetchall()
        if rows:
            matches.extend([(frame_no, h, r[0], r[1]) for r in rows])

    conn.close()
    return matches

def main():
    # 1. Ensure DB schema exists
    init_db()

    # 2. Path to the video you want to verify
    video_path = r"C:\Users\NITRO\OneDrive\Desktop\Capstone\Capstone_Code\Crime Footages\25.mp4"

    # 3. Run verification
    matches = verify_video(video_path, camera_id="CAM01")

    # 4. Print results
    print(f"Found {len(matches)} matching frames in DB")
    if matches:
        print("Sample matches:")
        for m in matches[:10]:
            print(f"New frame {m[0]} matched hash {m[1][:12]}... "
                  f"from stored video {m[2]} frame {m[3]}")

if __name__ == "__main__":
    main()

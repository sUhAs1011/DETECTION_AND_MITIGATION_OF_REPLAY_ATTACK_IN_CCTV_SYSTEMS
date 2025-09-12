import os
import glob
import subprocess

# ------------- Configuration ----------------
ENCODED_SDR_FOLDER = r"C:\Users\Shreyas S\Documents\htm.core\capstone\encoder_test\encoded_sdrs"       # Folder containing encoded SDR .npz files
CONVERTED_HTM_FOLDER = r"C:\Users\Shreyas S\Documents\htm.core\capstone\encoder_test\converted_htm"    # Output folder for converted HTM SDRs
OUT_SIZE = 2048                                    # Length of global SDR (M)
TARGET_ON = 40                                     # Target active bits (K)
SEED = 42                                         # Hash seed for projection
# ---------------------------------------------

os.makedirs(CONVERTED_HTM_FOLDER, exist_ok=True)

# Find all encoded SDR .npz files
encoded_files = glob.glob(os.path.join(ENCODED_SDR_FOLDER, "*_sdr.npz"))
if not encoded_files:
    print("No encoded SDR .npz files found in:", ENCODED_SDR_FOLDER)
    exit(1)

for npz_path in encoded_files:
    base_name = os.path.splitext(os.path.basename(npz_path))[0]  # e.g. video1_sdr
    out_name = base_name.replace("_sdr", f"_M{OUT_SIZE}_K{TARGET_ON}") + ".npz"
    out_path = os.path.join(CONVERTED_HTM_FOLDER, out_name)
    meta_path = out_path.replace(".npz", ".meta.json")

    cmd = [
        "python",
        "convert_encodings_to_htm_input.py",
        "--input", npz_path,
        "--out", out_path,
        "--out-meta", meta_path,
        "--out-size", str(OUT_SIZE),
        "--target-on", str(TARGET_ON),
        "--seed", str(SEED),
    ]
    print(f"Processing {npz_path} → {out_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error processing {npz_path}:\n{result.stderr}")
        break

print("✅ All files processed.")

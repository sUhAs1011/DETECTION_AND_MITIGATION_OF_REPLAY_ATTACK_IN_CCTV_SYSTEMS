# main_encoder.py
import os
import sys
import json
import numpy as np
import pickle
from encoder import OpticalFlowEncoder
from tqdm import tqdm

# --- Configuration ---
INPUT_FOLDER = r"C:\Users\Shreyas S\Documents\htm.core\capstone\data\optical_flow_data"  # Folder containing .npz files
OUTPUT_FOLDER = "encoded_sdrs"      # Folder to save SDRs (.npz + meta.json)
GRID_ROWS, GRID_COLS = 5, 5
MAX_MAGNITUDE = 15.0  # Used to normalize magnitude in encoder

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
encoder = OpticalFlowEncoder(max_magnitude=MAX_MAGNITUDE)

def process_video(npz_path, grid_rows=GRID_ROWS, grid_cols=GRID_COLS):
    """
    Load a .npz file containing optical flow, slice it into grid cells,
    encode each cell using the OpticalFlowEncoder, and return an object-array
    indices structure plus metadata.
    """
    # Memory-safe load
    npz = np.load(npz_path, mmap_mode='r')
    if 'motion_vectors' not in npz:
        raise KeyError("npz file does not contain 'motion_vectors' key: " + npz_path)

    flows = npz['motion_vectors']  # shape (T, H, W, 2) but mapped on disk
    T = int(flows.shape[0])
    H = int(flows.shape[1])
    W = int(flows.shape[2])

    cell_h = H // grid_rows
    cell_w = W // grid_cols
    cells_per_frame = grid_rows * grid_cols

    # Build an object array to hold index arrays per (frame, cell)
    indices = np.empty((T, cells_per_frame), dtype=object)

    # process frames one by one to limit memory usage
    for t in tqdm(range(T), desc=f"Encoding {os.path.basename(npz_path)}"):
        flow = flows[t]  # this is a view into mmap, not a full copy
        cell_idx = 0
        for i in range(grid_rows):
            for j in range(grid_cols):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_rows - 1 else H
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_cols - 1 else W
                patch = flow[y_start:y_end, x_start:x_end]  # small view

                sdr = encoder.encode(patch)
                idx = np.array(sdr.sparse, dtype=np.int32) if getattr(sdr, "sparse", None) is not None else np.array([], dtype=np.int32)
                indices[t, cell_idx] = idx
                cell_idx += 1

    # metadata
    meta = {
        "source_npz": os.path.basename(npz_path),
        "frames": T,
        "height": H,
        "width": W,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "cells_per_frame": cells_per_frame,
        "encoder": {
            "max_magnitude": encoder.max_magnitude,
            "sin_size": encoder.sin_size,
            "cos_size": encoder.cos_size,
            "mag_size": encoder.mag_size,
            "total_size": encoder.size,
            "active_bits_expected": encoder.active_bits_expected
        },
        "notes": "indices stored as object-array of int32 arrays per (frame, cell)"
    }

    return indices, meta

def main():
    npz_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".npz")]
    if not npz_files:
        print(f"❌ No .npz files found in {INPUT_FOLDER}")
        return

    for file in npz_files:
        npz_path = os.path.join(INPUT_FOLDER, file)
        indices, meta = process_video(npz_path, grid_rows=GRID_ROWS, grid_cols=GRID_COLS)

        base = os.path.splitext(os.path.basename(npz_path))[0]
        out_npz = os.path.join(OUTPUT_FOLDER, f"{base}_sdr.npz")
        meta_json = os.path.join(OUTPUT_FOLDER, f"{base}_meta.json")

        # Save indices into a compressed npz (object array) and metadata separately
        # Note: object arrays will be pickled by numpy under the hood, but this is still
        # simpler and more portable than pickling SDR objects.
        np.savez_compressed(out_npz, indices=indices)
        with open(meta_json, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Encoded SDRs saved to: {out_npz}")
        print(f"✅ Metadata saved to: {meta_json}")
        print(f"  frames={meta['frames']}, cells/frame={meta['cells_per_frame']}, encoder.total_size={meta['encoder']['total_size']}")

    print("🎉 All videos encoded and saved successfully.")

if __name__ == "__main__":
    main()

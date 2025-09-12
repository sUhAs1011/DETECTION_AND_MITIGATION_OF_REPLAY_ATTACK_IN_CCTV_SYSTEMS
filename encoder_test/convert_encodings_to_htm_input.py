#!/usr/bin/env python3
"""
convert_encoded_to_htm_inputs.py

Convert your per-frame per-cell encoded SDRs (pickle list-of-frames or .npz) into
fixed-size, fixed-sparsity HTM-ready SDR index arrays using the per-cell block
allocation + deterministic hashing scheme.

Output: compressed .npz with:
  - indices: shape (T, K) dtype=int32 (sorted ascending per row)
  - meta.json (sidecar) with parameters & provenance

Usage:
  python convert_encoded_to_htm_inputs.py --input encoded_sdrs/video1_sdr.npz \
      --out converted_htm/video1_M2048_K40.npz --out-meta converted_htm/video1_M2048_K40.meta.json

Default params chosen for demo: M=2048, K=40, seed=42
"""
from __future__ import annotations
import argparse, os, json, pickle
from typing import List, Sequence, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from project_utils import project_frame_to_indices, load_encoded_file

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="encoded input (.npz or .pkl) produced by main_encoder")
    ap.add_argument("--out", "-o", required=True, help="output .npz path (contains indices array)")
    ap.add_argument("--out-meta", "-m", required=False, help="output meta json path (if omitted, same name + .meta.json)")
    ap.add_argument("--out-size", type=int, default=2048, help="length of final SDR (M)")
    ap.add_argument("--target-on", type=int, default=40, help="global target ON bits (K)")
    ap.add_argument("--seed", type=int, default=42, help="deterministic seed for projection")
    ap.add_argument("--cells", type=int, default=None, help="cells per frame override (if known)")
    ap.add_argument("--per-cell-on", type=int, default=None, help="force fixed ON bits per cell (optional)")
    return ap.parse_args()

def convert_file(input_path: str, out_path: str, out_meta_path: Optional[str],
                 out_size: int, target_on: int, seed: int, cells_override: Optional[int], per_cell_on: Optional[int]) -> None:

    data, format_hint = load_encoded_file(input_path)
    # data: list of frames, each frame is list-of-cells where each cell is int array of indices
    T = len(data)
    cells_per_frame = len(data[0]) if T > 0 else (cells_override or 0)
    if cells_override is not None:
        cells_per_frame = cells_override

    if per_cell_on is None:
        per_cell_on = int(np.ceil(target_on / max(1, cells_per_frame)))

    # unconditionally compute block_size as ceil(out_size / cells_per_frame) and recompute actual_out_size
    block_size = int(np.ceil(out_size / max(1, cells_per_frame)))
    actual_out_size = block_size * max(1, cells_per_frame)
    if actual_out_size != out_size:
        print(f"[info] Adjusted out_size {out_size} -> {actual_out_size} to fit integer blocks (block_size={block_size})")
        out_size = actual_out_size

    # results: (T, target_on)
    out_indices = np.zeros((T, target_on), dtype=np.int32)

    for t in range(T):
        frame = data[t]
        selected = project_frame_to_indices(frame,
                                            out_size=out_size,
                                            cells=cells_per_frame,
                                            per_cell_on=per_cell_on,
                                            block_size=block_size,
                                            seed=seed,
                                            target_on=target_on,
                                            frame_index=t)
        out_indices[t, :] = np.array(selected, dtype=np.int32)

    # save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, indices=out_indices)

    # meta
    meta = {
        "input_path": str(input_path),
        "format_hint": format_hint,
        "T": int(T),
        "cells_per_frame": int(cells_per_frame),
        "out_size": int(out_size),
        "target_on": int(target_on),
        "per_cell_on": int(per_cell_on),
        "block_size": int(block_size),
        "hash_seed": int(seed),
        "converter": "per-cell-block-v1"
    }
    meta_path = out_meta_path or (str(Path(out_path).with_suffix('')) + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved converted file: {out_path}")
    print(f"Saved meta: {meta_path}")

def main():
    args = parse_args()
    convert_file(args.input, args.out, args.out_meta,
                 out_size=args.out_size, target_on=args.target_on, seed=args.seed,
                 cells_override=args.cells, per_cell_on=args.per_cell_on)

if __name__ == "__main__":
    main()

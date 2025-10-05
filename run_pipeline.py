#!/usr/bin/env python3
"""
run_full_pipeline.py

End-to-end orchestrator: videos → optical flow → per-cell SDRs → HTM-ready SDRs → HTM training.

Reuses existing project modules without duplicating logic:
- pre_processing/pre-process.py : process_videos(...)
- encoder_test/main_encoder.py : process_video(...) + save logic
- encoder_test/convert_encodings_to_htm_input.py : convert_file(...)
- htm/htm_train.py : main() for training

Usage examples:
  python run_full_pipeline.py \
    --videos data/normal_videos1 \
    --flow-out data/optical_flow_data \
    --encoded-out encoder_test/encoded_sdrs \
    --converted-out encoder_test/converted_htm \
    --epochs 8

Notes:
- Idempotent: skips existing artifacts by default unless --overwrite is set.
- Assumes repository layout as in this project.
"""

from __future__ import annotations
import argparse
import os
import sys
import glob
import json
import importlib.util
from pathlib import Path
import numpy as np

# ------------------------- Helpers -------------------------

def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description="End-to-end pipeline: videos → HTM training")
    ap.add_argument("--videos", required=True, help="Input folder containing raw videos")
    ap.add_argument("--flow-out", default="data/optical_flow_data", help="Output folder for optical flow .npz")
    ap.add_argument("--encoded-out", default="encoder_test/encoded_sdrs", help="Output folder for per-cell SDR encodings")
    ap.add_argument("--converted-out", default="encoder_test/converted_htm", help="Output folder for HTM-ready SDRs")
    ap.add_argument("--grid-rows", type=int, default=5)
    ap.add_argument("--grid-cols", type=int, default=5)
    ap.add_argument("--max-magnitude", type=float, default=15.0)
    ap.add_argument("--out-size", type=int, default=2048, help="Global SDR size M")
    ap.add_argument("--target-on", type=int, default=40, help="Global ON bits K")
    ap.add_argument("--hash-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--model-dir", default="htm/models")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    return ap.parse_args()


def main():
    args = parse_args()

    videos_dir = os.path.abspath(args.videos)
    flow_out_dir = os.path.abspath(args.flow_out)
    encoded_out_dir = os.path.abspath(args.encoded_out)
    converted_out_dir = os.path.abspath(args.converted_out)
    model_dir = os.path.abspath(args.model_dir)

    ensure_dir(flow_out_dir)
    ensure_dir(encoded_out_dir)
    ensure_dir(converted_out_dir)
    ensure_dir(model_dir)

    # ------------------ Stage 1-3: Sequential per-video processing with safe cleanup ------------------
    print("\n=== Stage 1: Optical Flow Extraction ===")
    preproc_path = os.path.join("pre_processing", "pre-process.py")
    preproc_mod = load_module_from_path("pre_process_mod", preproc_path)

    print("\n=== Stage 2: Encoding per-cell SDRs ===")
    sys.path.insert(0, str(Path("encoder_test").resolve()))
    import encoder_test.main_encoder as enc_main  # reuses existing code

    # Configure encoder module globals for this run
    enc_main.INPUT_FOLDER = flow_out_dir
    enc_main.OUTPUT_FOLDER = encoded_out_dir
    enc_main.GRID_ROWS = args.grid_rows
    enc_main.GRID_COLS = args.grid_cols
    enc_main.MAX_MAGNITUDE = args.max_magnitude

    print("\n=== Stage 3: Convert to HTM-ready SDRs ===")
    from encoder_test.convert_encodings_to_htm_input import convert_file

    # Discover input videos
    video_exts = (".mp4", ".avi", ".mov")
    video_files = [f for f in sorted(os.listdir(videos_dir)) if f.lower().endswith(video_exts)]
    if not video_files:
        print("No videos found in input folder. Exiting.")
        sys.exit(1)

    for vid in video_files:
        vid_path = os.path.join(videos_dir, vid)
        base_name = os.path.splitext(vid)[0]
        flow_npz = os.path.join(flow_out_dir, f"{base_name}_flow.npz")
        enc_npz = os.path.join(encoded_out_dir, f"{base_name}_flow_sdr.npz")
        enc_meta = os.path.join(encoded_out_dir, f"{base_name}_flow_meta.json")
        conv_npz = os.path.join(converted_out_dir, f"{base_name}_flow_M{args.out_size}_K{args.target_on}.npz")
        conv_meta = conv_npz.replace(".npz", ".meta.json")

        try:
            # Fast-path idempotency checks to enable safe resume without redoing work
            has_converted = (os.path.exists(conv_npz) and os.path.exists(conv_meta))
            has_encoded = (os.path.exists(enc_npz) and os.path.exists(enc_meta))

            if has_converted and not args.overwrite:
                # We have final artifacts → no need to extract or encode. Cleanup leftover flow if present.
                print(f"⏭️  Skipping flow extract and encode (converted exists): {conv_npz}")
                if os.path.exists(flow_npz):
                    try:
                        os.remove(flow_npz)
                        print(f"🗑️  Deleted optical flow: {flow_npz}")
                    except Exception as del_err:
                        print(f"[warn] Could not delete optical flow {flow_npz}: {del_err}")
                # proceed to next video
                continue

            # If we already have encoded outputs (but not converted), skip extraction and encoding
            if has_encoded and not args.overwrite:
                print(f"⏭️  Skipping flow extract and encode (encoded exists): {enc_npz}")
            else:
                # --- Extract optical flow just for this video ---
                need_flow = args.overwrite or (not os.path.exists(flow_npz))
                if need_flow:
                    # Use a staging folder to reuse folder-based preprocessor for a single video
                    import shutil
                    staging_dir = os.path.join(flow_out_dir, "_staging_single_video")
                    os.makedirs(staging_dir, exist_ok=True)
                    staged_path = os.path.join(staging_dir, vid)
                    if os.path.exists(staged_path):
                        os.remove(staged_path)
                    shutil.copy2(vid_path, staged_path)

                    preproc_mod.process_videos(
                        video_folder=staging_dir,
                        output_folder=flow_out_dir,
                        target_width=640,
                        target_height=480,
                        normalized_fps=30,
                    )
                    # cleanup staging
                    try:
                        os.remove(staged_path)
                        os.rmdir(staging_dir)
                    except Exception:
                        pass
                else:
                    print(f"⏭️  Skipping flow extract (exists): {flow_npz}")

            # --- Encode per-cell SDRs for this flow ---
            if has_encoded and not args.overwrite:
                print(f"⏭️  Skipping encode (exists): {enc_npz}")
            else:
                indices, meta = enc_main.process_video(flow_npz, grid_rows=args.grid_rows, grid_cols=args.grid_cols)
                np.savez_compressed(enc_npz, indices=indices)
                with open(enc_meta, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"✅ Encoded: {enc_npz}")

            # --- Convert to HTM-ready SDRs for this video ---
            conversion_done = False
            if has_converted and not args.overwrite:
                print(f"⏭️  Skipping convert (exists): {conv_npz}")
                conversion_done = True
            else:
                convert_file(
                    input_path=enc_npz,
                    out_path=conv_npz,
                    out_meta_path=conv_meta,
                    out_size=args.out_size,
                    target_on=args.target_on,
                    seed=args.hash_seed,
                    cells_override=None,
                    per_cell_on=None,
                )
                conversion_done = True

            # --- Safe cleanup: delete only this video's flow npz after successful conversion ---
            if conversion_done and os.path.exists(flow_npz):
                try:
                    os.remove(flow_npz)
                    print(f"🗑️  Deleted optical flow: {flow_npz}")
                except Exception as del_err:
                    print(f"[warn] Could not delete optical flow {flow_npz}: {del_err}")

        except Exception as e:
            print(f"❌ Error processing video '{vid}': {e}. Continuing to next video.")
            continue

    # ------------------ Stage 4: Train HTM ------------------
    print("\n=== Stage 4: HTM Training (SP+TM) ===")
    # Load trainer directly from file to avoid package import issues
    trainer_path = os.path.join("htm", "htm_train.py")
    trainer = load_module_from_path("htm_train_mod", trainer_path)

    # Prepare argv for trainer.main() without changing trainer code
    argv = [
        "htm_train.py",
        "--input-size", str(args.out_size),
        "--active-bits", str(args.target_on),
        "--epochs", str(args.epochs),
        "--model-dir", model_dir,
        "--cells-dir", converted_out_dir,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        trainer.main()
    finally:
        sys.argv = old_argv

    print("\n🎉 Pipeline complete.")


if __name__ == "__main__":
    main()

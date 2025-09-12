# htm_train.py
# Train SP+TM on converted HTM SDR frames from a folder of .npz files, then save models + metadata.

from __future__ import annotations
import argparse, json, sys, platform, os, pathlib
import numpy as np
import glob
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
import csv
from datetime import datetime

def write_csv_header(filepath):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "epoch", "avg_anomaly", "std_anomaly",
            "tail_avg_anomaly", "tail_std_anomaly", "avg_active_columns"
        ])

def append_csv_row(filepath, row):
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def run_final_tail_check(sp, tm, train_frames, num_tail_frames=10):
    tm.reset()
    print("\nFinal Tail check on training stream (learn=False):")
    for i, sdr in enumerate(train_frames[-num_tail_frames:], 1):
        active_cols = SDR(sp.getColumnDimensions())
        sp.compute(sdr, learn=False, output=active_cols)
        tm.compute(active_cols, learn=False)
        print(f"  Step {i:02d}: anomaly={tm.anomaly:.3f}")

def parse_args():
    ap = argparse.ArgumentParser(description="Train HTM on converted HTM SDR frames and save models.")
    ap.add_argument("--input-size", type=int, default=2050)
    ap.add_argument("--active-bits", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="seed for initial SDR and evolution")
    ap.add_argument("--model-dir", type=str, default="models")
    ap.add_argument("--cells-dir", type=str, default="converted_htm", help="Folder containing converted HTM SDR .npz files")
    ap.add_argument("--num-columns", type=int, default=2048)
    ap.add_argument("--cells-per-column", type=int, default=32)
    ap.add_argument("--tm-activation-threshold", type=int, default=10)
    ap.add_argument("--tm-min-threshold", type=int, default=8)
    ap.add_argument("--tm-max-new-synapse", type=int, default=32)
    ap.add_argument("--save-tail", type=int, default=200, help="Save last N training frames to .npy for sanity checks")
    ap.add_argument("--log-csv", type=str, default="training_log.csv", help="CSV file to save training logs")
    ap.add_argument("--tail-check-frames", type=int, default=50, help="Frames count for tail anomaly check")
    return ap.parse_args()


def load_converted_sdrs_from_folder(folder: str, input_size: int) -> list[SDR]:
    """
    Loads all .npz files in the given folder, concatenates frames as SDRs.
    """
    all_frames = []
    npz_files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in folder: {folder}")
    for path in npz_files:
        print(f"Loading converted SDR frames from: {path}")
        data = np.load(path)
        indices = data['indices']  # shape (T, K)
        for row in indices:
            sdr = SDR(input_size)
            sdr.sparse = np.sort(row.astype(np.int32))
            all_frames.append(sdr)
    return all_frames


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # Load all converted SDR frames from folder
    train_frames = load_converted_sdrs_from_folder(args.cells_dir, input_size=args.input_size)
    print(f"Total training frames loaded: {len(train_frames)}")

    # ---- HTM model ----
    sp = SpatialPooler(
        inputDimensions=(args.input_size,),
        columnDimensions=(args.num_columns,),
        potentialPct=0.8,
        globalInhibition=True,
        synPermInactiveDec=0.005,
        synPermActiveInc=0.04,
        synPermConnected=0.1,
        minPctOverlapDutyCycle=0.001,
        dutyCyclePeriod=1000,
        boostStrength=3.0
    )

    tm = TemporalMemory(
        columnDimensions=(args.num_columns,),
        cellsPerColumn=args.cells_per_column,
        activationThreshold=args.tm_activation_threshold,
        initialPermanence=0.21,
        connectedPermanence=0.5,
        minThreshold=args.tm_min_threshold,
        maxNewSynapseCount=args.tm_max_new_synapse,
        permanenceIncrement=0.1,
        permanenceDecrement=0.1,
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=128,
        maxSynapsesPerSegment=128
    )


    # ---- Training ----
    print(f"--- 🧠 Training HTM on Normal Stream ---")

    write_csv_header(args.log_csv)

    for e in range(args.epochs):
        tm.reset()
        anomaly_scores = []
        active_columns_counts = []

        for pattern in train_frames:
            active_columns = SDR(sp.getColumnDimensions())
            sp.compute(pattern, learn=True, output=active_columns)
            tm.compute(active_columns, learn=True)

            anomaly_scores.append(tm.anomaly)
            active_columns_counts.append(int(active_columns.dense.sum()))

        avg_anomaly = np.mean(anomaly_scores)
        std_anomaly = np.std(anomaly_scores)
        avg_active_cols = np.mean(active_columns_counts)

        # Tail anomaly check on last frames with learn=False
        tm.reset()
        tail_n = min(args.tail_check_frames, len(train_frames))
        tail_anomalies = []
        for sdr in train_frames[-tail_n:]:
            active_cols = SDR(sp.getColumnDimensions())
            sp.compute(sdr, learn=False, output=active_cols)
            tm.compute(active_cols, learn=False)
            tail_anomalies.append(tm.anomaly)

        tail_avg = np.mean(tail_anomalies)
        tail_std = np.std(tail_anomalies)

        # Print summary
        print(f"Epoch {e+1}/{args.epochs} completed.")
        print(f"  Avg anomaly: {avg_anomaly:.4f} ± {std_anomaly:.4f}")
        print(f"  Avg active cols: {avg_active_cols:.2f}")
        print(f"  Tail anomaly (last {tail_n} frames): {tail_avg:.4f} ± {tail_std:.4f}\n")

        append_csv_row(args.log_csv, [
            datetime.now().isoformat(),
            e + 1,
            avg_anomaly,
            std_anomaly,
            tail_avg,
            tail_std,
            avg_active_cols
        ])

    run_final_tail_check(sp, tm, train_frames, num_tail_frames=10)

    # Save tail of training stream (exact SDR indices) for sanity checks
    tail_n = min(args.save_tail, len(train_frames))
    tail = np.stack([np.array(p.sparse, dtype=np.int32) for p in train_frames[-tail_n:]])
    np.save(os.path.join(args.model_dir, "train_tail.npy"), tail)


    # ---- Save models ----
    import pickle
    sp_path = os.path.join(args.model_dir, "sp_model.pkl")
    tm_path = os.path.join(args.model_dir, "tm_model.pkl")
    with open(sp_path, "wb") as f: pickle.dump(sp, f)
    with open(tm_path, "wb") as f: pickle.dump(tm, f)


    # ---- Save metadata ----
    meta = {
        "input_size": args.input_size,
        "active_bits": args.active_bits,
        "epochs": args.epochs,
        "train_frames": len(train_frames),
        "seed": args.seed,
        "num_columns": args.num_columns,
        "cells_per_column": args.cells_per_column,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "htm_bindings": "unknown"
    }
    meta_path = os.path.join(args.model_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Models saved to: {pathlib.Path(sp_path).resolve().parent}")
    print("   - sp_model.pkl, tm_model.pkl, model_meta.json")


if __name__ == "__main__":
    main()

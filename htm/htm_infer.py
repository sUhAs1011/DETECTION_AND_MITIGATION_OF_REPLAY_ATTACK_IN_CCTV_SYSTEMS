# htm_infer.py
# Load saved HTM, generate scenario-specific streams, run inference (no learning), print anomalies.

from __future__ import annotations
import argparse, json, os
import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from streams.synthetic import SDRStream, splice_loop_replay, splice_one_time_replay, random_sdr

FIRE_THRESH_DEFAULT = 0.5  # for 🔥 marker

def sdrs_from_index_array(idx_array: np.ndarray, input_size: int):
    """idx_array: shape (T, active_bits) of int indices per frame"""
    frames = []
    for row in idx_array:
        s = SDR(input_size)
        s.sparse = np.array(sorted(row.astype(np.int32)))
        frames.append(s)
    return frames

def parse_args():
    ap = argparse.ArgumentParser(description="Load HTM and test scenarios on synthetic streams.")
    ap.add_argument("--model-dir", type=str, default="models")
    ap.add_argument("--scenario", type=str, required=True,
                choices=["normal_continuation", "loop_replay", "one_time_replay", "sudden_anomaly", "normal_seen"])
    ap.add_argument("--length", type=int, default=200, help="total frames to output for the scenario")
    ap.add_argument("--seed", type=int, default=43, help="base seed for scenario generation")
    ap.add_argument("--warmup-sp", type=int, default=0, help="SP warmup steps (learn=True) before scoring")
    ap.add_argument("--fire-thresh", type=float, default=FIRE_THRESH_DEFAULT)
    # Scenario knobs (sane defaults)
    ap.add_argument("--loop-len", type=int, default=15)
    ap.add_argument("--loop-repeats", type=int, default=3)
    ap.add_argument("--insert-len", type=int, default=20)
    ap.add_argument("--sudden-step", type=int, default=50)
    # in parse_args()
    ap.add_argument("--warmup-tm", type=int, default=0, help="TM warmup steps (learn=True) before scoring")
    ap.add_argument("--tm-online", type=int, default=0,
                help="Keep TM learn=True for the first N scored frames, then freeze.")
    return ap.parse_args()

def load_models(model_dir: str):
    import pickle
    with open(os.path.join(model_dir, "model_meta.json"), "r") as f:
        meta = json.load(f)
    meta["model_dir"] = model_dir  # <-- add this
    with open(os.path.join(model_dir, "sp_model.pkl"), "rb") as f:
        sp = pickle.load(f)
    with open(os.path.join(model_dir, "tm_model.pkl"), "rb") as f:
        tm = pickle.load(f)
    assert sp.getColumnDimensions()[0] == meta["num_columns"], "SP columns mismatch metadata"
    return sp, tm, meta


def make_normal_continuation(meta, length: int, seed: int):
    # Ignore the test seed for true continuity; use the training seed & same generator path.
    gen = SDRStream(meta["input_size"], meta["active_bits"], meta["evolution_bits"], seed=meta["seed"])
    full = gen.generate(length=meta["train_length"] + length)  # extend the same path
    return full[-length:]

def make_loop_replay(meta, length: int, seed: int, loop_len: int, repeats: int):
    gen = SDRStream(meta["input_size"], meta["active_bits"], meta["evolution_bits"], seed=meta["seed"])
    base_len = max(meta["train_length"], loop_len + 60)  # a bit more to be safe
    base = gen.generate(length=base_len + length)  # full training + enough continuation

    last_train = base[meta["train_length"] - 1]
    # prefix: continuation frames after training
    prefix_len = max(10, min(30, length // 5))
    prefix = base[meta["train_length"]: meta["train_length"] + prefix_len]

    # loop segment from the training middle
    start_idx = len(base)//3
    loop_segment = base[start_idx:start_idx + loop_len]

    out = splice_loop_replay(prefix, loop_segment, repeats=repeats)

    # extend with more continuation from the same path if needed
    if len(out) < length:
        fill = base[meta["train_length"] + prefix_len : meta["train_length"] + prefix_len + (length - len(out))]
        out.extend(fill)
    else:
        out = out[:length]
    return out

def make_one_time_replay(meta, length: int, seed: int, insert_len: int):
    gen = SDRStream(meta["input_size"], meta["active_bits"], meta["evolution_bits"], seed=meta["seed"])
    base_len = max(meta["train_length"], insert_len + 60)
    base = gen.generate(length=base_len + length)

    # prefix: continuation after training
    pre_len = max(10, min(30, length // 4))
    prefix = base[meta["train_length"] : meta["train_length"] + pre_len]

    # insert from training center
    start_idx = len(base)//2 - insert_len//2
    insert = base[start_idx : start_idx + insert_len]

    # suffix: next continuation frames
    needed = length - (len(prefix) + len(insert))
    suffix = base[meta["train_length"] + pre_len : meta["train_length"] + pre_len + max(0, needed)]

    return prefix + insert + suffix

def make_normal_seen(meta, length: int):
    # Prefer exact saved frames from training tail
    tail_path = os.path.join(meta["model_dir"] if "model_dir" in meta else "models", "train_tail.npy")
    if os.path.exists(tail_path):
        arr = np.load(tail_path)  # shape (T, active_bits)
        if length <= arr.shape[0]:
            arr = arr[-length:]
        frames = sdrs_from_index_array(arr, meta["input_size"])
        return frames
    # Fallback: reconstruct via generator (should match, but saved frames are safer)
    gen = SDRStream(meta["input_size"], meta["active_bits"], meta["evolution_bits"], seed=meta["seed"])
    base = gen.generate(length=meta["train_length"])
    return base[-length:]

def make_sudden_anomaly(meta, length: int, seed: int, sudden_step: int):
    # Normal continuation, but replace one frame with a random SDR at sudden_step (1-indexed in prints)
    seq = make_normal_continuation(meta, length=length, seed=seed)
    sudden_step = max(1, min(length, sudden_step)) - 1  # 0-index
    seq[sudden_step] = random_sdr(meta["input_size"], meta["active_bits"], seed=seed + 999)
    return seq

def run_inference(sp, tm, frames, fire_thresh: float, warmup_sp: int, warmup_tm: int = 0, tm_online: int = 0):
    # Avoid SP warmup with a pretrained TM; it changes the mapping TM learned.
    if warmup_sp > 0:
        print("[warn] SP warmup alters mapping learned by TM; recommend --warmup-sp 0 for pretrained TM.")

    # --- TM warmup (learn=True) on the first few frames; keep context (do NOT reset) ---
    start_idx = 0
    if warmup_tm > 0:
        for sdr in frames[:min(warmup_tm, len(frames))]:
            ac = SDR(sp.getColumnDimensions())
            sp.compute(sdr, learn=False, output=ac)
            tm.compute(ac, learn=True)   # TM adapts to continuation transitions
        # keep temporal context rolling; don't reset
        start_idx = warmup_tm

    print(f"Total frames: {len(frames)}")

    # --- Scoring loop ---
    for i in range(start_idx, len(frames)):
        sdr = frames[i]
        ac = SDR(sp.getColumnDimensions())
        sp.compute(sdr, learn=False, output=ac)

        # One-time debug at first scored step
        if i == start_idx:
            num_active_cols = int(ac.dense.sum())
            print(f"[debug] active columns at first scored step: {num_active_cols}")

        # ONLINE ADAPTATION: allow TM to keep learning for the first 'tm_online' scored frames
        learn_now = (i - start_idx) < tm_online
        tm.compute(ac, learn=learn_now)

        fire = " 🔥" if tm.anomaly >= fire_thresh else ""
        print(f"Step {i+1:02d}: Anomaly = {tm.anomaly:.3f}{fire}")



def main():
    args = parse_args()
    sp, tm, meta = load_models(args.model_dir)

    # Build scenario frames
    if args.scenario == "normal_continuation":
        frames = make_normal_continuation(meta, length=args.length, seed=args.seed)

    elif args.scenario == "loop_replay":
        frames = make_loop_replay(
            meta, length=args.length, seed=args.seed,
            loop_len=args.loop_len, repeats=args.loop_repeats
        )

    elif args.scenario == "one_time_replay":
        frames = make_one_time_replay(
            meta, length=args.length, seed=args.seed,
            insert_len=args.insert_len
        )

    elif args.scenario == "sudden_anomaly":
        frames = make_sudden_anomaly(
            meta, length=args.length, seed=args.seed,
            sudden_step=args.sudden_step
        )
    elif args.scenario == "normal_seen":
        frames = make_normal_seen(meta, length=args.length)
    else:
        raise ValueError("Unknown scenario.")

    # Always start clean per scenario
    tm.reset()
    print(f"\n--- ▶️  Testing: {args.scenario.replace('_',' ').title()} ---")
    run_inference(
    sp, tm, frames,
    fire_thresh=args.fire_thresh,
    warmup_sp=args.warmup_sp,
    warmup_tm=args.warmup_tm,
    tm_online=args.tm_online
)

if __name__ == "__main__":
    main()

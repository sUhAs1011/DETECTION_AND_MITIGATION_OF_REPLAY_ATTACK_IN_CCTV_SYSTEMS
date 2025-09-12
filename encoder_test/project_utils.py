# project_utils.py
"""
Small helper utilities:
 - load_encoded_file(path): loads .npz or .pkl produced by your encoder, returns list-of-frames
 - project_frame_to_indices(frame, ...) : the projector algorithm (per-cell block hashing)
"""

from __future__ import annotations
from typing import List, Sequence, Tuple, Any, Optional
import numpy as np
import pickle, os

def load_encoded_file(path: str) -> Tuple[List[List[np.ndarray]], str]:
    """Load encoded file (.pkl or .npz) and return data as list-of-frames, each frame is list of per-cell numpy int arrays.
       Returns (data, format_hint).
       Accepts:
         - .pkl : pickled list-of-frames (old main_encoder behavior)
         - .npz : expected to contain 'indices' or single array compatible with (T, cells, active) or object array
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # Expect obj: list-of-frames
        return _normalize_loaded_structure(obj), "pkl"
    elif ext == ".npz":
        # try memory-mapped load for large arrays; allow_pickle not used for mmap_mode
        try:
            arr = np.load(path, allow_pickle=True)
            format_hint = "npz(mmap)"
        except Exception:
            # fallback to usual loading (object arrays / pickled content may require allow_pickle)
            arr = np.load(path, allow_pickle=True)
            format_hint = "npz(allow_pickle)"
        # try 'indices' key first
        if 'indices' in arr:
            a = arr['indices']
            return _normalize_loaded_structure(a), f"{format_hint}/indices"
        # fallback: take first array stored
        keys = list(arr.files)
        if len(keys) == 1:
            a = arr[keys[0]]
            return _normalize_loaded_structure(a), f"{format_hint}/{keys[0]}"
        if 'arr_0' in arr:
            return _normalize_loaded_structure(arr['arr_0']), f"{format_hint}/arr_0"
        raise ValueError("Unrecognized .npz structure; expected key 'indices' or single array.")

    else:
        raise ValueError("Unsupported file extension. Expected .pkl or .npz")

def _normalize_loaded_structure(obj) -> List[List[np.ndarray]]:
    """
    Normalize a few possible storage shapes into list-of-frames where each frame is list-of-cells and each cell is 1D int array.
    Possible input shapes:
      - list-of-frames (where frame is list-of-cells and each cell is arraylike)
      - numpy array shape (T, cells, active)  -> convert to list-of-frames
      - numpy object array shape (T,) where each element is list-of-cells
    """
    # If it's a Python list-of-frames already:
    if isinstance(obj, list):
        return [[np.array(cell, dtype=np.int32) for cell in frame] for frame in obj]
    # If it's numpy array
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            # each element could be a list-of-cells
            out = []
            for t in range(obj.shape[0]):
                frame = obj[t]
                out_frame = []
                for cell in frame:
                    out_frame.append(np.array(cell, dtype=np.int32))
                out.append(out_frame)
            return out
        elif obj.ndim == 3:
            # shape (T, cells, active)
            T, C, A = obj.shape
            out = []
            for t in range(T):
                frame = [np.array(obj[t, c, :], dtype=np.int32) for c in range(C)]
                out.append(frame)
            return out
        elif obj.ndim == 2:
            # shape (T, cells) where each element might be an index into something (unlikely).
            # fallback: treat each row as single cell containing those indices
            T, C = obj.shape
            out = []
            for t in range(T):
                frame = []
                for c in range(C):
                    v = obj[t, c]
                    frame.append(np.array(v, dtype=np.int32))
                out.append(frame)
            return out
    raise ValueError("Cannot normalize encoded structure: unsupported type/shape.")

# ----------------- Projection algorithm -----------------

def _mix_uint64_array(idxs: np.ndarray, cell_id: int, seed: int) -> np.ndarray:
    """Vectorized 64-bit integer mixer. Returns uint64 numpy array."""
    # ensure idxs is uint64
    x = idxs.astype(np.uint64)
    # constants (from splitmix64-like mixers)
    x = x * np.uint64(11400714819323198485)  # golden ratio
    x ^= np.uint64(cell_id) * np.uint64(14029467366897019727)
    x ^= np.uint64(seed) * np.uint64(1609587929392839161)
    x ^= (x >> np.uint64(33))
    x = x * np.uint64(0xff51afd7ed558ccd)
    x ^= (x >> np.uint64(33))
    return x

def project_frame_to_indices(frame: Sequence[Sequence[int]],
                             out_size: int,
                             cells: int,
                             per_cell_on: int,
                             block_size: int,
                             seed: int,
                             target_on: int,
                             frame_index: int = 0) -> List[int]:
    """
    Project one frame (list of cell index arrays) -> list of exactly target_on unique indices in [0, out_size).
    frame: sequence of length == cells (or shorter/longer; function will handle).
    per_cell_on: number of picks attempted per cell (before global trim/pad)
    block_size: size of block reserved for each cell
    """
    selected_pos = []  # collect ints
    cells_in_frame = len(frame)
    # 'cells' parameter is the expected number of cells (from encoder / conversion)
    expected_cells = int(cells)

    # iterate only over expected cells; use data when available (frame may have fewer cells)
    for c in range(expected_cells):
        if c < cells_in_frame:
            cell_idxs = np.asarray(frame[c], dtype=np.int64)
            if cell_idxs.size == 0:
                # empty cell: skip for now (we'll pad later)
                continue
            # map each index to block-relative position deterministically
            mixed = _mix_uint64_array(cell_idxs, c, seed)
            pos_in_block = (mixed % np.uint64(block_size)).astype(np.int32)
            global_pos = (c * block_size) + pos_in_block
            # pick the top per_cell_on unique positions by simple counting
            uniq, counts = np.unique(global_pos, return_counts=True)
            # sort by (-count, pos)
            order = np.lexsort((uniq, -counts))
            picked = uniq[order][:per_cell_on]
            selected_pos.extend(int(x) for x in picked)
        else:
            # frame has fewer cells than expected; skip
            continue


    # Now selected_pos may be > target_on (trim) or < target_on (pad)
    selected_pos = list(dict.fromkeys(selected_pos))  # preserve order, drop duplicates
    if len(selected_pos) > target_on:
        # Trim deterministically by sorting by (count heuristic already used), but simplest deterministic trim: sort ascending and keep first target_on
        selected_pos = sorted(selected_pos)[:target_on]
    elif len(selected_pos) < target_on:
        # Pad deterministically: generate fallback hashed sequence from (frame_index, incremental counter)
        pad_needed = target_on - len(selected_pos)
        occupied = set(selected_pos)
        pad_vals = []
        i = 0
        while len(pad_vals) < pad_needed:
            # deterministic source: mix frame_index and i
            seed_val = (frame_index << 16) ^ i ^ seed
            # small deterministic generator:
            v = (_mix_uint64_array(np.array([i], dtype=np.uint64), seed_val & 0xFFFFFFFF, seed) % np.uint64(out_size))[0]
            v = int(v)
            if v not in occupied:
                pad_vals.append(v)
                occupied.add(v)
            i += 1
            if i > out_size * 2:
                # fallback sequential scan
                for cand in range(out_size):
                    if cand not in occupied:
                        pad_vals.append(cand)
                        occupied.add(cand)
                        if len(pad_vals) >= pad_needed:
                            break
                break
        selected_pos.extend(pad_vals)

    # final cleanup: ensure exactly target_on unique positions, sorted
    final = sorted(list(dict.fromkeys(selected_pos))[:target_on])
    # safety pad if something weird happened
    if len(final) < target_on:
        for cand in range(out_size):
            if cand not in final:
                final.append(cand)
            if len(final) >= target_on:
                break
        final = sorted(final)

    return final

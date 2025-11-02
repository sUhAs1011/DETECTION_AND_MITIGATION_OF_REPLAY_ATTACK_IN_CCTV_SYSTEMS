#!/usr/bin/env python3
"""
htm_infer_sample.py

Live anomaly scoring simulation using video file input and pre-trained HTM models.
Simulates real-time streaming without concurrency complexity.

Features:
- Video file input (placeholder for future RTSP stream)
- Sliding window buffer for optical flow computation
- Reuses existing preprocessing, encoding, and projection modules
- HTM inference with pre-trained SP+TM models
- Sequential frame-by-frame anomaly scoring

Usage:
    python htm_infer_sample.py --video path/to/video.mp4 --model-dir htm/models

Future RTSP Integration:
    Replace the video file reading section with RTSP stream capture.
    All other pipeline components remain unchanged.
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional
from datetime import datetime
import atexit

# Add project paths for module imports
project_root = Path(__file__).parent.parent  # Go up from htm/ to project root
sys.path.insert(0, str(project_root / "encoder_test"))
sys.path.insert(0, str(project_root))

# Import existing project modules
from encoder import OpticalFlowEncoder
from project_utils import project_frame_to_indices
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory

import numpy as np

def smooth_and_adapt_threshold(anomaly_scores, alpha=0.3, calibration_window=200, sigma_mult=2.0):
    """
    Smooth raw anomaly scores using exponential moving average and compute adaptive threshold.
    
    Parameters:
    - anomaly_scores: List[float], raw anomaly scores per frame.
    - alpha: float, smoothing factor for EMA.
    - calibration_window: int, number of initial frames to estimate normal baseline.
    - sigma_mult: float, multiplier for threshold (mean + sigma_mult * std).
    
    Returns:
    - smoothed_scores: List[float], EMA smoothed anomaly scores.
    - threshold: float, computed adaptive alert threshold.
    - alerts: List[bool], True if smoothed score exceeds threshold.
    """
    smoothed_scores = []
    for i, score in enumerate(anomaly_scores):
        if i == 0:
            smoothed_scores.append(score)
        else:
            smoothed_val = alpha * score + (1 - alpha) * smoothed_scores[-1]
            smoothed_scores.append(smoothed_val)
    
    # Calculate adaptive threshold from initial frames considered normal
    calibration_scores = smoothed_scores[:calibration_window]
    mean = np.mean(calibration_scores)
    std = np.std(calibration_scores)
    threshold = mean + sigma_mult * std
    
    alerts = [s >= threshold for s in smoothed_scores]
    return smoothed_scores, threshold, alerts



class VideoStreamSimulator:
    """
    Simulates live video stream from file input.
    
    Future RTSP Integration:
    Replace this class with RTSP stream capture logic.
    The interface (get_frame, is_open, release) should remain the same.
    """
    
    def __init__(self, video_path: str, target_width: int = 640, target_height: int = 480):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.target_width = target_width
        self.target_height = target_height
        self.frame_count = 0
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video loaded: {os.path.basename(video_path)}")
        print(f"   Resolution: {target_width}x{target_height}")
        print(f"   FPS: {self.fps:.2f}, Total frames: {self.total_frames}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame, preprocessed and resized."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Preprocess exactly as in training pipeline
        frame = cv2.resize(frame, (self.target_width, self.target_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.frame_count += 1
        return gray
    
    def is_open(self) -> bool:
        """Check if stream is still open."""
        return self.cap.isOpened()
    
    def release(self):
        """Release video capture resources."""
        self.cap.release()
    
    def get_progress(self) -> float:
        """Get processing progress (0.0 to 1.0)."""
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.frame_count / self.total_frames)


class OpticalFlowProcessor:
    """Handles optical flow computation using sliding window buffer."""
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        self.flow_buffer = deque(maxlen=buffer_size - 1)  # One less than frames
        
    def add_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Add frame to buffer and compute optical flow if buffer is full.
        Returns optical flow between last two frames, or None if not enough frames.
        """
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) >= 2:
            # Compute optical flow between last two frames
            prev_frame = self.frame_buffer[-2]
            curr_frame = self.frame_buffer[-1]
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            self.flow_buffer.append(flow)
            return flow
        
        return None


class SDREncoder:
    """Handles SDR encoding and projection using existing modules."""
    
    def __init__(self, grid_rows: int = 5, grid_cols: int = 5, 
                 max_magnitude: float = 15.0, out_size: int = 2048, 
                 target_on: int = 40, hash_seed: int = 42):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cells_per_frame = grid_rows * grid_cols
        
        # Initialize encoder (reuse existing)
        self.encoder = OpticalFlowEncoder(max_magnitude=max_magnitude)
        
        # Projection parameters (match training)
        self.out_size = out_size
        self.target_on = target_on
        self.hash_seed = hash_seed
        self.per_cell_on = int(np.ceil(target_on / max(1, self.cells_per_frame)))
        self.block_size = int(np.ceil(out_size / max(1, self.cells_per_frame)))
        
        # Adjust out_size to fit integer blocks (as in training)
        actual_out_size = self.block_size * max(1, self.cells_per_frame)
        if actual_out_size != out_size:
            print(f"[info] Adjusted out_size {out_size} -> {actual_out_size} to fit integer blocks")
            self.out_size = actual_out_size
    
    def encode_flow_to_sdr(self, flow: np.ndarray) -> SDR:
        """
        Encode optical flow to global SDR using existing pipeline.
        Returns SDR ready for HTM inference.
        """
        H, W = flow.shape[:2]
        cell_h = H // self.grid_rows
        cell_w = W // self.grid_cols
        
        # Extract per-cell SDRs
        cell_sdrs = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < self.grid_rows - 1 else H
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < self.grid_cols - 1 else W
                
                patch = flow[y_start:y_end, x_start:x_end]
                sdr = self.encoder.encode(patch)
                
                # Extract sparse indices
                indices = np.array(sdr.sparse, dtype=np.int32) if getattr(sdr, "sparse", None) is not None else np.array([], dtype=np.int32)
                cell_sdrs.append(indices)
        
        # Project to global SDR (reuse existing logic)
        global_indices = project_frame_to_indices(
            frame=cell_sdrs,
            out_size=self.out_size,
            cells=self.cells_per_frame,
            per_cell_on=self.per_cell_on,
            block_size=self.block_size,
            seed=self.hash_seed,
            target_on=self.target_on,
            frame_index=0  # Not critical for inference
        )
        
        # Create SDR object
        global_sdr = SDR(self.out_size)
        global_sdr.sparse = np.array(sorted(global_indices), dtype=np.int32)
        return global_sdr


class HTMInference:
    """Handles HTM model loading and inference."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.sp = None
        self.tm = None
        self.input_size = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained SP and TM models."""
        import pickle
        
        # Load metadata
        meta_path = os.path.join(self.model_dir, "model_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        self.input_size = meta["input_size"]
        
        # Load models
        sp_path = os.path.join(self.model_dir, "sp_model.pkl")
        tm_path = os.path.join(self.model_dir, "tm_model.pkl")
        
        with open(sp_path, "rb") as f:
            self.sp = pickle.load(f)
        with open(tm_path, "rb") as f:
            self.tm = pickle.load(f)
        
        print(f"🧠 Models loaded from {self.model_dir}")
        print(f"   Input size: {self.input_size}")
        print(f"   SP columns: {self.sp.getColumnDimensions()[0]}")
        print(f"   TM cells per column: {self.tm.getCellsPerColumn()}")
    
    def compute_anomaly(self, sdr: SDR) -> float:
        """
        Compute anomaly score for given SDR.
        Returns anomaly score (0.0 to 1.0).
        """
        # SP inference (no learning)
        active_columns = SDR(self.sp.getColumnDimensions())
        self.sp.compute(sdr, learn=False, output=active_columns)
        
        # TM inference (no learning)
        self.tm.compute(active_columns, learn=False)
        
        return self.tm.anomaly


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser(description="HTM inference simulation on video file")
    ap.add_argument("--video", required=True, help="Input video file path")
    ap.add_argument("--model-dir", default="htm/models", help="Directory containing trained models")
    ap.add_argument("--buffer-size", type=int, default=5, help="Sliding window buffer size")
    ap.add_argument("--grid-rows", type=int, default=5, help="Grid rows for flow encoding")
    ap.add_argument("--grid-cols", type=int, default=5, help="Grid columns for flow encoding")
    ap.add_argument("--max-magnitude", type=float, default=15.0, help="Max flow magnitude for encoding")
    ap.add_argument("--out-size", type=int, default=2048, help="Global SDR size")
    ap.add_argument("--target-on", type=int, default=40, help="Global SDR active bits")
    ap.add_argument("--hash-seed", type=int, default=42, help="Hash seed for projection")
    ap.add_argument("--fire-threshold", type=float, default=0.5, help="Anomaly threshold for alerts")
    ap.add_argument("--postprocess", action="store_true", help="Apply anomaly smoothing and adaptive thresholding after inference")

    return ap.parse_args()


def main():
    """Main inference pipeline."""
    args = parse_args()

        # === AUTOMATIC LOGGING SETUP ===
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file_path = os.path.join(
        logs_dir,
        f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    logfile = open(log_file_path, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except ValueError:
                    # Ignore flush errors if file already closed
                    pass

    sys.stdout = Tee(sys.__stdout__, logfile)
    print(f"[INFO] Logging inference output to {log_file_path}")
    # Register logfile close to safely close after all processing finishes
    atexit.register(logfile.close)
    # === END LOGGING SETUP ===
    
    print("🚀 Starting HTM inference simulation...")
    print(f"   Video: {args.video}")
    print(f"   Models: {args.model_dir}")
    print(f"   Buffer size: {args.buffer_size}")
    
    # Initialize components
    try:
        stream = VideoStreamSimulator(args.video)
        flow_processor = OpticalFlowProcessor(args.buffer_size)
        sdr_encoder = SDREncoder(
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            max_magnitude=args.max_magnitude,
            out_size=args.out_size,
            target_on=args.target_on,
            hash_seed=args.hash_seed
        )
        htm_inference = HTMInference(args.model_dir)
        
        # Verify input size compatibility
        if sdr_encoder.out_size != htm_inference.input_size:
            print(f"⚠️  Warning: SDR size mismatch!")
            print(f"   Encoder output: {sdr_encoder.out_size}")
            print(f"   Model input: {htm_inference.input_size}")
            print("   This may cause inference errors.")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return 1
    
    # Inference loop
    print("\n📊 Starting inference loop...")
    print("Frame | Anomaly | Status")
    print("-" * 30)
    
    frame_idx = 0
    anomaly_scores = []
    
    try:
        while stream.is_open():
            # Get next frame
            frame = stream.get_frame()
            if frame is None:
                break
            
            # Add to buffer and compute flow
            flow = flow_processor.add_frame(frame)
            
            if flow is not None:
                # Encode flow to SDR
                sdr = sdr_encoder.encode_flow_to_sdr(flow)
                
                # Compute anomaly
                anomaly = htm_inference.compute_anomaly(sdr)
                anomaly_scores.append(anomaly)
                
                # Output result
                # status = "🔥 ALERT" if anomaly >= args.fire_threshold else "Normal"
                # progress = stream.get_progress()
                # print(f"{frame_idx:5d} | {anomaly:.4f} | {status} ({progress:.1%})")
                
                # frame_idx += 1
                frame_idx += 1
            
            # Optional: Add small delay to simulate real-time processing
            # time.sleep(1.0 / stream.fps)  # Uncomment for real-time simulation
    
    except KeyboardInterrupt:
        print("\n⏹️  Inference interrupted by user")
    except Exception as e:
        print(f"\n❌ Inference error: {e}")
        return 1
    finally:
        stream.release()

    if args.postprocess:
        smoothed_scores, adaptive_threshold, alerts = smooth_and_adapt_threshold(
            anomaly_scores,
            alpha=0.3,
            calibration_window=200,
            sigma_mult=2.0
        )
        print("\nUsing post-processed anomaly scores with adaptive thresholding:")
        print("Frame | Smoothed Anomaly | Status")
        print("-" * 35)
        for i, (score, alert) in enumerate(zip(smoothed_scores, alerts)):
            status = "🔥 ALERT" if alert else "Normal"
            print(f"{i:5d} | {score:.4f}          | {status}")
    else:
        print("\nUsing raw anomaly scores with fixed thresholding:")
        print("Frame | Anomaly | Status")
        print("-" * 25)
        for i, score in enumerate(anomaly_scores):
            status = "🔥 ALERT" if score >= args.fire_threshold else "Normal"
            print(f"{i:5d} | {score:.4f}  | {status}")

    
    # Summary
    if anomaly_scores:
        if args.postprocess:
            alert_count = sum(1 for a in alerts if a)  # alerts from postprocessing
            avg_anomaly = np.mean(smoothed_scores)
            max_anomaly = np.max(smoothed_scores)
        else:
            alert_count = sum(1 for a in anomaly_scores if a >= args.fire_threshold)
            avg_anomaly = np.mean(anomaly_scores)
            max_anomaly = np.max(anomaly_scores)

    print(f"\n📈 Inference Summary:")
    print(f"   Frames processed: {len(anomaly_scores)}")
    print(f"   Average anomaly: {avg_anomaly:.4f}")
    print(f"   Maximum anomaly: {max_anomaly:.4f}")
    print(f"   Alerts triggered: {alert_count}")
    print(f"   Alert rate: {alert_count/len(anomaly_scores)*100:.1f}%")

    
    print("\n✅ Inference complete")

    return 0
    
if __name__ == "__main__":

    sys.exit(main())
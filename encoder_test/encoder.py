# encoder.py
import numpy as np
from math import atan2, sqrt, sin, cos
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters
from htm.bindings.sdr import SDR

class OpticalFlowEncoder:
    def __init__(self, max_magnitude=15.0):
        self.max_magnitude = max_magnitude

        # sin(theta)
        sin_params = ScalarEncoderParameters()
        sin_params.minimum = -1.0
        sin_params.maximum = 1.0
        sin_params.size = 100
        sin_params.activeBits = 21
        self.sin_encoder = ScalarEncoder(sin_params)
        self.sin_size = self.sin_encoder.size
        self.sin_active = sin_params.activeBits

        # cos(theta)
        cos_params = ScalarEncoderParameters()
        cos_params.minimum = -1.0
        cos_params.maximum = 1.0
        cos_params.size = 100
        cos_params.activeBits = 21
        self.cos_encoder = ScalarEncoder(cos_params)
        self.cos_size = self.cos_encoder.size
        self.cos_active = cos_params.activeBits

        # magnitude
        mag_params = ScalarEncoderParameters()
        mag_params.minimum = 0.0
        mag_params.maximum = self.max_magnitude
        mag_params.size = 100
        mag_params.activeBits = 21
        self.mag_encoder = ScalarEncoder(mag_params)
        self.mag_size = self.mag_encoder.size
        self.mag_active = mag_params.activeBits

        # final size (for convenience)
        self.size = self.sin_size + self.cos_size + self.mag_size
        self.active_bits_expected = self.sin_active + self.cos_active + self.mag_active

    def encode(self, flow_patch):
        """
        flow_patch: H x W x 2 array of dx,dy
        returns: htm.bindings.sdr.SDR object with combined sparse indices
        """
        dx = float(np.median(flow_patch[..., 0]))
        dy = float(np.median(flow_patch[..., 1]))

        magnitude = sqrt(dx**2 + dy**2)
        angle = atan2(dy, dx)

        sin_theta = sin(angle)
        cos_theta = cos(angle)
        magnitude = float(np.clip(magnitude, 0.0, self.max_magnitude))

        sin_sdr = self.sin_encoder.encode(sin_theta)   # SDR relative to [0, sin_size)
        cos_sdr = self.cos_encoder.encode(cos_theta)   # SDR relative to [0, cos_size)
        mag_sdr = self.mag_encoder.encode(magnitude)   # SDR relative to [0, mag_size)

        # extract sparse indices and offset them
        sin_idx = np.asarray(sin_sdr.sparse, dtype=np.int32) if getattr(sin_sdr, "sparse", None) is not None else np.array([], dtype=np.int32)
        cos_idx = np.asarray(cos_sdr.sparse, dtype=np.int32) if getattr(cos_sdr, "sparse", None) is not None else np.array([], dtype=np.int32)
        mag_idx = np.asarray(mag_sdr.sparse, dtype=np.int32) if getattr(mag_sdr, "sparse", None) is not None else np.array([], dtype=np.int32)

        # offset cos and mag indices
        cos_idx_off = cos_idx + self.sin_size
        mag_idx_off = mag_idx + (self.sin_size + self.cos_size)

        # combine
        combined = np.concatenate([sin_idx, cos_idx_off, mag_idx_off]).astype(np.int32)

        # sort (optional, but nice for deterministic ordering)
        if combined.size:
            combined = np.unique(combined)  # union, sorted
        else:
            combined = combined

        out = SDR(self.size)
        out.sparse = combined
        return out

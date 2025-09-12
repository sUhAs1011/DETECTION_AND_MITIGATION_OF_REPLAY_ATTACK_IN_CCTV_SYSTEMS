import numpy as np
from encoder import OpticalFlowEncoder

# Simulate a random 96x96 grid cell flow patch
flow_patch = np.random.uniform(low=-2.0, high=2.0, size=(96, 96, 2))  # dx and dy in range [-2, 2]

# Initialize encoder
encoder = OpticalFlowEncoder(max_magnitude=15.0)

# Encode the patch
sdr = encoder.encode(flow_patch)

# Output the SDR details
print("SDR Encoding Complete")
print(f"SDR shape: {sdr.size}")
print(f"Active bits: {sdr.sparse}")
print(f"Number of active bits: {len(sdr.sparse)}")

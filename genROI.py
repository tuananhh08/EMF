"""6-DoF(x, y, z, roll, pitch, yaw) trong vùng ROI"""

import numpy as np
import csv
import os
import time

# ============================================================
# ROI DEFINITION: 20 cm × 20 cm × 15 cm
# ============================================================
X_MIN, X_MAX = -0.10,     0.10        # mét
Y_MIN, Y_MAX =  0.667882, 0.867882    # mét
Z_MIN, Z_MAX =  0.05879,  0.20879     # mét

ANGLE_MIN, ANGLE_MAX = 0.0, 180.0     # độ

# Số điểm trên mỗi trục
NX,     NY,      NZ      = 15, 15, 10
N_ROLL, N_PITCH, N_YAW  = 10, 10, 10

TOTAL = NX * NY * NZ * N_ROLL * N_PITCH * N_YAW
print(f"Tổng điểm: {NX}x{NY}x{NZ}x{N_ROLL}x{N_PITCH}x{N_YAW} = {TOTAL:,}")

# ============================================================
# AXES
# ============================================================
xv     = np.linspace(X_MIN,    X_MAX,    NX,     dtype=np.float32)
yv     = np.linspace(Y_MIN,    Y_MAX,    NY,     dtype=np.float32)
zv     = np.linspace(Z_MIN,    Z_MAX,    NZ,     dtype=np.float32)
rollv  = np.linspace(ANGLE_MIN, ANGLE_MAX, N_ROLL,  dtype=np.float32)
pitchv = np.linspace(ANGLE_MIN, ANGLE_MAX, N_PITCH, dtype=np.float32)
yawv   = np.linspace(ANGLE_MIN, ANGLE_MAX, N_YAW,   dtype=np.float32)

# cos các góc 
cos_roll  = np.cos(np.deg2rad(rollv)).astype(np.float32)
cos_pitch = np.cos(np.deg2rad(pitchv)).astype(np.float32)
cos_yaw   = np.cos(np.deg2rad(yawv)).astype(np.float32)

# ============================================================
# WRITE CSV THEO CHUNK  
# ============================================================
OUTPUT_FILE = "roi_grid.csv"
HEADER = ["x", "y", "z", "cos_roll", "cos_pitch", "cos_yaw"]

t0 = time.time()
written = 0

with open(OUTPUT_FILE, "w", newline="", buffering=1 << 20) as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)

    for ix, x in enumerate(xv):
        # Mesh 5 truc con lai — chunk ~150,000 dong
        Y, Z, CR, CP, CY = np.meshgrid(
            yv, zv, cos_roll, cos_pitch, cos_yaw,
            indexing="ij"
        )
        n_chunk = Y.size
        block = np.column_stack([
            np.full(n_chunk, x, dtype=np.float32),
            Y.ravel(), Z.ravel(),
            CR.ravel(), CP.ravel(), CY.ravel()
        ])
        writer.writerows(block.tolist())
        del Y, Z, CR, CP, CY, block

        written += n_chunk
        elapsed = time.time() - t0
        pct = written / TOTAL * 100
        eta = elapsed / pct * (100 - pct) if pct > 0 else 0
        print(f"  x={ix+1:2d}/{NX}  {written:>12,}/{TOTAL:,}"
              f"  ({pct:5.1f}%)  {elapsed:.1f}s  ETA={eta:.0f}s", flush=True)

sz = os.path.getsize(OUTPUT_FILE) / 1e6
print(f"\nXong! -> {OUTPUT_FILE}  ({sz:.1f} MB)"
      f" | {written:,} dong | {time.time()-t0:.1f}s")
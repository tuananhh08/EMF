import numpy as np
import pandas as pd
import os
import time

# ============================================================
# THAM SỐ 
# ============================================================
SNR_DB     = 40        # SNR(dB)
CHUNK_SIZE = 50_000    # số dòng đọc mỗi lần 
INPUT_FILE  = "roi_grid.csv"
OUTPUT_FILE = "emf_data.csv"

# ============================================================
# THÔNG SỐ HỆ THỐNG
# ============================================================
MU0 = 4 * np.pi * 1e-7   # H/m

# ── TX coils ─────────────────────────────
# euler_deg = (alpha=roll, beta=pitch, gamma=yaw) của TX
# m_vec = N_tx * I * A_tx * D_l   (I = 1 A)
TX_SPECS = [
    {
        "pos_m"     : np.array([8.50,  0.00,  7.25]) * 1e-2,
        "euler_deg" : (0.0, -15.0,   0.0),
        "turns"     : 300,
        "freq_hz"   : 4000,
        "R_coil_m"  : 0.06,   # bán kính TX (m) 
    },
    {
        "pos_m"     : np.array([-4.25,  7.36,  7.25]) * 1e-2,
        "euler_deg" : (15.0,   0.0,  30.0),
        "turns"     : 300,
        "freq_hz"   : 4500,
        "R_coil_m"  : 0.06,
    },
    {
        "pos_m"     : np.array([-4.25, -7.36,  7.25]) * 1e-2,
        "euler_deg" : (15.0,   0.0, 120.0),
        "turns"     : 300,
        "freq_hz"   : 5000,
        "R_coil_m"  : 0.06,
    },
]

# ── RX coils ──────────────────────
# axis0: trục ban đầu trong capsule frame, xoay theo R(roll,pitch,yaw)
RX_SPECS = [
    {"turns": 250, "area_m2": 3 * np.pi * 1e-6,  "axis0": np.array([1., 0., 0.])},  # 3pi mm²
    {"turns": 200, "area_m2": 10 * 7   * 1e-6,   "axis0": np.array([0., 1., 0.])},  # 70 mm²
    {"turns": 200, "area_m2": 10 * 10  * 1e-6,   "axis0": np.array([0., 0., 1.])},  # 100 mm²
]

# ============================================================
# PRECOMPUTE: dipole moment vector m_l cho từng TX
# ============================================================
def _rot_x(d): r=np.deg2rad(d); c,s=np.cos(r),np.sin(r); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _rot_y(d): r=np.deg2rad(d); c,s=np.cos(r),np.sin(r); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _rot_z(d): r=np.deg2rad(d); c,s=np.cos(r),np.sin(r); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

TX = []
for sp in TX_SPECS:
    al, be, ga = sp["euler_deg"]
    R_tx = _rot_z(ga) @ _rot_y(be) @ _rot_x(al)
    D_l  = R_tx @ np.array([0., 0., 1.])          # hướng moment TX
    A_tx = np.pi * sp["R_coil_m"] ** 2            # diện tích TX (m²)
    I    = 1.0                                     # dòng chuẩn hoá (A)
    M_l  = sp["turns"] * I * A_tx                 # biên độ moment (A·m²)
    TX.append({
        "pos_m" : sp["pos_m"].astype(np.float64),
        "m_vec" : (M_l * D_l).astype(np.float64),
        "omega" : 2 * np.pi * sp["freq_hz"],
    })

RX = RX_SPECS   

# ============================================================
# HÀM TÍNH B  (vectorised trên batch N)
# ============================================================
def compute_B(P_batch, tx):
    """P_batch: (N,3) -> B: (N,3)"""
    Pi = P_batch - tx["pos_m"]                        # (N,3)
    r  = np.linalg.norm(Pi, axis=1, keepdims=True)    # (N,1)
    u  = Pi / r                                        # (N,3)
    m  = tx["m_vec"]                                   # (3,)
    udm = (u @ m).reshape(-1, 1)                       # (N,1)
    return (MU0 / (4 * np.pi)) / r**3 * (3 * udm * u - m)

# ============================================================
# HÀM XÂY DỰNG R_batch  (ZYX: Rz*Ry*Rx)
# Input: cos_roll, cos_pitch, cos_yaw  mỗi cái (N,)
# Output: R (N,3,3)
# ============================================================
def build_R_batch(cr_arr, cp_arr, cy_arr):
    roll  = np.arccos(np.clip(cr_arr, -1, 1))
    pitch = np.arccos(np.clip(cp_arr, -1, 1))
    yaw   = np.arccos(np.clip(cy_arr, -1, 1))

    sr, cr = np.sin(roll),  np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw),   np.cos(yaw)

    N = len(roll)
    R = np.zeros((N, 3, 3), dtype=np.float64)

    # Rz(yaw) * Ry(pitch) * Rx(roll)
    R[:, 0, 0] =  cy*cp
    R[:, 0, 1] =  cy*sp*sr - sy*cr
    R[:, 0, 2] =  cy*sp*cr + sy*sr
    R[:, 1, 0] =  sy*cp
    R[:, 1, 1] =  sy*sp*sr + cy*cr
    R[:, 1, 2] =  sy*sp*cr - cy*sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] =  cp*sr
    R[:, 2, 2] =  cp*cr
    return R

# ============================================================
# MAIN
# ============================================================
emf_cols = [f"EMF_TX{l+1}_RX{k+1}" for l in range(3) for k in range(3)]
header_out = ["x","y","z","cos_roll","cos_pitch","cos_yaw"] + emf_cols

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print(f"SNR   : {SNR_DB} dB  |  chunk = {CHUNK_SIZE:,} dong")
print(f"Columns EMF: {emf_cols}\n")

t0 = time.time()
total_written = 0

with open(OUTPUT_FILE, "w") as fout:
    # Không ghi header

    for chunk_idx, df in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE)):
        N = len(df)
        P   = df[["x","y","z"]].values.astype(np.float64)
        cr  = df["cos_roll"].values.astype(np.float64)
        cp  = df["cos_pitch"].values.astype(np.float64)
        cy  = df["cos_yaw"].values.astype(np.float64)

        R_batch  = build_R_batch(cr, cp, cy)   # (N,3,3)
        EMF_mat  = np.zeros((N, 9), dtype=np.float64)

        col = 0
        for l, tx in enumerate(TX):
            B = compute_B(P, tx)               # (N,3)
            for k, rx in enumerate(RX):
                # Trục RX sau khi xoay theo capsule orientation
                Rxhat   = R_batch @ rx["axis0"]                       # (N,3)
                dot_val = np.einsum("ni,ni->n", B, Rxhat)             # (N,)
                EMF_mat[:, col] = rx["turns"] * tx["omega"] * rx["area_m2"] * dot_val
                col += 1

        # ── Add nhiễu Gauss (SNR tính theo từng sample) ──────
        sig_pwr  = np.mean(EMF_mat ** 2, axis=1, keepdims=True)      # (N,1)
        nse_pwr  = sig_pwr / (10 ** (SNR_DB / 10))
        EMF_noisy = EMF_mat + np.sqrt(nse_pwr) * np.random.randn(N, 9)

        # ── Ghi ──────────────────────────────────────────────
        pd.DataFrame(EMF_noisy).to_csv(
            fout, index=False, header=False, float_format="%.8g"
        )

        total_written += N
        print(f"  chunk {chunk_idx+1:4d}  |  {total_written:>12,} dong"
              f"  |  {time.time()-t0:.1f}s", flush=True)

sz = os.path.getsize(OUTPUT_FILE) / 1e6
print(f"\nXong! -> {OUTPUT_FILE}  ({sz:.1f} MB)"
      f" | {total_written:,} dong | {time.time()-t0:.1f}s")
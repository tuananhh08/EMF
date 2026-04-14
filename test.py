"""
Chạy:
  python test.py \
      --emf        /path/to/emf_test.csv       \
      --label      /path/to/roi_grid_test.csv   \
      --ckpt_dir   /path/to/checkpoints         \
      --out        /path/to/output.png          \
      --seed       42
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  # noqa: F401 (unused after refactor)

from model import predict


# ============================================================
# ARGPARSE
# ============================================================

def get_args():
    p = argparse.ArgumentParser(description="Test 6-DoF Decision Tree Model")
    p.add_argument("--emf",      type=str, required=True,
                   help="EMF test CSV (9 cols, no header)")
    p.add_argument("--label",    type=str, required=True,
                   help="Label test CSV (6 cols, with header)")
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Thư mục chứa dt_model.pkl (hoặc dt_model_scenario2.pkl)")
    p.add_argument("--out",      type=str, default=None,
                   help="Đường dẫn file PNG output (mặc định: ckpt_dir/test_result.png)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(ckpt_dir: str):

    candidates = ["dt_model.pkl", "dt_model_scenario2.pkl"]
    for name in candidates:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                ckpt = pickle.load(f)
            print(f"[Checkpoint] Loaded ← {path}")
            print(f"  backend    : {ckpt['backend']}")
            print(f"  best_depth : {ckpt.get('best_depth', 'N/A')}")
            return ckpt["model"], ckpt["backend"], ckpt
    raise FileNotFoundError(
        f"Không tìm thấy checkpoint trong {ckpt_dir}. "
        f"Cần: {candidates[0]} hoặc {candidates[1]}"
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data(emf_path, label_path, chunk=200_000):
    print(f"\n[Data] EMF  : {emf_path}")
    X = np.vstack([c.values.astype(np.float32)
                   for c in pd.read_csv(emf_path, header=None, chunksize=chunk)])
    print(f"[Data] Label: {label_path}")
    y = np.vstack([c.values.astype(np.float32)
                   for c in pd.read_csv(label_path, chunksize=chunk)])
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 9 and y.shape[1] == 6
    print(f"[Data] X={X.shape}  y={y.shape}\n")
    return X, y


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:

    # --- Position (m → mm) ---
    diff_pos   = (y_true[:, :3] - y_pred[:, :3]) * 1000       
    pos_err    = np.linalg.norm(diff_pos, axis=1)             

    # --- Orientation in cos-space (direct regression target) ---
    diff_cos = y_true[:, 3:] - y_pred[:, 3:]
    cos_mae = np.mean(np.abs(diff_cos), axis=0)  # (3,)
    cos_rmse = np.sqrt(np.mean(diff_cos ** 2, axis=0))  # (3,)
    cos_mae_mean = float(np.mean(cos_mae))
    cos_rmse_mean = float(np.mean(cos_rmse))

    # --- Orientation: cos → angle (rad) → deg ---
    # NOTE: This maps cos(theta) to theta in [0, 180] degrees.
    ang_true   = np.degrees(np.arccos(np.clip(y_true[:, 3:], -1, 1)))  # (N,3)
    ang_pred   = np.degrees(np.arccos(np.clip(y_pred[:, 3:], -1, 1)))  # (N,3)
    diff_ang   = ang_true - ang_pred                                     # (N,3)
    ori_err    = np.linalg.norm(diff_ang, axis=1)                       # (N,)  deg

    return {
        "pos_error_per_sample" : pos_err,
        "ori_error_per_sample" : ori_err,
        "mean_pos_error_mm"    : float(pos_err.mean()),
        "mean_ori_error_deg"   : float(ori_err.mean()),
        "rmse_pos_mm"          : float(np.sqrt(np.mean(pos_err ** 2))),
        "rmse_ori_deg"         : float(np.sqrt(np.mean(ori_err ** 2))),
        "cos_mae_per_axis"      : cos_mae.astype(float),
        "cos_rmse_per_axis"     : cos_rmse.astype(float),
        "cos_mae_mean"          : cos_mae_mean,
        "cos_rmse_mean"         : cos_rmse_mean,
    }


# ============================================================
# PLOT
# ============================================================

def _base_dir(out_path: str) -> tuple:
    """Trả về (thư mục, stem) từ out_path. VD: '/a/b/result.png' → ('/a/b', 'result')"""
    d    = os.path.dirname(out_path) or "."
    stem = os.path.splitext(os.path.basename(out_path))[0]
    return d, stem


def make_figures(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 metrics: dict,
                 out_path: str):
    """
    Lưu 3 file PNG riêng biệt:
      <stem>_fig1_scatter.png
      <stem>_fig2_pos_error.png
      <stem>_fig3_ori_error.png
    """
    d, stem = _base_dir(out_path)
    N          = len(y_true)
    sample_idx = np.arange(N)

    pos_true_mm = y_true[:, :3] * 1000
    pos_pred_mm = y_pred[:, :3] * 1000
    pos_err     = metrics["pos_error_per_sample"]
    ori_err     = metrics["ori_error_per_sample"]

    # ── Fig 1: 3D scatter GT vs Pred ─────────────────────────────
    fig1 = plt.figure(figsize=(9, 7))
    ax1  = fig1.add_subplot(111, projection="3d")

    ax1.scatter(pos_true_mm[:, 0], pos_true_mm[:, 1], pos_true_mm[:, 2],
                s=8, alpha=0.55, color="#2563EB", label="Ground Truth")
    ax1.scatter(pos_pred_mm[:, 0], pos_pred_mm[:, 1], pos_pred_mm[:, 2],
                s=8, alpha=0.55, color="#DC2626", marker="^", label="Predicted")

    ax1.set_xlabel("X (mm)", fontsize=9)
    ax1.set_ylabel("Y (mm)", fontsize=9)
    ax1.set_zlabel("Z (mm)", fontsize=9)
    ax1.set_title("Test Set Inference", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(fontsize=9, loc="upper left")

    metrics_str = (
        f"Mean Position Error    : {metrics['mean_pos_error_mm']:.4f} mm\n"
        f"Mean Orientation Error : {metrics['mean_ori_error_deg']:.4f}°\n"
        f"RMSE Position          : {metrics['rmse_pos_mm']:.4f} mm\n"
        f"RMSE Orientation       : {metrics['rmse_ori_deg']:.4f}°\n"
        f"Cos-MAE (mean/axis)    : {metrics['cos_mae_mean']:.6f}\n"
        f"Cos-RMSE (mean/axis)   : {metrics['cos_rmse_mean']:.6f}\n"
        f"N samples              : {N:,}"
    )
    ax1.text2D(
        0.02, 0.02, metrics_str,
        transform=ax1.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#94A3B8", alpha=0.88)
    )

    p1 = os.path.join(d, f"{stem}_fig1_scatter.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Plot] Fig1 → {p1}")

    # ── Fig 2: Position error per sample ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 4))

    ax2.plot(sample_idx, pos_err,
             linewidth=0.9, color="#2563EB", alpha=0.8, label="Position Error")
    ax2.axhline(metrics["mean_pos_error_mm"], color="#F59E0B",
                linewidth=1.5, linestyle="--",
                label=f"Mean = {metrics['mean_pos_error_mm']:.4f} mm")
    ax2.axhline(metrics["rmse_pos_mm"], color="#DC2626",
                linewidth=1.5, linestyle=":",
                label=f"RMSE = {metrics['rmse_pos_mm']:.4f} mm")

    ax2.set_xlabel("Sample Index", fontsize=10)
    ax2.set_ylabel("Position Error (mm)", fontsize=10)
    ax2.set_title("Position Error per Sample", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.35)
    plt.tight_layout()

    p2 = os.path.join(d, f"{stem}_fig2_pos_error.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Fig2 → {p2}")

    # ── Fig 3: Orientation error per sample ──────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 4))

    ax3.plot(sample_idx, ori_err,
             linewidth=0.9, color="#7C3AED", alpha=0.8, label="Orientation Error")
    ax3.axhline(metrics["mean_ori_error_deg"], color="#F59E0B",
                linewidth=1.5, linestyle="--",
                label=f"Mean = {metrics['mean_ori_error_deg']:.4f}°")
    ax3.axhline(metrics["rmse_ori_deg"], color="#DC2626",
                linewidth=1.5, linestyle=":",
                label=f"RMSE = {metrics['rmse_ori_deg']:.4f}°")

    ax3.set_xlabel("Sample Index", fontsize=10)
    ax3.set_ylabel("Orientation Error (°)", fontsize=10)
    ax3.set_title("Orientation Error per Sample", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.35)
    plt.tight_layout()

    p3 = os.path.join(d, f"{stem}_fig3_ori_error.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"[Plot] Fig3 → {p3}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = get_args()
    np.random.seed(args.seed)

    out_path = args.out if args.out else \
               os.path.join(args.ckpt_dir, "test_result.png")

    # Load model
    model, backend, ckpt_info = load_checkpoint(args.ckpt_dir)

    # Load data
    X, y_true = load_data(args.emf, args.label)
    preproc = ckpt_info.get("preproc", None)
    if preproc is not None:
        X = preproc.transform(X)
        print("[Preprocess] Applied preprocessor from checkpoint\n")

    # Inference
    print("[Infer] Running inference...")
    import time
    t0     = time.time()
    y_pred = predict(model, backend, X).astype(np.float64)
    inf_ms = (time.time() - t0) / len(X) * 1000
    print(f"[Infer] Done  {inf_ms:.4f} ms/sample\n")

    y_true = y_true.astype(np.float64)

    # Metrics
    metrics = compute_metrics(y_true, y_pred)

    print(f"{'='*48}")
    print(f"  Test Results")
    print(f"{'='*48}")
    print(f"  Mean Position Error   : {metrics['mean_pos_error_mm']:.4f} mm")
    print(f"  Mean Orientation Error: {metrics['mean_ori_error_deg']:.4f} deg")
    print(f"  RMSE Position         : {metrics['rmse_pos_mm']:.4f} mm")
    print(f"  RMSE Orientation      : {metrics['rmse_ori_deg']:.4f} deg")
    print(f"  Cos MAE (mean)        : {metrics['cos_mae_mean']:.6f}")
    print(f"  Cos RMSE (mean)       : {metrics['cos_rmse_mean']:.6f}")
    print(f"  Inference time        : {inf_ms:.4f} ms/sample")
    print(f"{'='*48}\n")

    # Plot
    make_figures(y_true, y_pred, metrics, out_path)


if __name__ == "__main__":
    main()
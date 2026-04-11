#train, test tren 2 tap khac nhau
import argparse
import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import build_model, fit, predict
from loss  import HuberPoseLoss


# ============================================================
# ARGPARSE
# ============================================================

def get_args():
    p = argparse.ArgumentParser(
        description="Train2: train/val cùng tập, test tập ngoài")
    p.add_argument("--emf",        type=str,   required=True,
                   help="EMF data CSV dùng để train+val (9 cols, no header)")
    p.add_argument("--label",      type=str,   required=True,
                   help="Label CSV dùng để train+val (6 cols, with header)")
    p.add_argument("--emf_test",   type=str,   required=True,
                   help="EMF data CSV của tập test ngoài")
    p.add_argument("--label_test", type=str,   required=True,
                   help="Label CSV của tập test ngoài")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    p.add_argument("--max_depth",  type=int,   default=30)
    p.add_argument("--val_ratio",  type=float, default=0.20,
                   help="Tỉ lệ val tách từ tập train+val (mặc định 0.20)")
    p.add_argument("--ang_weight", type=float, default=1.0)
    p.add_argument("--delta_xyz",  type=float, default=0.055)
    p.add_argument("--delta_ang",  type=float, default=0.16)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ============================================================
# DATA
# ============================================================

def load_data(emf_path, label_path, chunk=200_000, tag=""):
    print(f"\n[Data{tag}] EMF  : {emf_path}")
    X = np.vstack([c.values.astype(np.float32)
                   for c in pd.read_csv(emf_path, header=None, chunksize=chunk)])
    print(f"[Data{tag}] Label: {label_path}")
    y = np.vstack([c.values.astype(np.float32)
                   for c in pd.read_csv(label_path, chunksize=chunk)])
    assert X.shape[0] == y.shape[0], \
        f"Số dòng không khớp: X={X.shape[0]}, y={y.shape[0]}"
    assert X.shape[1] == 9 and y.shape[1] == 6
    print(f"[Data{tag}] X={X.shape}  y={y.shape}")
    return X, y


def split_trainval(X, y, val_ratio, seed):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed, shuffle=True)
    print(f"\n[Split] train={len(X_tr):,}  val={len(X_val):,}")
    return X_tr, X_val, y_tr, y_val


# ============================================================
# LEARNING CURVE  
# ============================================================

def run_learning_curve(X_tr, y_tr, X_val, y_val,
                       criterion, max_depth, seed):
    depths, train_losses, val_losses = [], [], []

    print(f"\n[Curve] Learning curve  depth 1 → {max_depth}")
    print(f"{'Depth':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Time':>7}")
    print("-" * 42)

    for d in range(1, max_depth + 1):
        t0 = time.time()
        m, backend = build_model(max_depth=d, random_state=seed)
        m = fit(m, backend, X_tr, y_tr)

        tl, _, _ = criterion(
            predict(m, backend, X_tr).astype(np.float64),
            y_tr.astype(np.float64))
        vl, _, _ = criterion(
            predict(m, backend, X_val).astype(np.float64),
            y_val.astype(np.float64))

        depths.append(d)
        train_losses.append(tl)
        val_losses.append(vl)

        print(f"{d:>6}  {tl:>12.6f}  {vl:>10.6f}  {time.time()-t0:>6.1f}s")

    return depths, train_losses, val_losses


# ============================================================
# PLOT
# ============================================================

def plot_loss_curve(depths, train_losses, val_losses,
                    save_path: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(depths, train_losses, marker="o", markersize=4,
            linewidth=1.8, label="Train Loss", color="#2563EB")
    ax.plot(depths, val_losses,   marker="s", markersize=4,
            linewidth=1.8, label="Val Loss",   color="#DC2626", linestyle="--")

    best_d = depths[int(np.argmin(val_losses))]
    best_v = min(val_losses)
    ax.axvline(best_d, color="#16A34A", linestyle=":", linewidth=1.4,
               label=f"Best depth={best_d}  val={best_v:.5f}")

    ax.set_xlabel("max_depth", fontsize=12)
    ax.set_ylabel("Huber Loss", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Load train+val
    X, y = load_data(args.emf, args.label, tag=" train+val")
    X_tr, X_val, y_tr, y_val = split_trainval(X, y, args.val_ratio, args.seed)

    # Load test ngoài
    X_te, y_te = load_data(args.emf_test, args.label_test, tag=" test")

    # Loss
    criterion = HuberPoseLoss(ang_weight=args.ang_weight,
                               delta_xyz=args.delta_xyz,
                               delta_ang=args.delta_ang)
    print(f"\n[Loss] HuberPoseLoss  ang_weight={args.ang_weight}"
          f"  delta_xyz={args.delta_xyz}  delta_ang={args.delta_ang}")

    # Learning curve
    depths, train_losses, val_losses = run_learning_curve(
        X_tr, y_tr, X_val, y_val, criterion, args.max_depth, args.seed)

    # Plot
    plot_path = os.path.join(args.ckpt_dir, "learning_curve_scenario2.png")
    plot_loss_curve(depths, train_losses, val_losses,
                    save_path=plot_path,
                    title="Decision Tree — Huber Loss vs max_depth")

    # Fit final model ở best depth
    best_depth = depths[int(np.argmin(val_losses))]
    print(f"\n[Final] Fit model với best depth={best_depth}")
    t0 = time.time()
    model, backend = build_model(max_depth=best_depth, random_state=args.seed)
    model = fit(model, backend, X_tr, y_tr)
    print(f"[Final] Xong ({time.time()-t0:.1f}s)")

    # Evaluate
    tl_f, lx_f, la_f = criterion(
        predict(model, backend, X_tr).astype(np.float64),  y_tr.astype(np.float64))
    vl_f, vx_f, va_f = criterion(
        predict(model, backend, X_val).astype(np.float64), y_val.astype(np.float64))
    te_l, te_x, te_a = criterion(
        predict(model, backend, X_te).astype(np.float64),  y_te.astype(np.float64))

    # Inference time
    t_inf  = time.time()
    _      = predict(model, backend, X_te)
    inf_ms = (time.time() - t_inf) / len(X_te) * 1000

    # Save checkpoint
    ckpt = os.path.join(args.ckpt_dir, "dt_model_scenario2.pkl")
    with open(ckpt, "wb") as f:
        pickle.dump({"model": model, "backend": backend,
                     "best_depth": best_depth, "args": vars(args)}, f)

    # Summary
    print(f"\n{'='*52}")
    print(f"  Summary — Scenario 2")
    print(f"{'='*52}")
    print(f"  Backend       : {backend}")
    print(f"  Best depth    : {best_depth}")
    print(f"  Train Loss    : {tl_f:.6f}  (xyz={lx_f:.6f}  ang={la_f:.6f})")
    print(f"  Val   Loss    : {vl_f:.6f}  (xyz={vx_f:.6f}  ang={va_f:.6f})")
    print(f"  Test  Loss    : {te_l:.6f}  (xyz={te_x:.6f}  ang={te_a:.6f})")
    print(f"  Infer time    : {inf_ms:.4f} ms/sample")
    print(f"  Checkpoint    : {ckpt}")
    print(f"  Plot          : {plot_path}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
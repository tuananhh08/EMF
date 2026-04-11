import argparse
import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import build_model, fit, predict
from loss  import HuberPoseLoss


# ============================================================
# ARGPARSE
# ============================================================

def get_args():
    p = argparse.ArgumentParser(description="Train Decision Tree 6-DoF Localization")
    p.add_argument("--emf",        type=str, default="emf_data.csv",
                   help="Path to EMF data CSV (9 cols, no header)")
    p.add_argument("--label",      type=str, default="roi_grid.csv",
                   help="Path to pose label CSV (6 cols, with header)")
    p.add_argument("--ckpt_dir",   type=str, default="checkpoints",
                   help="Thư mục lưu model checkpoint")
    p.add_argument("--max_depth",  type=int,   default=30)
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--ang_weight", type=float, default=1.0,
                   help="Trọng số loss_ang trong HuberPoseLoss")
    p.add_argument("--delta_xyz",  type=float, default=0.055,
                   help="Ngưỡng Huber cho xyz")
    p.add_argument("--delta_ang",  type=float, default=0.16,
                   help="Ngưỡng Huber cho cos angles")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(emf_path: str, label_path: str, chunk: int = 200_000):
    print(f"\n[Data] Đọc EMF  : {emf_path}")
    X = np.vstack([
        c.values.astype(np.float32)
        for c in pd.read_csv(emf_path, header=None, chunksize=chunk)
    ])

    print(f"[Data] Đọc Label: {label_path}")
    y = np.vstack([
        c.values.astype(np.float32)
        for c in pd.read_csv(label_path, chunksize=chunk)
    ])

    assert X.shape[0] == y.shape[0], \
        f"Số dòng không khớp: X={X.shape[0]}, y={y.shape[0]}"
    assert X.shape[1] == 9, f"EMF phải có 9 cột, hiện có {X.shape[1]}"
    assert y.shape[1] == 6, f"Label phải có 6 cột, hiện có {y.shape[1]}"

    print(f"[Data] X={X.shape}  y={y.shape}\n")
    return X, y


# ============================================================
# SPLIT
# ============================================================

def split_data(X, y, val_ratio, test_ratio, seed):
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_frac,
        random_state=seed
    )
    print(f"[Split] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}\n")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# EVALUATE (in loss)
# ============================================================

def evaluate(model, backend, X, y, criterion, split_name: str):
    y_pred = predict(model, backend, X)
    total, loss_xyz, loss_ang = criterion(
        y_pred.astype(np.float64),
        y.astype(np.float64)
    )
    print(f"[{split_name:5s}] "
          f"Loss={total:.6f}  "
          f"loss_xyz={loss_xyz:.6f}  "
          f"loss_ang={loss_ang:.6f}")
    return total


# ============================================================
# MAIN
# ============================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # --- Load ---
    X, y = load_data(args.emf, args.label)

    # --- Split ---
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, args.val_ratio, args.test_ratio, args.seed
    )

    # --- Loss ---
    criterion = HuberPoseLoss(
        ang_weight=args.ang_weight,
        delta_xyz=args.delta_xyz,
        delta_ang=args.delta_ang,
    )
    print(f"[Loss] HuberPoseLoss  ang_weight={args.ang_weight}"
          f"  delta_xyz={args.delta_xyz}  delta_ang={args.delta_ang}\n")

    # --- Build model ---
    model, backend = build_model(
        max_depth=args.max_depth,
        random_state=args.seed,
    )

    # --- Train ---
    print(f"\n[Train] Bắt đầu training...")
    t0 = time.time()
    model = fit(model, backend, X_train, y_train)
    train_time = time.time() - t0
    print(f"[Train] Xong  ({train_time:.1f}s)\n")

    # --- Evaluate ---
    evaluate(model, backend, X_train, y_train, criterion, "Train")
    evaluate(model, backend, X_val,   y_val,   criterion, "Val")
    evaluate(model, backend, X_test,  y_test,  criterion, "Test")

    # --- Inference time ---
    t_inf    = time.time()
    _        = predict(model, backend, X_test)
    inf_ms   = (time.time() - t_inf) / len(X_test) * 1000
    print(f"\n[Infer] {inf_ms:.4f} ms/sample")

    # --- Save checkpoint ---
    ckpt_path = os.path.join(args.ckpt_dir, "dt_model.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump({"model": model, "backend": backend, "args": vars(args)}, f)
    sz = os.path.getsize(ckpt_path) / 1e6
    print(f"\n[Save] Checkpoint → {ckpt_path}  ({sz:.1f} MB)")

    # --- Summary ---
    print(f"\n{'='*45}")
    print(f"  Training Summary")
    print(f"{'='*45}")
    print(f"  Backend     : {backend}")
    print(f"  max_depth   : {args.max_depth}")
    print(f"  Train time  : {train_time:.1f}s")
    print(f"  Infer time  : {inf_ms:.4f} ms/sample")
    print(f"  Checkpoint  : {ckpt_path}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
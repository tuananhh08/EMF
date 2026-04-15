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
from preprocess import EMFPreprocessor


# ============================================================
# ARGPARSE
# ============================================================

def get_args():
    p = argparse.ArgumentParser(description="Decision Tree 6-DoF Localization")
    p.add_argument("--emf",        type=str,   default="emf_data.csv")
    p.add_argument("--label",      type=str,   default="roi_grid.csv")
    p.add_argument("--ckpt_dir",   type=str,   default="checkpoints")
    p.add_argument("--max_depth",  type=int,   default=30)
    p.add_argument("--min_leaf_list", type=str, default="1,5,20")
    p.add_argument("--min_split", type=int, default=2)
    p.add_argument("--max_features_list", type=str, default="None,sqrt,log2")
    p.add_argument("--splitter", type=str, default="best",
                   choices=["best", "random"])
    p.add_argument("--separate_heads", action="store_true")
    p.add_argument("--split_mode", type=str, default="random",
                   choices=["random", "interior_block"])
    p.add_argument("--block_frac", type=float, default=0.20)
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--ang_weight", type=float, default=1.0)
    p.add_argument("--delta_xyz",  type=float, default=0.055)
    p.add_argument("--delta_ang",  type=float, default=0.16)
    p.add_argument("--use_signed_log", action="store_true")
    p.add_argument("--no_standardize", action="store_true")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ============================================================
# DATA
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


def split_data(X, y, val_ratio, test_ratio, seed):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=val_ratio+test_ratio, random_state=seed, shuffle=True)
    vf = val_ratio / (val_ratio + test_ratio)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=1-vf, random_state=seed)
    print(f"[Split] train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}\n")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ============================================================
# RESUME LEARNING CURVE
# ============================================================

def run_learning_curve(X_tr, y_tr, X_val, y_val,
                       criterion, max_depth, seed,
                       min_leaf_list, min_split, max_features_list,
                       splitter, separate_heads, ckpt_dir):

    progress_path = os.path.join(ckpt_dir, "curve_progress.pkl")

    # LOAD PROGRESS
    if os.path.exists(progress_path):
        print(f"\n[Resume] Load {progress_path}")
        with open(progress_path, "rb") as f:
            prog = pickle.load(f)

        depths       = prog["depths"]
        train_losses = prog["train_losses"]
        val_losses   = prog["val_losses"]
        best_overall = prog["best_overall"]

        start_depth = depths[-1] + 1
        print(f"[Resume] Continue from depth = {start_depth}\n")
    else:
        depths, train_losses, val_losses = [], [], []
        best_overall = {"val_loss": None, "depth": None, "min_leaf": None, "max_features": None}
        start_depth = 1

    for d in range(start_depth, max_depth + 1):
        best_row = None
        best_vl = None

        for leaf in min_leaf_list:
            for mf in max_features_list:
                t0 = time.time()

                m, backend = build_model(
                    max_depth=d,
                    random_state=seed,
                    min_samples_leaf=leaf,
                    min_samples_split=min_split,
                    max_features=mf,
                    splitter=splitter,
                    separate_heads=separate_heads,
                )
                m = fit(m, backend, X_tr, y_tr)

                pred_tr  = predict(m, backend, X_tr)
                pred_val = predict(m, backend, X_val)

                tl, _, _ = criterion(pred_tr, y_tr)
                vl, _, _ = criterion(pred_val, y_val)

                if best_vl is None or vl < best_vl:
                    best_vl = vl
                    best_row = (leaf, mf, tl, vl)

                if best_overall["val_loss"] is None or vl < best_overall["val_loss"]:
                    best_overall = {
                        "val_loss": vl,
                        "depth": d,
                        "min_leaf": leaf,
                        "max_features": mf
                    }

                print(f"[d={d}] leaf={leaf} mf={mf}  train={tl:.6f} val={vl:.6f}")

        leaf, mf, tl, vl = best_row
        depths.append(d)
        train_losses.append(tl)
        val_losses.append(vl)

        # SAVE PROGRESS
        with open(progress_path, "wb") as f:
            pickle.dump({
                "depths": depths,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_overall": best_overall
            }, f)

        print(f"[Save] depth {d}")

    return depths, train_losses, val_losses, best_overall


# ============================================================
# MAIN
# ============================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    X, y = load_data(args.emf, args.label)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(
        X, y, args.val_ratio, args.test_ratio, args.seed)

    preproc = EMFPreprocessor(
        use_signed_log=args.use_signed_log,
        use_standardize=(not args.no_standardize),
    )
    X_tr  = preproc.fit_transform(X_tr)
    X_val = preproc.transform(X_val)
    X_te  = preproc.transform(X_te)

    criterion = HuberPoseLoss(
        ang_weight=args.ang_weight,
        delta_xyz=args.delta_xyz,
        delta_ang=args.delta_ang
    )

    min_leaf_list = [int(x) for x in args.min_leaf_list.split(",")]
    max_features_list = [None if x=="None" else x for x in args.max_features_list.split(",")]

    depths, train_losses, val_losses, best_overall = run_learning_curve(
        X_tr, y_tr, X_val, y_val,
        criterion, args.max_depth, args.seed,
        min_leaf_list, args.min_split, max_features_list,
        args.splitter, args.separate_heads,
        args.ckpt_dir
    )

    print("\n[Best]", best_overall)


if __name__ == "__main__":
    main()
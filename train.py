#train/valid/test tren cung 1 tap data
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
    p.add_argument("--min_leaf_list", type=str, default="1,5,20",
                   help="Comma-separated list for min_samples_leaf sweep")
    p.add_argument("--min_split", type=int, default=2,
                   help="min_samples_split for DecisionTree")
    p.add_argument("--max_features_list", type=str, default="None,sqrt,log2",
                   help="Comma-separated list: None,sqrt,log2 or float in (0,1]")
    p.add_argument("--splitter", type=str, default="best",
                   choices=["best", "random"])
    p.add_argument("--separate_heads", action="store_true",
                   help="Train 2 trees: xyz-head and orientation-head")
    p.add_argument("--split_mode", type=str, default="random",
                   choices=["random", "interior_block"],
                   help="Split strategy: random IID or hold out an interior xyz block")
    p.add_argument("--block_frac", type=float, default=0.20,
                   help="Interior block fraction per axis (0<frac<1), only for split_mode=interior_block")
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--ang_weight", type=float, default=1.0)
    p.add_argument("--delta_xyz",  type=float, default=0.055)
    p.add_argument("--delta_ang",  type=float, default=0.16)
    p.add_argument("--use_signed_log", action="store_true",
                   help="Apply signed-log compression to EMF features before training")
    p.add_argument("--no_standardize", action="store_true",
                   help="Disable EMF standardization (default: enabled)")
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


def split_data_interior_block(X, y, val_ratio, block_frac, seed):
    """
    Hold out a contiguous interior block in xyz-space as test set.
    This stays inside ROI (interpolation) but avoids neighbor leakage of random split.
    """
    if not (0.0 < block_frac < 1.0):
        raise ValueError("block_frac must be in (0,1)")

    pos = y[:, :3]
    lo_q = 0.5 - block_frac / 2.0
    hi_q = 0.5 + block_frac / 2.0
    lo = np.quantile(pos, lo_q, axis=0)
    hi = np.quantile(pos, hi_q, axis=0)
    mask_te = (
        (pos[:, 0] >= lo[0]) & (pos[:, 0] <= hi[0]) &
        (pos[:, 1] >= lo[1]) & (pos[:, 1] <= hi[1]) &
        (pos[:, 2] >= lo[2]) & (pos[:, 2] <= hi[2])
    )

    idx_te = np.where(mask_te)[0]
    idx_rest = np.where(~mask_te)[0]
    if len(idx_te) < 10:
        raise RuntimeError(
            f"Interior block produced too few test samples: {len(idx_te)}. "
            f"Try increasing block_frac."
        )

    # train/val split from the remaining set
    X_rest = X[idx_rest]
    y_rest = y[idx_rest]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rest, y_rest, test_size=val_ratio, random_state=seed, shuffle=True
    )
    X_te = X[idx_te]
    y_te = y[idx_te]

    print(
        f"[Split] mode=interior_block  block_frac={block_frac}"
        f"  test(block)={len(X_te):,}  rest={len(X_rest):,}"
    )
    print(f"[Split] train={len(X_tr):,}  val={len(X_val):,}  test={len(X_te):,}\n")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ============================================================
# LEARNING CURVE  (loss vs max_depth)
# ============================================================

def run_learning_curve(X_tr, y_tr, X_val, y_val,
                       criterion, max_depth, seed,
                       min_leaf_list, min_split, max_features_list,
                       splitter: str, separate_heads: bool):
    """
    Fit DT với từng depth 1 → max_depth, ghi train/val loss.

    Returns
    -------
    depths       : list[int]
    train_losses : list[float]
    val_losses   : list[float]
    """
    depths, train_losses, val_losses = [], [], []
    best_overall = {"val_loss": None, "depth": None, "min_leaf": None, "max_features": None}

    print(f"\n[Curve] Sweep depth/leaf/max_features")
    print(f"  depth: 1 → {max_depth}")
    print(f"  min_leaf: {min_leaf_list}")
    print(f"  max_features: {max_features_list}")
    print(f"  splitter={splitter}  separate_heads={separate_heads}\n")

    print(f"{'Depth':>6}  {'min_leaf':>8}  {'max_feat':>10}  {'Train':>12}  {'Val':>12}  {'Time':>7}")
    print("-" * 75)

    for d in range(1, max_depth + 1):
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

                pred_tr  = predict(m, backend, X_tr).astype(np.float64)
                pred_val = predict(m, backend, X_val).astype(np.float64)

                tl, _, _ = criterion(pred_tr,  y_tr.astype(np.float64))
                vl, _, _ = criterion(pred_val, y_val.astype(np.float64))

                if best_vl is None or vl < best_vl:
                    best_vl = vl
                    best_row = (leaf, mf, tl, vl)
                if best_overall["val_loss"] is None or vl < best_overall["val_loss"]:
                    best_overall = {"val_loss": vl, "depth": d, "min_leaf": leaf, "max_features": mf}

                print(
                    f"{d:>6}  {leaf:>8}  {str(mf):>10}  {tl:>12.6f}  {vl:>12.6f}  {time.time()-t0:>6.1f}s"
                )

        # store best val per depth (for plotting)
        leaf, mf, tl, vl = best_row
        depths.append(d)
        train_losses.append(tl)
        val_losses.append(vl)

    print(
        "\n[Curve] Best overall"
        f"  depth={best_overall['depth']}"
        f"  min_leaf={best_overall['min_leaf']}"
        f"  max_features={best_overall['max_features']}"
        f"  val_loss={best_overall['val_loss']:.6f}\n"
    )
    return depths, train_losses, val_losses, best_overall


def _parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_max_features_list(s: str):
    out = []
    for raw in [x.strip() for x in s.split(",") if x.strip()]:
        if raw.lower() == "none":
            out.append(None)
        elif raw.lower() in ("sqrt", "log2"):
            out.append(raw.lower())
        else:
            out.append(float(raw))
    return out


# ============================================================
# PLOT
# ============================================================

def plot_loss_curve(depths, train_losses, val_losses,
                    save_path: str, title: str = "Learning Curve"):
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

    # Load & split
    X, y = load_data(args.emf, args.label)
    if args.split_mode == "random":
        X_tr_raw, X_val_raw, X_te_raw, y_tr, y_val, y_te = split_data(
            X, y, args.val_ratio, args.test_ratio, args.seed)
    else:
        X_tr_raw, X_val_raw, X_te_raw, y_tr, y_val, y_te = split_data_interior_block(
            X, y, args.val_ratio, args.block_frac, args.seed)

    # Luu test split ra file de test.py load dung phan da tach
    test_emf_path   = os.path.join(args.ckpt_dir, 'test_emf.csv')
    test_label_path = os.path.join(args.ckpt_dir, 'test_label.csv')
    pd.DataFrame(X_te_raw).to_csv(test_emf_path, index=False, header=False)
    pd.DataFrame(y_te, columns=['x_m','y_m','z_m','cos_roll','cos_pitch','cos_yaw']
                 ).to_csv(test_label_path, index=False)
    print(f'[Split] test_emf   -> {test_emf_path}')
    print(f'[Split] test_label -> {test_label_path}')

    # Preprocess (fit on train only, apply to val/test)
    preproc = EMFPreprocessor(
        use_signed_log=args.use_signed_log,
        use_standardize=(not args.no_standardize),
    )
    X_tr  = preproc.fit_transform(X_tr_raw)
    X_val = preproc.transform(X_val_raw)
    X_te  = preproc.transform(X_te_raw)
    print(f"[Preprocess] signed_log={args.use_signed_log}  standardize={not args.no_standardize}")

    # Loss
    criterion = HuberPoseLoss(ang_weight=args.ang_weight,
                               delta_xyz=args.delta_xyz,
                               delta_ang=args.delta_ang)
    print(f"[Loss] HuberPoseLoss  ang_weight={args.ang_weight}"
          f"  delta_xyz={args.delta_xyz}  delta_ang={args.delta_ang}")

    # Learning curve
    min_leaf_list = _parse_int_list(args.min_leaf_list)
    max_features_list = _parse_max_features_list(args.max_features_list)
    depths, train_losses, val_losses, best_overall = run_learning_curve(
        X_tr, y_tr, X_val, y_val,
        criterion, args.max_depth, args.seed,
        min_leaf_list=min_leaf_list,
        min_split=args.min_split,
        max_features_list=max_features_list,
        splitter=args.splitter,
        separate_heads=args.separate_heads,
    )

    # Plot
    plot_path = os.path.join(args.ckpt_dir, "learning_curve.png")
    plot_loss_curve(depths, train_losses, val_losses,
                    save_path=plot_path,
                    title="Decision Tree — Huber Loss vs max_depth")

    # Fit final model ở best params
    best_depth = int(best_overall["depth"])
    best_leaf = int(best_overall["min_leaf"])
    best_mf = best_overall["max_features"]
    print(
        f"\n[Final] Fit model với best params:"
        f" depth={best_depth}  min_leaf={best_leaf}  max_features={best_mf}"
    )
    t0 = time.time()
    model, backend = build_model(
        max_depth=best_depth,
        random_state=args.seed,
        min_samples_leaf=best_leaf,
        min_samples_split=args.min_split,
        max_features=best_mf,
        splitter=args.splitter,
        separate_heads=args.separate_heads,
    )
    model = fit(model, backend, X_tr, y_tr)
    print(f"[Final] Xong ({time.time()-t0:.1f}s)")

    # Evaluate test
    pred_te  = predict(model, backend, X_te).astype(np.float64)
    tl_f, lx_f, la_f = criterion(
        predict(model, backend, X_tr).astype(np.float64), y_tr.astype(np.float64))
    vl_f, _, _        = criterion(
        predict(model, backend, X_val).astype(np.float64), y_val.astype(np.float64))
    te_l, te_x, te_a  = criterion(pred_te, y_te.astype(np.float64))

    # Inference time
    t_inf  = time.time()
    _      = predict(model, backend, X_te)
    inf_ms = (time.time() - t_inf) / len(X_te) * 1000

    # Save checkpoint
    ckpt = os.path.join(args.ckpt_dir, "dt_model.pkl")
    with open(ckpt, "wb") as f:
        pickle.dump({"model": model, "backend": backend, "preproc": preproc,
                     "best_depth": best_depth, "args": vars(args)}, f)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    print(f"  Backend       : {backend}")
    print(f"  Best depth    : {best_depth}")
    print(f"  Train Loss    : {tl_f:.6f}  (xyz={lx_f:.6f}  ang={la_f:.6f})")
    print(f"  Val   Loss    : {vl_f:.6f}")
    print(f"  Test  Loss    : {te_l:.6f}  (xyz={te_x:.6f}  ang={te_a:.6f})")
    print(f"  Infer time    : {inf_ms:.4f} ms/sample")
    print(f"  Checkpoint    : {ckpt}")
    print(f"  Plot          : {plot_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
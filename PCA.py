import os
import shutil
from pathlib import Path
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from zerosylprobes.train_linear_probe import (
    frames_to_zerosyl_segments,
    print_results_table,
)
from zerosyl.eval.boundaries import evaluate_boundary_metrics

FEATURE_OF_INTEREST = 376   # the single WavLM dimension we compare against the PCs
BATCH = 20000               # rows processed at a time, to keep memory use low


# ------------------------------------------------------------------
# 1. Exact PCA via batched covariance accumulation
# ------------------------------------------------------------------
def fit_exact_pca(X_tr, standardize=False, batch_size=BATCH):
    """
    Returns mean, scale, eigvals (desc), comps (D x D, columns = PCs, desc).
    standardize=False -> centre only (preserves raw geometry / L2 norm).
    """
    n, d = X_tr.shape
    # running totals: sum of rows, and sum of (row x row) products
    s = np.zeros(d, dtype=np.float64)
    ss = np.zeros((d, d), dtype=np.float64)
    # go through the data in batches so it never has to fit in memory at once
    for a in range(0, n, batch_size):
        b = np.asarray(X_tr[a:a + batch_size], dtype=np.float64)
        s += b.sum(0)
        ss += b.T @ b
    # covariance = E[x x^T] - mean mean^T
    mean = s / n
    cov = ss / n - np.outer(mean, mean)

    if standardize:
        # divide by each dimension's std (turns the covariance into a correlation matrix)
        scale = np.sqrt(np.diag(cov)) + 1e-12
        cov = cov / np.outer(scale, scale)
    else:
        scale = np.ones(d)

    # eigh returns eigenvalues smallest-first, so re-sort largest-first
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, comps = eigvals[order], eigvecs[:, order]
    # print how much variance the top 1, 10, 100 and 200 PCs hold
    print(f"[PCA] standardize={standardize}; top-1/10/100/200 cumulative "
          f"variance: " + ", ".join(
              f"{eigvals[:k].sum() / eigvals.sum() * 100:.1f}%"
              for k in (1, 10, 100, 200)))
    return mean, scale, eigvals, comps


def project(X, mean, scale, comps, k, batch_size=BATCH):
    # project the data onto the first k PCs (again in batches)
    n = X.shape[0]
    out = np.empty((n, k), dtype=np.float32)
    W = comps[:, :k].astype(np.float32)
    m = mean.astype(np.float32)
    sc = scale.astype(np.float32)
    for a in range(0, n, batch_size):
        b = np.asarray(X[a:a + batch_size], dtype=np.float32)
        # centre (and scale if standardising), then rotate onto the PCs
        out[a:a + batch_size] = ((b - m) / sc) @ W
    return out


# ------------------------------------------------------------------
# 2. Evaluate a set of frame-level probabilities
# ------------------------------------------------------------------
def score_probs(name, probs_ev, threshold, utt_ev, tg_ev, frame_rate=50.0):
    # folders for the predicted segments and a copy of the matching TextGrids
    pred_dir = Path(f"pred_segments_{name}").resolve()
    tg_dir = Path(f"eval_textgrids_{name}").resolve()
    # start from empty folders so old results don't get mixed in
    for d in (pred_dir, tg_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # turn probabilities into 0/1 boundary predictions
    y_pred = (probs_ev > threshold).astype(int)
    # save each utterance's predicted segments and copy its TextGrid
    for utt in np.unique(utt_ev):
        m = utt_ev == utt
        torch.save(frames_to_zerosyl_segments(y_pred[m]), pred_dir / f"{utt}.pt")
        shutil.copy(Path(tg_ev[m][0]).resolve(), tg_dir / f"{utt}.TextGrid")

    # compare predicted boundaries to the true ones (50 ms tolerance)
    p, r, f1, os_val, rval, tp, tr, tf1 = evaluate_boundary_metrics(
        segments_dir=pred_dir, textgrid_dir=tg_dir,
        segments_pattern="*.pt", textgrid_pattern="*.TextGrid",
        frame_rate=frame_rate, constant_shift=0.0, tolerance=0.05,
    )
    return {"P": p, "R": r, "F1": f1, "OS": os_val, "R-val": rval,
            "Token-P": tp, "Token-R": tr, "Token-F1": tf1}


def make_val_split(utt_tr, val_frac=0.1, seed=42):
    # pick a random 10% of utterances and return the indices of their frames
    rng = np.random.RandomState(seed)
    u = np.unique(utt_tr)
    val_utts = rng.choice(u, size=max(1, int(len(u) * val_frac)), replace=False)
    return np.where(np.isin(utt_tr, val_utts))[0]





def train_and_eval(name, cols, Z_tr, y_tr, Z_ev, utt_ev, tg_ev):
    """
    Fit logreg on the chosen PC columns using EXACTLY the same settings as the
    full-vector probe (class_weight="balanced", max_iter=100, default C=1.0,
    default 0.5 decision threshold, no threshold tuning), then evaluate.
    """
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",   # boundaries are rare, so weight them up
        random_state=42,
        n_jobs=1,
    )
    # only the chosen PC columns are used
    clf.fit(Z_tr[:, cols], y_tr)
    y_pred = clf.predict(Z_ev[:, cols])          # 0.5 threshold, like the full probe
    res = score_probs(name, y_pred.astype(np.float32), 0.5, utt_ev, tg_ev)
    print(f"  {name}: Token-F1 = {res['Token-F1'] * 100:.2f}%")
    return res, clf


# ------------------------------------------------------------------
# 3. Analysis relating the PCA basis to feature 376 / L2 norm
# ------------------------------------------------------------------
def analyse_axis_vs_pcs(comps, eigvals, feat=FEATURE_OF_INTEREST, top=10):
    """How is the raw axis e_feat spread across the PCs?"""
    load = comps[feat, :]                # e_feat expressed in PC coordinates
    energy = load ** 2                   # sums to 1
    cum = np.cumsum(energy)
    print(f"\n[axis {feat} in PC basis]")
    # how many PCs it takes to cover 50%, 90% and 99% of the axis
    for frac in (0.5, 0.9, 0.99):
        print(f"  PCs needed to hold {frac * 100:.0f}% of axis-{feat} energy: "
              f"{int(np.searchsorted(cum, frac) + 1)}")
    # small number = concentrated in a few PCs, large number = spread over many
    part_ratio = 1.0 / np.sum(energy ** 2)
    print(f"  participation ratio (effective # PCs): {part_ratio:.1f}")
    # the PCs that load most heavily on this feature
    print(f"  top {top} PCs by loading on axis {feat}:")
    for i in np.argsort(energy)[::-1][:top]:
        print(f"    PC{i:<4d} energy={energy[i]:.3f}  "
              f"(PC variance rank {i}, eigval={eigvals[i]:.4f})")


def correlate_pcs_with_norm(Z_ev, X_dev, top=10):
    """Pearson correlation of each PC score with the raw L2 norm."""
    # z-score both sides, so the dot product / n is the Pearson correlation
    norms = np.linalg.norm(np.asarray(X_dev, dtype=np.float32), axis=1)
    norms = (norms - norms.mean()) / (norms.std() + 1e-12)
    Zs = (Z_ev - Z_ev.mean(0)) / (Z_ev.std(0) + 1e-12)
    corr = Zs.T @ norms / len(norms)
    print("\n[PC correlation with raw L2 norm]")
    # show the PCs most strongly correlated with the norm (positive or negative)
    for i in np.argsort(np.abs(corr))[::-1][:top]:
        print(f"    PC{i:<4d} r={corr[i]:+.3f}")
    return corr


def probe_direction_in_original_space(clf, comps, scale, k, feat=FEATURE_OF_INTEREST):
    """Map PC-space probe weights back to the original 1024-D axes."""
    w_pc = clf.coef_.ravel()
    w_orig = (comps[:, :k] @ w_pc) / scale
    # unit vector pointing along the feature of interest
    e = np.zeros_like(w_orig)
    e[feat] = 1.0
    # cosine = how closely the probe direction lines up with that one feature
    cos = float(w_orig @ e / (np.linalg.norm(w_orig) + 1e-12))
    # share = fraction of the probe's squared weight sitting on that feature
    share = float(w_orig[feat] ** 2 / np.sum(w_orig ** 2))
    print(f"    cosine(w_orig, e_{feat}) = {cos:+.3f}; "
          f"share of squared weight on dim {feat} = {share * 100:.2f}%")
    return w_orig


# ------------------------------------------------------------------
def main(standardize=False,
         k_variance=(1024, 200, 100, 50, 20, 10, 5, 2, 1),
         k_max=1024):
    # k=1024 is a control: a full rotation keeps all the information, so it
    # should reproduce the full-vector probe. If memory is tight, remove 1024
    # from k_variance and set k_max=200.
    d = "saved_features"
    # train features are memory-mapped (big), dev is loaded fully
    X_tr = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
    y_tr = np.load(os.path.join(d, "y_train.npy"), mmap_mode="r")
    utt_tr = np.load(os.path.join(d, "utt_train.npy"))
    tg_tr = np.load(os.path.join(d, "tg_train.npy"))
    X_dev = np.load(os.path.join(d, "X_dev.npy"))
    utt_dev = np.load(os.path.join(d, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(d, "tg_dev.npy"))
    y_tr = np.asarray(y_tr)

    # fit PCA on train only, then see how feature 376 is spread across the PCs
    mean, scale, eigvals, comps = fit_exact_pca(X_tr, standardize=standardize)
    analyse_axis_vs_pcs(comps, eigvals)

    # project train and dev onto the PCs and check which ones track the norm
    print(f"\nProjecting to top {k_max} PCs...")
    Z_tr = project(X_tr, mean, scale, comps, k_max)
    Z_ev = project(X_dev, mean, scale, comps, k_max)
    corr = correlate_pcs_with_norm(Z_ev, X_dev)

    # "std" or "ctr" (centred), used to name the output files
    tag = "std" if standardize else "ctr"
    results = {}

    # A) top-K PCs by variance
    print("\n[A] Top-K PCs by variance")
    clfs = {}
    for k in k_variance:
        name = f"pca{k}_{tag}"
        res, clf = train_and_eval(name, list(range(k)), Z_tr, y_tr, Z_ev,
                                  utt_dev, tg_dev)
        results[f"Top {k} PCs (variance)"] = res
        clfs[k] = clf
        probe_direction_in_original_space(clf, comps, scale, k)

    # B) top-K PCs by probe importance (|coef| * PC std) from the k_max probe
    print("\n[B] Top-K PCs by probe weight (from the largest probe)")
    big = clfs[max(clfs)]
    # importance = weight size x how much the PC varies (sqrt of eigenvalue)
    importance = np.abs(big.coef_.ravel()) * np.sqrt(eigvals[:len(big.coef_.ravel())])
    ranked = np.argsort(importance)[::-1]
    print("  most important PCs:", ranked[:10].tolist())
    # retrain using only the most important PCs
    for k in (20, 10, 5, 2, 1):
        name = f"pca_imp{k}_{tag}"
        res, _ = train_and_eval(name, ranked[:k].tolist(), Z_tr, y_tr, Z_ev,
                                utt_dev, tg_dev)
        results[f"Top {k} PCs (importance)"] = res

    # ---- save everything the plotting script needs ----
    with open(f"pca_probe_results_{tag}.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(f"pca_analysis_{tag}.npz",
             eigvals=eigvals, mean=mean, scale=scale,
             comps_top=comps[:, :50], corr=corr,
             axis_energy=comps[FEATURE_OF_INTEREST, :] ** 2,
             feature=FEATURE_OF_INTEREST)
    print(f"Saved pca_probe_results_{tag}.json and pca_analysis_{tag}.npz")

    # final summary table
    print_results_table(results, title=f"PCA PROBE ({tag}) vs FEATURE {FEATURE_OF_INTEREST}")


if __name__ == "__main__":
    main(standardize=False)   # centre-only: preserves L2-norm geometry
    # main(standardize=True)  # run again to compare with z-scored PCA
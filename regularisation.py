"""
Fit ONE L1-regularised logistic regression probe, save it, and evaluate it
with the real segment-level boundary metric.

  1. z-score TRAIN only -> fit the model ->
     save the model + scaler stats to disk -> free the (large) train array.
  2. THEN z-score DEV (much smaller) and evaluate.

"""

import os

# limit how many CPU threads numpy / sklearn can use (2 unless already set)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import matplotlib.pyplot as plt
import numpy as np

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from zerosyl.eval.boundaries import evaluate_boundary_metrics

OUTPUT_DIR = Path("saved_features")

# Model artifact -- small, single file, safe to keep in your project folder.
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "l1_probe.joblib"       # the trained probe
META_FILE = MODEL_DIR / "l1_probe_meta.json"     # settings + the mean/std used to z-score

# Per-utterance eval files -- kept outside the workspace, as before, to avoid
# overloading VS Code's file watcher.
OUTPUT_BASE = Path(os.environ.get("ZEROSYL_TMP_DIR", tempfile.gettempdir())) / "zerosyl_probe_outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# C is the inverse regularisation strength: smaller C = stronger L1 = fewer features kept
C = float(os.environ.get("L1_PROBE_C", "0.0001"))
MAX_ITER = int(os.environ.get("L1_PROBE_MAX_ITER", "500"))
TOL = float(os.environ.get("L1_PROBE_TOL", "1e-3"))


def log_mem(tag=""):
    # print the peak memory used so far (just for monitoring)
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"  [mem] peak RSS so far{(' @ ' + tag) if tag else ''}: {peak_mb:.0f} MB")
    except Exception:
        # the resource module doesn't exist on every system, so just skip
        pass


def compute_train_mean_std(train_path, chunk_rows=50_000):
    """Streaming pass: never loads the full train array into RAM."""
    X_raw = np.load(train_path, mmap_mode="r")
    n, d = X_raw.shape
    # running totals of x and x^2 for every feature
    sum_ = np.zeros(d, dtype=np.float64)
    sumsq = np.zeros(d, dtype=np.float64)
    # go through the file one chunk at a time (memory saving)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        chunk = X_raw[start:end].astype(np.float64)
        sum_ += chunk.sum(axis=0)
        sumsq += np.square(chunk).sum(axis=0)
    mean = (sum_ / n).astype(np.float32)
    # variance = E[x^2] - (E[x])^2
    var = (sumsq / n) - (sum_ / n) ** 2
    # the tiny floor stops a divide-by-zero for features that never change
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return mean, std


def zscore_apply(path, mean, std, chunk_rows=50_000):
    """Chunked z-scoring using precomputed mean/std -- used for both train
    (once, during fitting) and dev (later, after train is freed)."""
    X_raw = np.load(path, mmap_mode="r")
    n, d = X_raw.shape
    # the output is stored as float32 to save memory
    X_z = np.empty((n, d), dtype=np.float32)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        X_z[start:end] = (X_raw[start:end].astype(np.float32) - mean) / std
    return X_z


def frames_to_segments(pred_frames):
    # turns a 0/1 array of predicted boundaries (one per frame) into
    # [start, end, 0] segments, which is the format the evaluation expects
    boundary_indices = np.where(pred_frames == 1)[0]
    total_frames = len(pred_frames)
    if len(boundary_indices) > 0:
        # neighbouring boundary frames count as one boundary, so keep the middle of each run
        runs = np.split(boundary_indices, np.where(np.diff(boundary_indices) != 1)[0] + 1)
        boundary_indices = np.array([r[len(r) // 2] for r in runs])
    else:
        boundary_indices = np.array([])
    # the utterance start and end are always cut points
    points = sorted(set([0, total_frames, *boundary_indices.tolist()]))
    # each segment runs from one cut point to the next
    segments = [[points[i], points[i + 1], 0] for i in range(len(points) - 1)]
    # fallback: treat the whole utterance as one segment
    if not segments:
        segments = [[0, total_frames, 0]]
    return torch.tensor(segments, dtype=torch.long)


def main():
    # compute z-score stats and z-score train only
    print(f"Fitting L1 probe: C={C}, max_iter={MAX_ITER}, tol={TOL}")
    print("\nComputing train mean/std (streaming, low memory)...")
    mean, std = compute_train_mean_std(OUTPUT_DIR / "X_train.npy")
    log_mem("after computing mean/std")

    print("Z-scoring train set...")
    X_train_z = zscore_apply(OUTPUT_DIR / "X_train.npy", mean, std)
    y_train = np.load(OUTPUT_DIR / "y_train.npy")
    log_mem("after z-scoring train")

    #Fit ONE L1 model ----
    print("\nFitting L1 logistic regression (single fit, no CV)...")
    t0 = time.perf_counter()
    clf = LogisticRegression(
        C=C, l1_ratio=1.0,          # l1_ratio=1.0 == pure L1
        solver="saga",              # required for L1 via the l1_ratio API
        class_weight="balanced",    # boundaries are rare, so weight them up
        max_iter=MAX_ITER, tol=TOL, random_state=42,
    )
    clf.fit(X_train_z, y_train)
    # if n_iter hit the max, the solver stopped before it converged
    n_iter = int(np.max(clf.n_iter_))
    print(f"Done in {time.perf_counter() - t0:.1f}s  "
          f"(n_iter={n_iter}/{MAX_ITER}"
          f"{' -- did not fully converge, consider raising L1_PROBE_MAX_ITER' if n_iter >= MAX_ITER else ' -- converged'})")
    log_mem("after fitting")

    # how many features did L1 keep (weight not zero)?
    coefs = clf.coef_.ravel()
    n_nonzero = int(np.sum(np.abs(coefs) > 1e-6))
    print(f"\nSparsity: {n_nonzero}/{len(coefs)} nonzero weights "
          f"({100 * n_nonzero / len(coefs):.1f}% of features kept)")

    #Save the model now, before moving to dev
    joblib.dump(clf, MODEL_FILE)
    # save the settings and the train mean/std so the same z-scoring can be reused later
    with open(META_FILE, "w") as f:
        json.dump({
            "C": C, "max_iter": MAX_ITER, "tol": TOL, "n_iter": n_iter,
            "n_nonzero": n_nonzero, "n_features": int(len(coefs)),
            "mean": mean.tolist(), "std": std.tolist(),
        }, f)
    print(f"Saved model to {MODEL_FILE} and {META_FILE}")

    #Free the big train array before loading dev
    del X_train_z, y_train
    gc.collect()
    log_mem("after freeing train array")

    #Z-score dev (small) and evaluate
    print("\nZ-scoring dev set and evaluating...")
    # dev is z-scored with the train mean/std, not its own
    X_dev_z = zscore_apply(OUTPUT_DIR / "X_dev.npy", mean, std)
    y_dev = np.load(OUTPUT_DIR / "y_dev.npy")
    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    tg_dev = np.load(OUTPUT_DIR / "tg_dev.npy")

    # quick frame-by-frame F1 (sanity check)
    dev_frame_f1 = f1_score(y_dev, clf.predict(X_dev_z))
    print(f"Dev frame-level F1 (quick sanity check): {dev_frame_f1:.4f}")

    # 0/1 prediction for every dev frame
    y_pred = clf.predict(X_dev_z)
    # folders for the predicted segments and a copy of the matching TextGrids
    pred_dir = OUTPUT_BASE / "pred_segments_l1_probe"
    matched_tg_dir = OUTPUT_BASE / "eval_textgrids_l1_probe"
    # start from empty folders so old results don't get mixed in
    for d in [pred_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # save each utterance's predicted segments and copy its TextGrid
    unique_utts = np.unique(utt_dev)
    for utt in unique_utts:
        mask = (utt_dev == utt)
        utt_tg_path = Path(tg_dev[mask][0]).resolve()
        torch.save(frames_to_segments(y_pred[mask]), pred_dir / f"{utt}.pt")
        tg_file = matched_tg_dir / f"{utt}.TextGrid"
        if not tg_file.exists():
            shutil.copy(utt_tg_path, tg_file)

    # compare predicted boundaries to the true ones (50 ms tolerance)
    metrics = evaluate_boundary_metrics(
        segments_dir=pred_dir, textgrid_dir=matched_tg_dir,
        segments_pattern="*.pt", textgrid_pattern="*.TextGrid",
        frame_rate=50.0, constant_shift=0.0, tolerance=0.05,
    )
    p, r, f1, os_val, rval, tp, tr, tf1 = metrics
    print("\nReal segment-level boundary metrics on dev:")
    for name, val in [("P", p), ("R", r), ("F1", f1), ("OS", os_val),
                       ("R-val", rval), ("Token-P", tp), ("Token-R", tr), ("Token-F1", tf1)]:
        print(f"  {name:<9}: {val:.4f}")

    
    # sort features by the size of their weight, biggest first
    order = np.argsort(np.abs(coefs))[::-1]
    # keep the top 20 that are not zero
    nonzero_order = [i for i in order if abs(coefs[i]) > 1e-6][:20]
    print(f"\nTop nonzero features (this is your interpretability result):")
    print(f"  {'feature':<12} | {'coef':>10} | direction")
    print(f"  {'-'*12}-+-{'-'*10}-+----------")
    for i in nonzero_order:
        # positive weight pushes towards "boundary", negative towards "no boundary"
        direction = "-> boundary" if coefs[i] > 0 else "-> no boundary"
        print(f"  feat_{i:<7} | {coefs[i]:>10.4f} | {direction}")


    # Load coefficients from the probe
    coefs = clf.coef_.ravel()
    active_indices = np.where(np.abs(coefs) > 1e-6)[0]

    # Visualize active weight magnitudes
    # one stem per surviving feature, height = its weight
    plt.figure(figsize=(10, 4))
    plt.stem(active_indices, coefs[active_indices], markerfmt=" ", basefmt="k-")
    plt.xlabel("Feature / Hidden Dimension Index")
    plt.ylabel("Weight Magnitude")
    plt.title("L1 Active Feature Weights for Syllable Boundary Probing")
    plt.grid(True, alpha=0.3)
    plt.show()

    log_mem("end of script")


if __name__ == "__main__":
    main()
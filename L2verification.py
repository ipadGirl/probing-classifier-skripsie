import os

##limit how many CPU threads numpy/sklearn can use
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import gc
import json
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from zerosyl.eval.boundaries import evaluate_boundary_metrics

OUTPUT_DIR = Path("saved_features")
MODEL_DIR = Path("models")
L1_MODEL_FILE = MODEL_DIR / "l1_probe.joblib"

##temporary folder for the prediction files
OUTPUT_BASE = Path(os.environ.get("ZEROSYL_TMP_DIR", tempfile.gettempdir())) / "zerosyl_probe_outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Hyperparameters for L2 Verification
K = int(os.environ.get("L2_VERIFY_K", "5")) # how many top k features to keep
C_L2 = float(os.environ.get("L2_VERIFY_C", "1.0"))  # Standard unconstrained / weakly regularised L2


def frames_to_zerosyl_segments(pred_frames):
    # turns a 0/1 array of predicted boundaries (one per frame) into a list of segments expected by evaluation code
    boundary_indices = np.where(pred_frames == 1)[0]
    total_frames = len(pred_frames)

    if len(boundary_indices) > 0:
        runs = np.split(boundary_indices, np.where(np.diff(boundary_indices) != 1)[0] + 1)
        boundary_indices = np.array([r[len(r) // 2] for r in runs])
    else:
        boundary_indices = np.array([])

    #make sure the utterance start and end are always cut points
    points = list(boundary_indices)
    if 0 not in points:
        points.insert(0, 0)
    if total_frames not in points:
        points.append(total_frames)
    points = sorted(set(points))

    #each segment runs from one cut point to the next
    segments = [[points[i], points[i + 1], 0] for i in range(len(points) - 1)]
    #fallback: if nothing was made - treat the whole utterance as one segment
    if not segments:
        segments.append([0, total_frames, 0])

    return torch.tensor(segments, dtype=torch.long)


def main():
    print(f"==================================================")
    print(f" Running L2 Verification Test on Top-{K} Features")
    print(f"==================================================")

    # 1. Load L1 model and rank feature indices
    ## the L1 probe must be trained and saved
    if not L1_MODEL_FILE.exists():
        raise FileNotFoundError(f"Could not find L1 model at {L1_MODEL_FILE}. Run L1 probe first!")

    l1_clf = joblib.load(L1_MODEL_FILE)
    l1_coefs = l1_clf.coef_.ravel()
    
    # Sort features by absolute L1 weight magnitude
    # [::-1] flips it so the biggest weights come first
    top_k_indices = np.argsort(np.abs(l1_coefs))[::-1][:K]
    print(f"Extracted top {K} features from L1 probe weights.")

    # 2. Load raw features and slice columns to keep top-K only
    # mmap_mode="r" avoids loading the whole file into memory before slicing
    print("\nLoading and slicing dataset features...")
    X_train_raw = np.load(OUTPUT_DIR / "X_train.npy", mmap_mode="r")[:, top_k_indices]
    y_train = np.load(OUTPUT_DIR / "y_train.npy")

    X_dev_raw = np.load(OUTPUT_DIR / "X_dev.npy", mmap_mode="r")[:, top_k_indices]
    y_dev = np.load(OUTPUT_DIR / "y_dev.npy")
    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    tg_dev = np.load(OUTPUT_DIR / "tg_dev.npy")

    # 3. Z-score standard scaling fit ON TRAIN ONLY
    # (dev uses the train mean/std so nothing leaks from dev into training)
    print("Z-scoring sliced features...")
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train_raw.astype(np.float32, copy=False))
    X_dev_z = scaler.transform(X_dev_raw.astype(np.float32, copy=False))

    # free the unscaled copies
    del X_train_raw, X_dev_raw
    gc.collect()

    # 4. Train L2 Logistic Regression on top-K space
    print(f"\nFitting L2 probe on top-{K} subspace (C={C_L2})...")
    t0 = time.perf_counter()
    l2_clf = LogisticRegression(
        penalty="l2",
        C=C_L2,
        solver="lbfgs",
        class_weight="balanced", ##NB
        max_iter=500,
        random_state=42,
        n_jobs=4,
    )
    l2_clf.fit(X_train_z, y_train)
    print(f"Done fitting L2 probe in {time.perf_counter() - t0:.2f} seconds.")

    # Clean up train data memory
    del X_train_z, y_train
    gc.collect()

    # 5. Evaluate predictions on dev set
    print("\nGenerating boundary predictions and evaluating...")
    ## 0/1 prediction for every dev frame
    y_pred = l2_clf.predict(X_dev_z)

    #folders for the predicted segments and a copy of the matching TextGrids
    pred_dir = OUTPUT_BASE / f"pred_segments_l2_top{K}"
    matched_tg_dir = OUTPUT_BASE / f"eval_textgrids_l2_top{K}"

    # start from empty folders so old results don't get mixed in
    for d in [pred_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # go through each utterance and save its predictions and its TextGrid
    unique_utts = np.unique(utt_dev)
    for utt in unique_utts:
        mask = (utt_dev == utt)
        utt_tg_path = Path(tg_dev[mask][0]).resolve()

        torch.save(frames_to_zerosyl_segments(y_pred[mask]), pred_dir / f"{utt}.pt")
        tg_file = matched_tg_dir / f"{utt}.TextGrid"
        if not tg_file.exists():
            shutil.copy(utt_tg_path, tg_file)

    # compare predicted boundaries to the true ones (50 ms tolerance)
    metrics = evaluate_boundary_metrics(
        segments_dir=pred_dir,
        textgrid_dir=matched_tg_dir,
        segments_pattern="*.pt",
        textgrid_pattern="*.TextGrid",
        frame_rate=50.0,
        constant_shift=0.0,
        tolerance=0.05,
    )

    p, r, f1, os_val, rval, tp, tr, tf1 = metrics

    print("\n" + "=" * 65)
    print(f" L2 VERIFICATION TEST RESULTS (Top K={K} Features)")
    print("=" * 65)
    print(f"  Token Precision : {tp * 100:.2f}%")
    print(f"  Token Recall    : {tr * 100:.2f}%")
    print(f"  Token-F1 Score  : {tf1 * 100:.2f}%  <-- KEY METRIC")
    print(f"  Frame-level F1  : {f1 * 100:.2f}%")
    print(f"  R-value         : {rval:.4f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
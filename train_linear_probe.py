import os
import shutil
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

from zerosyl.eval.boundaries import evaluate_boundary_metrics
from zerosylprobes.normseg import generate_normseg_preds

def select_threshold_by_boundary_metric(scores, utt_ids, tg_paths,
                                          frame_rate=50.0, tolerance=0.05,
                                          n_thresholds=25, metric="F1",
                                          tmp_dir="threshold_search"):
    # tries a range of thresholds on the probe's scores and returns the one
    # that gives the best boundary metric (instead of just using 0.5)
    tmp_dir = Path(tmp_dir).resolve()
    # start from an empty working folder
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    pred_dir = tmp_dir / "pred"
    tg_dir = tmp_dir / "tg"
    pred_dir.mkdir(parents=True)
    tg_dir.mkdir(parents=True)

    # copy each utterance's TextGrid in once (they don't change between thresholds)
    unique_utts = np.unique(utt_ids)
    for utt in unique_utts:
        mask = (utt_ids == utt)
        tg_file = tg_dir / f"{utt}.TextGrid"
        if not tg_file.exists():
            shutil.copy(Path(tg_paths[mask][0]).resolve(), tg_file)

    # candidate thresholds spread evenly between the lowest and highest score
    candidates = np.linspace(scores.min(), scores.max(), n_thresholds)
    # position of the chosen metric in the tuple evaluate_boundary_metrics returns
    metric_idx = {"P": 0, "R": 1, "F1": 2, "OS": 3, "R-val": 4}[metric]
    best_thr, best_score = candidates[0], -np.inf

    for thr in candidates:
        # frames above the threshold are predicted boundaries
        pred_frames_all = (scores > thr).astype(int)
        for utt in unique_utts:
            mask = (utt_ids == utt)
            seg = frames_to_zerosyl_segments(pred_frames_all[mask])
            torch.save(seg, pred_dir / f"{utt}.pt")

        # score this threshold with the real boundary metric
        m = evaluate_boundary_metrics(
            segments_dir=pred_dir, textgrid_dir=tg_dir,
            segments_pattern="*.pt", textgrid_pattern="*.TextGrid",
            frame_rate=frame_rate, constant_shift=0.0, tolerance=tolerance
        )
        val = m[metric_idx]
        # keep the best one (ignoring nan / inf results)
        if np.isfinite(val) and val > best_score:
            best_score, best_thr = val, thr

    print(f"Selected threshold={best_thr:.4f} ({metric}={best_score:.4f})")
    return best_thr
def fit_and_predict_classic_rbf(X_tr, y_tr, X_ev, utt_tr=None, tg_tr=None,
                                 gamma=0.001, n_centres=2000, alpha=1.0,
                                 batch_size=20000, threshold=None,
                                 val_frac=0.1, metric="F1", random_state=42):
    # trains an RBF kernel ridge probe on a random subset of training frames
    # and returns 0/1 predictions plus 0-1 scores for every eval frame
    X_tr = X_tr.astype(np.float32, copy=False)
    X_ev = X_ev.astype(np.float32, copy=False)

    # z-score using the train mean/std (the same scaler is applied to eval later)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    # Select a random subset of training points to serve as kernel centres
    rng = np.random.RandomState(random_state)
    n_tr = X_tr.shape[0]
    centre_idx = rng.choice(n_tr, size=min(n_centres, n_tr), replace=False)
    X_centres = X_tr_scaled[centre_idx]
    y_centres = y_tr[centre_idx]

    # Fit classic Gaussian kernel regression: computes the full
    # (n_centres x n_centres) kernel matrix internally
    model = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    model.fit(X_centres, y_centres)

    # --- Threshold selection via actual tolerance-based boundary metric ---
    # only done if no threshold was given
    if threshold is None:
        if utt_tr is None or tg_tr is None:
            raise ValueError(
                "utt_tr and tg_tr are required to select a threshold using "
                "evaluate_boundary_metrics."
            )
        # hold out a random 10% of the training utterances to pick the threshold on
        unique_tr_utts = np.unique(utt_tr)
        val_utts = rng.choice(unique_tr_utts,
                               size=max(1, int(len(unique_tr_utts) * val_frac)),
                               replace=False)
        val_mask = np.isin(utt_tr, val_utts)

        # score the held-out frames in batches to keep memory down
        val_scores = np.zeros(val_mask.sum(), dtype=np.float32)
        X_val = X_tr_scaled[val_mask]
        for start in range(0, X_val.shape[0], batch_size):
            end = min(start + batch_size, X_val.shape[0])
            val_scores[start:end] = model.predict(X_val[start:end])

        threshold = select_threshold_by_boundary_metric(
            val_scores, utt_tr[val_mask], tg_tr[val_mask], metric=metric
        )

    # --- Eval set (batched prediction to stay memory-safe) ---
    X_ev_scaled = scaler.transform(X_ev)
    n_ev = X_ev.shape[0]
    scores = np.zeros(n_ev, dtype=np.float32)
    for start in range(0, n_ev, batch_size):
        end = min(start + batch_size, n_ev)
        scores[start:end] = model.predict(X_ev_scaled[start:end])

    # 0/1 predictions from the chosen threshold
    y_pred = (scores > threshold).astype(int)
    # scores squeezed into 0-1 (min-max), saved as "probabilities"
    y_probs = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return y_pred, y_probs

def frames_to_zerosyl_segments(pred_frames):
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
    points = list(boundary_indices)
    if 0 not in points:
        points.insert(0, 0)
    if total_frames not in points:
        points.append(total_frames)
    points = sorted(list(set(points)))

    # each segment runs from one cut point to the next
    segments = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i+1]
        segments.append([start, end, 0])

    # fallback: treat the whole utterance as one segment
    if len(segments) == 0:
        segments.append([0, total_frames, 0])

    return torch.tensor(segments, dtype=torch.long)


def generate_fixed_interval_preds(total_frames, frame_rate=50.0, interval_ms=100):
    # control baseline: a boundary every interval_ms, ignoring the audio entirely
    # (100 ms at 50 frames/s = every 5th frame)
    stride_frames = int(round((interval_ms / 1000.0) * frame_rate))
    pred_frames = np.zeros(total_frames, dtype=int)
    if stride_frames > 0:
        pred_frames[::stride_frames] = 1
    return pred_frames


def run_experiment(variant_name, X_tr, y_tr, X_ev, y_ev, utt_ev, tg_ev,utt_tr=None, tg_tr=None, mode="probe", interval_ms=100, frame_rate=50.0, gamma=0.001, n_centres=500):
    # runs one experiment end to end and returns its scores.
    # mode="probe" = RBF probe, mode="fixed_interval" = the fixed 100 ms baseline
    print(f"\n==================================================")
    print(f" Running Experiment Variant: {variant_name}")
    print(f"==================================================")

    # folders for predicted segments, score curves, and matching TextGrids
    pred_dir = Path(f"pred_segments_{variant_name}").resolve()
    probs_dir = Path(f"pred_probs_{variant_name}").resolve()
    matched_tg_dir = Path(f"eval_textgrids_{variant_name}").resolve()

    # start from empty folders so old results don't get mixed in
    for d in [pred_dir, probs_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # train the probe once and predict for all eval frames
    if mode == "probe":
        y_pred, y_probs = fit_and_predict_classic_rbf(
            X_tr, y_tr, X_ev, utt_tr=utt_tr, tg_tr=tg_tr, gamma=gamma, n_centres=n_centres
        )

    # then save each utterance's predictions and copy its TextGrid
    unique_utts = np.unique(utt_ev)
    for utt in unique_utts:
        mask = (utt_ev == utt)
        utt_tg_path = Path(tg_ev[mask][0]).resolve()

        pt_file = pred_dir / f"{utt}.pt"
        prob_file = probs_dir / f"{utt}_probs.npy"
        tg_file = matched_tg_dir / f"{utt}.TextGrid"

        if mode == "probe":
            utt_preds = y_pred[mask]
            utt_probs = y_probs[mask]
        elif mode == "fixed_interval":
            total_utt_frames = np.sum(mask)
            utt_preds = generate_fixed_interval_preds(total_utt_frames, frame_rate=frame_rate, interval_ms=interval_ms)
            utt_probs = utt_preds.astype(np.float32)

        torch.save(frames_to_zerosyl_segments(utt_preds), pt_file)
        np.save(prob_file, utt_probs)

        if not tg_file.exists():
            shutil.copy(utt_tg_path, tg_file)

    # compare predicted boundaries to the true ones (50 ms tolerance)
    metrics = evaluate_boundary_metrics(
        segments_dir=pred_dir,
        textgrid_dir=matched_tg_dir,
        segments_pattern="*.pt",
        textgrid_pattern="*.TextGrid",
        frame_rate=frame_rate,
        constant_shift=0.0,
        tolerance=0.05
    )

    p, r, f1, os_val, rval, tp, tr, tf1 = metrics

    return {
        "P": p, "R": r, "F1": f1,
        "OS": os_val, "R-val": rval,
        "Token-P": tp, "Token-R": tr, "Token-F1": tf1
    }


def evaluate_normseg_dynamically(normseg_preds_dict, utt_ev, tg_ev, frame_rate=50.0):
    # scores NormSeg predictions (a dict of {utterance id: 0/1 frame predictions})
    # with the same evaluation used for the probes
    print(f"\n==================================================")
    print(f" Running Dynamic Evaluation: NormSeg")
    print(f"==================================================")

    pred_dir = Path("pred_segments_normseg").resolve()
    matched_tg_dir = Path("eval_textgrids_normseg").resolve()

    # start from empty folders so old results don't get mixed in
    for d in [pred_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    unique_utts = np.unique(utt_ev)
    missing_count = 0

    for utt in unique_utts:
        mask = (utt_ev == utt)
        utt_tg_path = Path(tg_ev[mask][0]).resolve()

        pt_file = pred_dir / f"{utt}.pt"
        tg_file = matched_tg_dir / f"{utt}.TextGrid"

        # only utterances that have predictions get saved and scored
        if utt in normseg_preds_dict:
            utt_preds = normseg_preds_dict[utt]
            torch.save(frames_to_zerosyl_segments(utt_preds), pt_file)
            if not tg_file.exists():
                shutil.copy(utt_tg_path, tg_file)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} utterances were missing from the NormSeg predictions dictionary.")

    metrics = evaluate_boundary_metrics(
        segments_dir=pred_dir,
        textgrid_dir=matched_tg_dir,
        segments_pattern="*.pt",
        textgrid_pattern="*.TextGrid",
        frame_rate=frame_rate,
        constant_shift=0.0,
        tolerance=0.05
    )

    p, r, f1, os_val, rval, tp, tr, tf1 = metrics

    return {
        "P": p, "R": r, "F1": f1,
        "OS": os_val, "R-val": rval,
        "Token-P": tp, "Token-R": tr, "Token-F1": tf1
    }


def print_results_table(results, title="RESULTS"):
    """Pretty-print a dict of {name: metrics_dict} as an aligned table."""
    print("\n" + "=" * 82)
    print(f"{title:^82}")
    print("=" * 82)
    header = f"{'Method / Variant':<24} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'F1 (%)':<9} | {'Token-F1 (%)':<13} | {'OS':<8} | {'R-val':<8}"
    print(header)
    print("-" * len(header))
    # one row per method; P, R and F1 are shown as percentages
    for name, res in results.items():
        print(
            f"{name:<24} | "
            f"{res['P']*100:>8.2f} | "
            f"{res['R']*100:>8.2f} | "
            f"{res['F1']*100:>8.2f} | "
            f"{res['Token-F1']*100:>12.2f} | "
            f"{res['OS']:>8.3f} | "
            f"{res['R-val']:>8.3f}"
        )
    print("=" * 82 + "\n")


def run_gamma_sweep(gamma_values, X_train, y_train, X_dev, y_dev, utt_dev, tg_dev,
                     utt_train, tg_train, n_centres=1000,
                     out_dir="gamma_sweep_outputs"):
    """
    Runs the RBF Gaussian kernel probe for each gamma value in gamma_values,
    collects the boundary-detection metrics for each, prints a comparison
    table, and saves a plot of every metric vs. gamma.

    Returns a dict {gamma_value: metrics_dict}.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # run the probe once per gamma value
    sweep_results = {}
    for gamma in gamma_values:
        variant_name = f"rbf_gamma_{gamma:g}"
        metrics = run_experiment(
            variant_name, X_train, y_train, X_dev, y_dev, utt_dev, tg_dev,
            utt_tr=utt_train, tg_tr=tg_train,
            mode="probe", gamma=gamma, n_centres=n_centres
        )
        sweep_results[gamma] = metrics

    # --- Comparison table, keyed by a readable "gamma=..." label ---
    # (print_results_table already includes the Token-F1 column)
    table_results = {f"gamma={g:g}": m for g, m in sweep_results.items()}
    print_results_table(table_results, title="RBF PROBE: GAMMA SWEEP COMPARISON")

    # --- Plot every metric vs gamma ---
    # pull each metric out as a list, in order of increasing gamma
    gammas_sorted = sorted(sweep_results.keys())
    precision = [sweep_results[g]["P"] * 100 for g in gammas_sorted]
    recall = [sweep_results[g]["R"] * 100 for g in gammas_sorted]
    f1 = [sweep_results[g]["F1"] * 100 for g in gammas_sorted]
    token_f1 = [sweep_results[g]["Token-F1"] * 100 for g in gammas_sorted]
    os_val = [sweep_results[g]["OS"] for g in gammas_sorted]
    rval = [sweep_results[g]["R-val"] for g in gammas_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: P / R / F1 / Token-F1 (%) on a log-x axis
    ax = axes[0]
    ax.plot(gammas_sorted, precision, marker="o", label="Precision")
    ax.plot(gammas_sorted, recall, marker="o", label="Recall")
    ax.plot(gammas_sorted, f1, marker="o", label="F1", linewidth=2)
    ax.plot(gammas_sorted, token_f1, marker="s", label="Token-F1", linewidth=2, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Score (%)")
    ax.set_title("Precision / Recall / F1 / Token-F1 vs gamma")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right panel: OS and R-val (different scale/behaviour than P/R/F1)
    ax = axes[1]
    ax.plot(gammas_sorted, os_val, marker="o", color="tab:red", label="OS")
    ax.plot(gammas_sorted, rval, marker="o", color="tab:purple", label="R-val")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("Value")
    ax.set_title("Over-Segmentation (OS) / R-value vs gamma")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("RBF Gaussian Kernel Probe: Metric Sensitivity to gamma", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # save the figure instead of showing it
    plot_path = out_dir / "gamma_sweep_metrics.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved gamma-sweep comparison plot to: {plot_path}")

    return sweep_results


def main():
    start = time.perf_counter()
    output_dir = "saved_features"
    
    # train features are memory-mapped (big), dev is loaded fully
    print("Loading extracted feature arrays...")
    X_train = np.load(os.path.join(output_dir, "X_train.npy"), mmap_mode="r")
    y_train = np.load(os.path.join(output_dir, "y_train.npy"), mmap_mode="r")
    utt_train = np.load(os.path.join(output_dir, "utt_train.npy"))
    tg_train = np.load(os.path.join(output_dir, "tg_train.npy"))
    X_dev = np.load(os.path.join(output_dir, "X_dev.npy"))
    y_dev = np.load(os.path.join(output_dir, "y_dev.npy"))
    utt_dev = np.load(os.path.join(output_dir, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(output_dir, "tg_dev.npy"))

    results = {}

    # 1. Fixed-Interval Baseline Control (100 ms)
    # no features or training needed, so those inputs are None
    results["Fixed Baseline (100ms)"] = run_experiment(
        "fixed_100ms", None, None, None, y_dev, utt_dev, tg_dev, mode="fixed_interval", interval_ms=100
    )

    # 2. NormSeg Baseline
    # training-free peak detection on the L2 norm (settings from the ZeroSyl paper)
    normseg_dict = generate_normseg_preds(
        X_dev, utt_dev,
        prominence_multiplier=0.45,
        window_size=3,
    )
    results["NormSeg (L2 Norm)"] = evaluate_normseg_dynamically(normseg_dict, utt_dev, tg_dev)

    
    # 3. RBF Gaussian Kernel Probe (Memory-Safe)
    results["RBF Gaussian Probe"] = run_experiment(
        "classic_rbf_kernel", X_train, y_train, X_dev, y_dev, utt_dev, tg_dev, utt_tr=utt_train, tg_tr=tg_train, 
        mode="probe", gamma=1e-5, n_centres=1000
    )
    elapsed = time.perf_counter() - start

    print(f"\nGaussian probe runtime: {elapsed:.2f} seconds")
    print(f"Gaussian probe runtime: {elapsed/60:.2f} minutes")

    # SUMMARY TABLE
    print_results_table(results, title="SYLLABLE BOUNDARY EVALUATION RESULTS (dev-clean)")

    # Gamma sweep for the RBF probe
    # (gamma sets how far each kernel centre's influence reaches: bigger = more local)
    gamma_values = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    gamma_sweep_results = run_gamma_sweep(
       gamma_values, X_train, y_train, X_dev, y_dev, utt_dev, tg_dev,
       utt_train, tg_train, n_centres=1000
    )


if __name__ == "__main__":
    main()
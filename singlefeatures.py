"""
Single-feature probes for the highest-weight raw features, compared with PC0.

Uses train_and_eval from the PCA script, so the probe settings are identical to
every other row (class_weight="balanced", max_iter=100, 0.5 threshold).

OS and R-value are stored in the results JSON files and read back from there
when building the table. If a result dict doesn't already contain them (e.g.
an older pca_probe_results_ctr.json), they are computed from P and R as a
fallback:
    HR = R,  OS = R/P - 1
    r1 = sqrt((1-HR)^2 + OS^2),  r2 = (-OS + HR - 1)/sqrt(2)
    R-value = 1 - (|r1| + |r2|)/2

Outputs:
  single_feature_results.json   Token-F1, OS, R-value etc. per feature
  report_figures/table_single_features.tex
  console: correlation of each feature with the raw L2 norm and with each other
"""
import json
import os

import numpy as np

# Adjust if your PCA script lives under a different module name.
from zerosylprobes.PCA import train_and_eval
from zerosyl.eval.boundaries import get_os, get_rvalue

# the raw WavLM dimensions picked out earlier (highest probe weights)
FEATURES = [869, 376, 4, 838, 590]
FULL_PROBE_TOKEN_F1 = 71.92          # your full-vector probe, for "Retained (%)"
PCA_RESULTS_JSON = "pca_probe_results_ctr.json"   # written by the PCA script
SINGLE_RESULTS_JSON = "single_feature_results.json"
OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Key names to look for in a saved result dict (first match wins).
OS_KEYS = ("OS", "os", "over_segmentation", "Over-segmentation")
RVAL_KEYS = ("R-value", "R_value", "Rval", "r_value", "R-Value", "rval")


def os_and_rval(p, r):
    """p, r as fractions (0-1). Returns (OS, R-value) as fractions, using the
    same get_os / get_rvalue that evaluate_boundary_metrics uses, so the
    numbers match the rest of the results."""
    return float(get_os(p, r)), float(get_rvalue(p, r))


def _first_key(d, keys):
    # return the value for the first key that exists and isn't None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def get_os_rval(res):
    """Read OS and R-value from a saved result dict; compute from P and R
    for whichever one is missing."""
    os_ = _first_key(res, OS_KEYS)
    rval = _first_key(res, RVAL_KEYS)
    # if either is missing, work it out from precision and recall
    if os_ is None or rval is None:
        os_calc, rval_calc = os_and_rval(res["P"], res["R"])
        os_ = os_calc if os_ is None else os_
        rval = rval_calc if rval is None else rval
    return float(os_), float(rval)


def main():
    d = "saved_features"
    # train is memory-mapped (big), dev is loaded fully
    X_tr = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
    y_tr = np.asarray(np.load(os.path.join(d, "y_train.npy")))
    X_dev = np.load(os.path.join(d, "X_dev.npy"))
    utt_dev = np.load(os.path.join(d, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(d, "tg_dev.npy"))

    results = {}

    #each feature on its own
    for f in FEATURES:
        res, _ = train_and_eval(f"feat{f}", [f], X_tr, y_tr, X_dev,
                                utt_dev, tg_dev)
        results[f"Feature {f} alone"] = res

    #all of them together: if this is barely better than the best single
    #    feature, the features are largely redundant with each other
    res, _ = train_and_eval("feats_combined", FEATURES, X_tr, y_tr, X_dev,
                            utt_dev, tg_dev)
    results[f"Features {', '.join(map(str, FEATURES))} together"] = res

    #make sure OS and R-value are stored in every result, then save
    for name, res in results.items():
        res["OS"], res["R-value"] = get_os_rval(res)

    with open(SINGLE_RESULTS_JSON, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"Saved {SINGLE_RESULTS_JSON}")

    #redundancy evidence: correlations on dev
    # stack the 5 features plus the L2 norm as columns, then correlate every pair
    norm = np.linalg.norm(X_dev.astype(np.float32), axis=1)
    cols = np.column_stack([X_dev[:, f] for f in FEATURES] + [norm])
    C = np.corrcoef(cols, rowvar=False)
    names = [f"f{f}" for f in FEATURES] + ["L2norm"]
    # print the correlation matrix
    print("\nCorrelation on dev (features and raw L2 norm):")
    print(" " * 8 + "".join(f"{n:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:<8}" + "".join(f"{C[i, j]:>9.2f}" for j in range(len(names))))
    # the last column of the matrix is each feature's correlation with the norm
    print("\nCorrelation with L2 norm:")
    for i, f in enumerate(FEATURES):
        print(f"  feature {f}: r = {C[i, -1]:+.3f}")

    #table - everything read back from the saved JSON files
    with open(SINGLE_RESULTS_JSON) as fh:
        saved = json.load(fh)

    # one row per result: name, number of dims, then scores as percentages
    rows = []
    for name, r in saved.items():
        dims = "5" if "together" in name else "1"
        os_, rval = get_os_rval(r)
        rows.append((name, dims, r["P"] * 100, r["R"] * 100,
                     r["Token-F1"] * 100, os_ * 100, rval * 100))

    # add the PC0 and top-5 PC rows from the PCA results, if that file exists
    if os.path.exists(PCA_RESULTS_JSON):
        with open(PCA_RESULTS_JSON) as fh:
            pca = json.load(fh)
        for key, label, dims in [("Top 1 PCs (variance)", "PC0 alone", "1"),
                                 ("Top 5 PCs (variance)", "Top 5 PCs", "5")]:
            if key in pca:
                r = pca[key]
                os_, rval = get_os_rval(r)
                rows.append((label, dims, r["P"] * 100, r["R"] * 100,
                             r["Token-F1"] * 100, os_ * 100, rval * 100))
    else:
        print(f"\n(note: {PCA_RESULTS_JSON} not found, PC rows omitted)")

    # build the LaTeX table: header first, then one line per row
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Boundary detection from individual raw features versus the "
         r"first principal component (dev-clean). OS is over-segmentation and "
         r"R-value is the combined hit-rate/OS score. Retained is relative to "
         r"the full vector probe.}",
         r"\label{tab:single_features}", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{lccccccc}", r"\hline",
         r"\textbf{Input} & \textbf{Dims} & \textbf{Prec (\%)} & \textbf{Rec (\%)} "
         r"& \textbf{Token-F1 (\%)} & \textbf{OS (\%)} & \textbf{R-val (\%)} "
         r"& \textbf{Retained (\%)} \\ \hline"]
    for label, dims, p, r, tf1, os_, rval in rows:
        # "Retained" = this row's Token-F1 as a share of the full probe's
        L.append(f"{label} & {dims} & {p:.2f} & {r:.2f} & {tf1:.2f} & "
                 f"{os_:+.1f} & {rval:.2f} & "
                 f"{tf1 / FULL_PROBE_TOKEN_F1 * 100:.1f} \\\\")
    L += [r"\hline", r"\end{tabular}%", r"}", r"\end{table}"]
    path = f"{OUT_DIR}/table_single_features.tex"
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
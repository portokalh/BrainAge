#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:56:34 2025

@author: alex
"""

#!/usr/bin/env python3
# compare_models_allinone.py
# End-to-end:
#  - Reads *train_results*.csv (for fold/rep train-only calibration) and *test_results*.csv (OOF eval)
#  - Applies leakage-free calibration per fold/rep (pred_on_age or bag_on_age)
#  - Aggregates to one row per subject (raw + corrected); saves per-model CSVs and a single wide CSV
#  - Plots: Pred vs True (RAW & corrected), BAG vs Age (RAW & corrected), labeled boxplots (BA/BAG/cBAG), and model comparison bars (MAE/RMSE/R² with 95% CIs)
#  - Stats (paired on common subjects, corrected): per-model CIs, pairwise bootstrap diffs, Friedman test + post-hoc Wilcoxon with Holm–Bonferroni

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
import matplotlib.pyplot as plt

'''
python compare_models_allinone.py \
  --agg mean \
  --calib pred_on_age \
  --boots 20000 \
  --plot \
  --err sd \
  --outdir "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/Figures/ModelCompare_AllTests"
  
usage: compare_models_allinone.py [--outdir PATH]
                                  [--agg {mean,median}]
                                  [--calib {pred_on_age,bag_on_age}]
                                  [--boots INT] [--seed INT]
                                  [--plot] [--err {sd,sem}]
Options (with defaults)
--outdir PATH
Where to write all CSVs and figures.
Default: /Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/Figures/ModelCompare_AllTests

--agg {mean, median}
How to aggregate multiple OOF predictions per subject (across folds/seeds).
Default: mean

--calib {pred_on_age, bag_on_age}
Fold/rep train-only bias correction used on the matching test split:

pred_on_age: fits y_pred = a + b * y_true on TRAIN, then invert on TEST.

bag_on_age: fits BAG = g0 + g1*(y_true - mean_age_train) on TRAIN, then de-bias on TEST.
Default: pred_on_age

--boots INT
Number of bootstrap iterations (over subjects) for 95% CIs and paired diffs.
Default: 20000

--seed INT
RNG seed for bootstrap resampling.
Default: 123

--plot
If provided, generates all figures (scatter, BAG vs age, boxplots, comparison bars).
Default: off

--err {sd, sem}
Error bars for per-subject scatter plots: standard deviation or standard error.
Default: sd

Notes

Model directories are set at the top of the script in MODEL_DIRS. Edit that dict if you want to add/remove paths.

File patterns (searched recursively):

TRAIN: **/*train_results*.csv

TEST: **/*test_results*.csv

Expected columns in each CSV: a true age (y_true/age/true_age) and a prediction (y_pred/yhat/pred_age/etc.), plus an optional subject ID column (several names supported).

'''

# =================== EDIT MODEL DIRS IF NEEDED ===================
MODEL_DIRS = {
    "FA": "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/TrainingResults/FA/column_md_results/",
    "MD": "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/TrainingResults/MD/column_md_results/",
    "Thickness": "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/TrainingResults/Thickness/thickness_model_results/",
    "MD+Thickness": "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/TrainingResults/MD+thickness/column_md_results/",
    "QSM": "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/TrainingResults/QSM/column_md_results/",
    "QSM+MD+Thickness": "/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/TrainingResults/QSM+MD+thickness/column_md_results/",
}
TRAIN_PATTERN = "**/*train_results*.csv"
TEST_PATTERN  = "**/*test_results*.csv"

# Plot colors
BLUE = "#0B3D91"   # deep blue
RED  = "red"       # dashed identity

SUBJECT_CANDIDATES = [
    "subject_id","subject","sub_id","participant_id","participant",
    "rid","RID","id","ID","ptid","PTID"
]
FOLD_CANDS = ["fold","fold_id","Fold","FoldID"]
REP_CANDS  = ["rep","rep_id","Rep","RepID"]

# =================== helpers: columns & ids ===================
def detect_cols(df):
    cols_l = {c.lower(): c for c in df.columns}
    y_true = cols_l.get("y_true") or cols_l.get("age") or cols_l.get("true_age")
    y_pred = cols_l.get("y_pred") or cols_l.get("yhat") or cols_l.get("y_hat") \
             or cols_l.get("pred") or cols_l.get("pred_age") or cols_l.get("predicted_age")
    if y_true is None or y_pred is None:
        raise ValueError(f"Missing y_true/y_pred. Got: {list(df.columns)}")
    y_true = next(c for c in df.columns if c.lower()==y_true)
    y_pred = next(c for c in df.columns if c.lower()==y_pred)
    subj = next((c for c in SUBJECT_CANDIDATES if c in df.columns), None)
    return y_true, y_pred, subj

_fold_rep_re = re.compile(r"fold[_\-]?(\d+).*?rep[_\-]?(\d+)", re.IGNORECASE)
_fold_re     = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)
_rep_re      = re.compile(r"rep[_\-]?(\d+)",  re.IGNORECASE)

def parse_fold_rep_from_filename(p: Path):
    s = p.name
    m = _fold_rep_re.search(s)
    if m:
        return int(m.group(1)), int(m.group(2))
    f = _fold_re.search(s)
    r = _rep_re.search(s)
    fold = int(f.group(1)) if f else None
    rep  = int(r.group(1)) if r else None
    return fold, rep

def detect_fold_rep(df, path: Path):
    fold_col = next((c for c in FOLD_CANDS if c in df.columns), None)
    rep_col  = next((c for c in REP_CANDS  if c in df.columns), None)
    if fold_col is not None and rep_col is not None:
        return pd.Series({"fold": df[fold_col].iloc[0], "rep": df[rep_col].iloc[0]})
    fold, rep = parse_fold_rep_from_filename(path)
    return pd.Series({"fold": fold, "rep": rep})

def collect_files(model_dir: Path, pattern: str):
    return sorted([p for p in model_dir.glob(pattern) if p.is_file() and p.suffix.lower()==".csv"])

# =================== metrics & bootstrap ===================
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_pred - y_true)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_pred - y_true)**2)))
def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

def slope_and_intercept(x, y):
    x = np.asarray(x); y = np.asarray(y)
    xc = x - x.mean(); denom = float(np.sum(xc*xc))
    if denom == 0: return np.nan, np.nan
    beta = float(np.sum(xc*(y - y.mean())) / denom)
    alpha = float(y.mean() - beta * x.mean())
    return beta, alpha

def bootstrap_subject_cis(y_true, y_pred, n_boot=10000, seed=123):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    MAEs  = np.empty(n_boot); RMSEs = np.empty(n_boot); R2s = np.empty(n_boot)
    Bpred = np.empty(n_boot); Alph = np.empty(n_boot);  Bbag = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; yp = y_pred[idx]; bag = yp - yt
        MAEs[b]  = mae(yt, yp)
        RMSEs[b] = rmse(yt, yp)
        R2s[b]   = r2_score(yt, yp)
        bp, ap   = slope_and_intercept(yt, yp)
        bb, _    = slope_and_intercept(yt, bag)
        Bpred[b] = bp; Alph[b] = ap; Bbag[b] = bb
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {
        "MAE_CI":  pct(MAEs),
        "RMSE_CI": pct(RMSEs),
        "R2_CI":   pct(R2s),
        "beta_pred_true_CI": pct(Bpred),
        "alpha_pred_true_CI": pct(Alph),
        "beta_bag_age_CI": pct(Bbag),
    }

def paired_bootstrap_diff(metric_fn, ytA, ypA, ytB, ypB, n_boot=20000, seed=123):
    """Bootstrap over subjects; Δ = metric(A) − metric(B). Returns Δ, CI_lo, CI_hi, p_two_sided."""
    rng = np.random.default_rng(seed)
    n = len(ytA)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric_fn(ytA[idx], ypA[idx]) - metric_fn(ytB[idx], ypB[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0))
    return float(np.mean(diffs)), float(lo), float(hi), float(min(1.0, p))

# =================== calibration ===================
def learn_calibration_params(train_df, method="pred_on_age"):
    yt = train_df["y_true"].values
    yp = train_df["y_pred"].values
    if method == "pred_on_age":
        # y_pred = a + b * y_true
        b, a = np.polyfit(yt, yp, deg=1)
        return {"a": float(a), "b": float(b), "mean_age_train": float(np.mean(yt))}
    else:
        # BAG = yp - yt; fit BAG = g0 + g1*(y_true - m)
        m = float(np.mean(yt))
        bag = yp - yt
        g1, g0 = np.polyfit(yt - m, bag, deg=1)
        return {"gamma0": float(g0), "gamma1": float(g1), "mean_age_train": m}

def apply_calibration(test_df, params, method="pred_on_age", eps=1e-6):
    df = test_df.copy()
    if method == "pred_on_age":
        a = params["a"]; b = params["b"]
        b_safe = b if abs(b) >= eps else (np.sign(b) * eps if b != 0 else eps)
        df["y_pred_corr"] = (df["y_pred"] - a) / b_safe
    else:
        g0 = params["gamma0"]; g1 = params["gamma1"]; m = params["mean_age_train"]
        bag = df["y_pred"] - df["y_true"]
        bag_corr = bag - (g0 + g1*(df["y_true"] - m))
        df["y_pred_corr"] = df["y_true"] + bag_corr
    return df

# =================== plotting ===================
def _yerr_from_sd_sem(sd, n, mode):
    return np.where(n > 1, sd/np.sqrt(n), 0.0) if mode == "sem" else sd

def _pred_true_scatter(x, y, yerr, title, ylabel, outpath, txt=None):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3,
                 color=BLUE, ecolor=BLUE, markerfacecolor=BLUE, markeredgecolor=BLUE,
                 alpha=0.95, elinewidth=1)
    lo = min(np.min(x), np.min(y)); hi = max(np.max(x), np.max(y))
    plt.plot([lo, hi], [lo, hi], linestyle='--', color=RED, linewidth=1.6)
    xc = x - x.mean(); denom = np.sum(xc*xc)
    if denom > 0:
        beta = np.sum(xc*(y - y.mean()))/denom
        alpha = y.mean() - beta * x.mean()
        xs = np.linspace(x.min(), x.max(), 100); ys = alpha + beta*xs
        plt.plot(xs, ys)
    plt.xlabel("True age (years)")
    plt.ylabel(ylabel)
    plt.title(title)
    if txt:
        plt.gcf().text(0.99, 0.01, txt, ha="right", va="bottom",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def _bag_age_scatter(x, y, yerr, title, ylabel, outpath, txt=None):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3,
                 color=BLUE, ecolor=BLUE, markerfacecolor=BLUE, markeredgecolor=BLUE,
                 alpha=0.95, elinewidth=1)
    xc = x - x.mean(); denom = np.sum(xc*xc)
    if denom > 0:
        beta = np.sum(xc*(y - y.mean()))/denom
        alpha = y.mean() - beta * x.mean()
        xs = np.linspace(x.min(), x.max(), 100); ys = alpha + beta*xs
        plt.plot(xs, ys)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("True age (years)")
    plt.ylabel(ylabel)
    plt.title(title)
    if txt:
        plt.gcf().text(0.99, 0.01, txt, ha="right", va="bottom",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def make_comparison_bars(summary_df, outdir, suffix="corrected_common"):
    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    # MAE
    plt.figure()
    plt.bar(x, summary_df["MAE"],
            yerr=[summary_df["MAE"]-summary_df["MAE_CI95_lo"],
                  summary_df["MAE_CI95_hi"]-summary_df["MAE"]],
            capsize=4)
    plt.xticks(x, models, rotation=20)
    plt.ylabel("MAE (years)")
    plt.title("Model comparison: MAE (95% CI) [paired common subjects]")
    plt.tight_layout()
    plt.savefig(outdir / f"compare_MAE_{suffix}.png", dpi=200); plt.close()
    # RMSE
    plt.figure()
    plt.bar(x, summary_df["RMSE"],
            yerr=[summary_df["RMSE"]-summary_df["RMSE_CI95_lo"],
                  summary_df["RMSE_CI95_hi"]-summary_df["RMSE"]],
            capsize=4)
    plt.xticks(x, models, rotation=20)
    plt.ylabel("RMSE (years)")
    plt.title("Model comparison: RMSE (95% CI) [paired common subjects]")
    plt.tight_layout()
    plt.savefig(outdir / f"compare_RMSE_{suffix}.png", dpi=200); plt.close()
    # R²
    plt.figure()
    plt.bar(x, summary_df["R2"])
    plt.xticks(x, models, rotation=20)
    plt.ylabel("R²")
    plt.title("Model comparison: R² [paired common subjects]")
    plt.tight_layout()
    plt.savefig(outdir / f"compare_R2_{suffix}.png", dpi=200); plt.close()

# ---- Boxplots (explicit labels: BA, BAG, cBAG) ----
def boxplot_per_model_three_tests(model_name, tab, outdir, agg, err_mode):
    ba_ae_corr = np.abs(tab["pred_age_corr"].values - tab["true_age"].values)
    bag_raw    = tab["bag_raw"].values
    cbag_corr  = tab["bag_corr"].values
    data = [ba_ae_corr, bag_raw, cbag_corr]
    labels = ["BA (AE, corrected)", "BAG (raw)", "cBAG (corrected)"]
    plt.figure()
    #plt.boxplot(data, labels=labels, showfliers=True)
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.ylabel("Years")
    plt.title(f"{model_name}: BA/BAG/cBAG distributions ({agg}, {err_mode.upper()} bars used elsewhere)")
    plt.tight_layout()
    plt.savefig(outdir / f"{model_name}_BOX_BA_BAG_cBAG_{agg}.png", dpi=200); plt.close()

def boxplot_across_models_one_test(subj_tables, outdir, test_name, agg):
    vals, labels = [], []
    if test_name == "BA_AE_corr":
        title = "BA (AE, corrected) by model"; ylab = "Absolute error (years)"; fname = f"BOX_BA_AE_corrected_by_model_{agg}.png"
    elif test_name == "BAG_raw":
        title = "BAG (raw) by model";            ylab = "BAG (years)";           fname = f"BOX_BAG_raw_by_model_{agg}.png"
    elif test_name == "cBAG_corr":
        title = "cBAG (corrected) by model";      ylab = "BAG (years)";           fname = f"BOX_cBAG_corrected_by_model_{agg}.png"
    else:
        raise ValueError("Unknown test_name")
    for m in sorted(subj_tables.keys()):
        t = subj_tables[m]
        arr = (np.abs(t["pred_age_corr"].values - t["true_age"].values)
               if test_name=="BA_AE_corr" else
               (t["bag_raw"].values if test_name=="BAG_raw" else t["bag_corr"].values))
        vals.append(arr); labels.append(m)
    plt.figure()
    #plt.boxplot(vals, labels=labels, showfliers=True)
    plt.boxplot(vals, tick_labels=labels, showfliers=True)
    plt.ylabel(ylab); plt.title(title); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(outdir / fname, dpi=200); plt.close()

# =================== Holm–Bonferroni ===================
def holm_bonferroni(pvals, labels):
    pvals = np.asarray(pvals, dtype=float); labels = np.asarray(labels)
    m = len(pvals); order = np.argsort(pvals)
    adj = np.empty(m, dtype=float); running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        adj_p = (m - rank + 1) * pvals[idx]
        running_max = max(running_max, adj_p)
        adj[idx] = min(1.0, running_max)
    return pd.DataFrame({"comparison": labels, "p_raw": pvals, "p_holm": adj})

# =================== main ===================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/Users/alex/AlexBadea_MyPapers/HanwenLin_Columns092425/results/Figures/ModelCompare_AllTests",
                    help="Output directory")
    ap.add_argument("--agg", choices=["mean","median"], default="mean",
                    help="Aggregate per subject")
    ap.add_argument("--calib", choices=["pred_on_age","bag_on_age"], default="pred_on_age",
                    help="Fold-wise calibration learned on TRAIN and applied to TEST")
    ap.add_argument("--boots", type=int, default=20000, help="Bootstrap iterations (subjects)")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--plot", action="store_true", help="Create plots (raw + corrected)")
    ap.add_argument("--err", choices=["sd","sem"], default="sd", help="Error bars: SD or SEM")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)

    subj_tables = {}  # per-model subject-level (raw+corrected)

    # ---------- load, learn calibration per fold/rep, apply to TEST ----------
    for model, d in MODEL_DIRS.items():
        mdir = Path(d).expanduser()
        train_files = collect_files(mdir, TRAIN_PATTERN)
        test_files  = collect_files(mdir, TEST_PATTERN)
        if not test_files:
            print(f"[WARN] No *test_results*.csv under {mdir} for {model}; skipping.")
            continue
        if not train_files:
            print(f"[WARN] No *train_results*.csv under {mdir} for {model}. Using UNcorrected preds for those files.")

        calib_map = {}
        for tf in train_files:
            df = pd.read_csv(tf)
            ytc, ypc, _ = detect_cols(df)
            df = df[[ytc, ypc]].rename(columns={ytc:"y_true", ypc:"y_pred"}).copy()
            fr = detect_fold_rep(df, tf)
            key = (int(fr["fold"]) if pd.notna(fr["fold"]) else None,
                   int(fr["rep"])  if pd.notna(fr["rep"])  else None)
            calib_map[key] = learn_calibration_params(df, method=args.calib)

        rows = []
        for f in test_files:
            df = pd.read_csv(f)
            ytc, ypc, subj = detect_cols(df)
            take = df[[ytc, ypc]].rename(columns={ytc:"y_true", ypc:"y_pred"}).copy()
            fr = detect_fold_rep(df, f)
            key = (int(fr["fold"]) if pd.notna(fr["fold"]) else None,
                   int(fr["rep"])  if pd.notna(fr["rep"])  else None)
            take["subject_id"] = (df[subj].astype(str).values if subj is not None
                                  else np.arange(take.shape[0]).astype(str))
            if key in calib_map:
                take = apply_calibration(take, calib_map[key], method=args.calib)
            else:
                take["y_pred_corr"] = take["y_pred"]
                print(f"[WARN] No calibration for fold/rep {key} in {model}; keeping raw preds for this file.")
            take["fold"] = key[0]; take["rep"] = key[1]
            take["bag_row_raw"]  = take["y_pred"]      - take["y_true"]
            take["bag_row_corr"] = take["y_pred_corr"] - take["y_true"]
            rows.append(take)

        all_test = pd.concat(rows, ignore_index=True)

        # ---------- SUBJECT-LEVEL aggregation (raw + corrected) ----------
        g = all_test.groupby("subject_id", as_index=False).agg(
            true_age=("y_true", args.agg),
            pred_age_raw=("y_pred", args.agg),
            pred_age_corr=("y_pred_corr", args.agg),
            n_preds=("y_pred","size"),
            pred_sd_raw=("y_pred","std"),
            pred_sd_corr=("y_pred_corr","std"),
            bag_raw=("bag_row_raw", args.agg),
            bag_sd_raw=("bag_row_raw","std"),
            bag_corr=("bag_row_corr", args.agg),
            bag_sd_corr=("bag_row_corr","std"),
        )
        for c in ["pred_sd_raw","pred_sd_corr","bag_sd_raw","bag_sd_corr"]:
            g[c] = g[c].fillna(0.0)

        # checks
        g["bag_raw_check"]  = g["pred_age_raw"]  - g["true_age"]
        g["bag_corr_check"] = g["pred_age_corr"] - g["true_age"]
        if np.allclose(g["bag_raw"], g["bag_raw_check"], equal_nan=True):
            g = g.drop(columns=["bag_raw_check"])
        if np.allclose(g["bag_corr"], g["bag_corr_check"], equal_nan=True):
            g = g.drop(columns=["bag_corr_check"])

        subj_tables[model] = g.copy()
        g.to_csv(outdir / f"{model}_subject_level_{args.agg}_raw_and_corrected_{args.calib}.csv", index=False)

        # ---------- PLOTS per model ----------
        if args.plot:
            x = g["true_age"].values
            # corrected Pred vs True
            y = g["pred_age_corr"].values
            yerr = _yerr_from_sd_sem(g["pred_sd_corr"].values, g["n_preds"].values, args.err)
            cis = bootstrap_subject_cis(x, y, n_boot=min(args.boots, 10000), seed=args.seed)
            txt = (
                f"MAE:  {mae(x,y):.2f}  [95% CI {cis['MAE_CI'][0]:.2f}, {cis['MAE_CI'][1]:.2f}] yrs\n"
                f"RMSE: {rmse(x,y):.2f} [95% CI {cis['RMSE_CI'][0]:.2f}, {cis['RMSE_CI'][1]:.2f}] yrs\n"
                f"R²:   {r2_score(x,y):.3f} [95% CI {cis['R2_CI'][0]:.3f}, {cis['R2_CI'][1]:.3f}]"
            )
            _pred_true_scatter(
                x, y, yerr,
                title=f"{model}: Pred vs True (corrected, {args.agg}, {args.err.upper()} bars)",
                ylabel=f"Predicted age (corrected) — {args.agg} ± {args.err.upper()}",
                outpath=outdir / f"{model}_pred_vs_true_corrected_{args.agg}_{args.err}.png",
                txt=txt
            )
            # corrected BAG vs Age
            y = g["bag_corr"].values
            yerr = _yerr_from_sd_sem(g["bag_sd_corr"].values, g["n_preds"].values, args.err)
            _bag_age_scatter(
                x, y, yerr,
                title=f"{model}: BAG vs Age (corrected, {args.agg}, {args.err.upper()} bars)",
                ylabel=f"BAG (corrected, years) — {args.agg} ± {args.err.upper()}",
                outpath=outdir / f"{model}_bag_vs_age_corrected_{args.agg}_{args.err}.png"
            )
            # RAW Pred vs True
            y = g["pred_age_raw"].values
            yerr = _yerr_from_sd_sem(g["pred_sd_raw"].values, g["n_preds"].values, args.err)
            _pred_true_scatter(
                x, y, yerr,
                title=f"{model}: Pred vs True (RAW, {args.agg}, {args.err.upper()} bars)",
                ylabel=f"Predicted age (raw) — {args.agg} ± {args.err.upper()}",
                outpath=outdir / f"{model}_pred_vs_true_RAW_{args.agg}_{args.err}.png"
            )
            # RAW BAG vs Age
            y = g["bag_raw"].values
            yerr = _yerr_from_sd_sem(g["bag_sd_raw"].values, g["n_preds"].values, args.err)
            _bag_age_scatter(
                x, y, yerr,
                title=f"{model}: BAG vs Age (RAW, {args.agg}, {args.err.upper()} bars)",
                ylabel=f"BAG (raw, years) — {args.agg} ± {args.err.upper()}",
                outpath=outdir / f"{model}_bag_vs_age_RAW_{args.agg}_{args.err}.png"
            )
            # Per-model labeled boxplot: BA/BAG/cBAG
            boxplot_per_model_three_tests(model, g, outdir, args.agg, args.err)

    if not subj_tables:
        raise SystemExit("No models processed. Check directories/patterns.")

    # ---------- Across-model boxplots ----------
    if args.plot:
        boxplot_across_models_one_test(subj_tables, outdir, "BA_AE_corr", args.agg)
        boxplot_across_models_one_test(subj_tables, outdir, "BAG_raw",    args.agg)
        boxplot_across_models_one_test(subj_tables, outdir, "cBAG_corr",  args.agg)

    # ---------- paired intersection across models (corrected) ----------
    common = set.intersection(*[set(t["subject_id"]) for t in subj_tables.values()])
    if len(common) == 0:
        print("[WARN] No common subjects across models; skipping paired stats/bars and WIDE table.")
        return

    # build per-model, paired (corrected) summary metrics + CIs
    summaries = []
    model_order = sorted(subj_tables.keys())
    aligned = {}
    for m in model_order:
        t = subj_tables[m]
        tt = t[t["subject_id"].isin(common)].sort_values("subject_id").reset_index(drop=True)
        yt = tt["true_age"].values
        yp = tt["pred_age_corr"].values  # corrected
        aligned[m] = (yt, yp)
        cis = bootstrap_subject_cis(yt, yp, n_boot=min(args.boots, 20000), seed=args.seed)
        summaries.append({
            "model": m, "N_subjects_common": len(yt),
            "MAE": mae(yt, yp), "MAE_CI95_lo": cis["MAE_CI"][0], "MAE_CI95_hi": cis["MAE_CI"][1],
            "RMSE": rmse(yt, yp), "RMSE_CI95_lo": cis["RMSE_CI"][0], "RMSE_CI95_hi": cis["RMSE_CI"][1],
            "R2": r2_score(yt, yp), "R2_CI95_lo": cis["R2_CI"][0], "R2_CI95_hi": cis["R2_CI"][1],
        })
    summary_df = pd.DataFrame(summaries).sort_values("model")
    summary_df.to_csv(outdir / f"model_metrics_corrected_common_{args.agg}_{args.calib}.csv", index=False)

    # comparison bars (paired on common subjects)
    if args.plot:
        make_comparison_bars(summary_df, outdir, suffix=f"{args.calib}_{args.agg}")

    # ---------- Friedman + Wilcoxon (AE) on corrected preds ----------
    # wide AE table
    wide = None
    for m in model_order:
        yt, yp = aligned[m]
        ae = np.abs(yp - yt)
        col = pd.DataFrame({"subject_idx": np.arange(len(ae)), f"AE_{m}": ae})
        wide = col if wide is None else wide.merge(col, on="subject_idx", how="inner")
    ae_cols = [c for c in wide.columns if c.startswith("AE_")]
    ae_arrays = [wide[c].values for c in ae_cols]
    fried_stat, fried_p = stats.friedmanchisquare(*ae_arrays)
    pd.DataFrame([{"statistic": fried_stat, "p_value": fried_p,
                   "k_models": len(ae_cols), "N_subjects": len(wide)}])\
      .to_csv(outdir / f"friedman_AE_corrected_{args.agg}_{args.calib}.csv", index=False)

    # Post-hoc Wilcoxon (paired) with Holm-Bonferroni
    wilco_rows, pvals, labels = [], [], []
    for A, B in combinations(model_order, 2):
        x = wide[f"AE_{A}"].values
        y = wide[f"AE_{B}"].values
        comp = f"{A} vs {B}"
        try:
            stat, p = stats.wilcoxon(x, y, zero_method="pratt",
                                     alternative="two-sided", correction=False, mode="approx")
        except Exception:
            stat, p = np.nan, 1.0
        wilco_rows.append({"model_A":A,"model_B":B,"comparison":comp,"statistic":stat,"p_raw":p})
        pvals.append(p); labels.append(comp)
    wilco_df = pd.DataFrame(wilco_rows)
    if len(pvals) > 0:
        holm_df = holm_bonferroni(pvals, labels).drop(columns=["p_raw"])
        wilco_df = wilco_df.merge(holm_df, on="comparison", how="left")
    wilco_df = wilco_df[["model_A","model_B","comparison","statistic","p_raw","p_holm"]]
    wilco_df.to_csv(outdir / f"wilcoxon_AE_holm_corrected_{args.agg}_{args.calib}.csv", index=False)

    # ---------- Pairwise bootstrap differences (corrected) ----------
    def r2_metric(yt, yp): return r2_score(yt, yp)
    def beta_pred_metric(yt, yp): return slope_and_intercept(yt, yp)[0]
    def beta_bag_metric(yt, yp):  return slope_and_intercept(yt, yp-yt)[0]

    pair_rows = []
    for A, B in combinations(model_order, 2):
        ytA, ypA = aligned[A]; ytB, ypB = aligned[B]
        d_mae, lo_mae, hi_mae, p_mae = paired_bootstrap_diff(mae,  ytA, ypA, ytB, ypB, n_boot=args.boots, seed=args.seed)
        d_rmse,lo_rmse,hi_rmse,p_rmse = paired_bootstrap_diff(rmse, ytA, ypA, ytB, ypB, n_boot=args.boots, seed=args.seed)
        d_r2, lo_r2, hi_r2, p_r2     = paired_bootstrap_diff(r2_metric, ytA, ypA, ytB, ypB, n_boot=args.boots, seed=args.seed)
        d_bp, lo_bp, hi_bp, p_bp     = paired_bootstrap_diff(beta_pred_metric, ytA, ypA, ytB, ypB, n_boot=args.boots, seed=args.seed)
        d_bb, lo_bb, hi_bb, p_bb     = paired_bootstrap_diff(beta_bag_metric,  ytA, ypA, ytB, ypB, n_boot=args.boots, seed=args.seed)
        pair_rows += [
            {"metric":"MAE", "model_A":A,"model_B":B,"diff_A_minus_B":d_mae,"CI95_lo":lo_mae,"CI95_hi":hi_mae,"p":p_mae},
            {"metric":"RMSE","model_A":A,"model_B":B,"diff_A_minus_B":d_rmse,"CI95_lo":lo_rmse,"CI95_hi":hi_rmse,"p":p_rmse},
            {"metric":"R2",  "model_A":A,"model_B":B,"diff_A_minus_B":d_r2, "CI95_lo":lo_r2, "CI95_hi":hi_r2, "p":p_r2},
            {"metric":"beta_pred_true","model_A":A,"model_B":B,"diff_A_minus_B":d_bp,"CI95_lo":lo_bp,"CI95_hi":hi_bp,"p":p_bp},
            {"metric":"beta_bag_age", "model_A":A,"model_B":B,"diff_A_minus_B":d_bb,"CI95_lo":lo_bb,"CI95_hi":hi_bb,"p":p_bb},
        ]
    pd.DataFrame(pair_rows).to_csv(outdir / f"pairwise_bootstrap_diffs_corrected_{args.agg}_{args.calib}.csv", index=False)

    # ---------- WIDE table (common subjects, all modalities) ----------
    base = None
    for m in model_order:
        t = subj_tables[m]
        tt = t[t["subject_id"].isin(common)].sort_values("subject_id").reset_index(drop=True)
        colset = tt[["subject_id","true_age",
                     "pred_age_raw","pred_age_corr",
                     "bag_raw","bag_corr",
                     "n_preds","pred_sd_raw","pred_sd_corr",
                     "bag_sd_raw","bag_sd_corr"]].copy()
        ren = {
            "pred_age_raw":  f"{m}__pred_raw",
            "pred_age_corr": f"{m}__pred_corr",
            "bag_raw":       f"{m}__bag_raw",
            "bag_corr":      f"{m}__bag_corr",
            "n_preds":       f"{m}__n_preds",
            "pred_sd_raw":   f"{m}__pred_sd_raw",
            "pred_sd_corr":  f"{m}__pred_sd_corr",
            "bag_sd_raw":    f"{m}__bag_sd_raw",
            "bag_sd_corr":   f"{m}__bag_sd_corr",
        }
        colset = colset.rename(columns=ren)
    
        if base is None:
            # keep true_age from the first model
            base = colset
        else:
            # drop true_age to avoid duplicate columns on merge
            colset_nota = colset.drop(columns=["true_age"], errors="ignore")
            base = base.merge(colset_nota, on="subject_id", how="inner")
    
    # ensure single true_age exists
    if "true_age" not in base.columns:
        # extremely unlikely with logic above; fallback: derive from any residual *_true_age
        true_cols = [c for c in base.columns if c.endswith("__true_age")]
        if true_cols:
            base["true_age"] = base[true_cols].mean(axis=1)
            base = base.drop(columns=true_cols)
    
    fixed = ["subject_id","true_age"]
    model_cols = [c for c in base.columns if c not in fixed]
    wide = base[fixed + sorted(model_cols)]
    wide.to_csv(outdir / f"subject_level_ALL_MODALITIES_wide_common_{args.agg}_{args.calib}.csv", index=False)


    print(f"\nDone. Outputs saved to: {outdir}")

if __name__ == "__main__":
    main()

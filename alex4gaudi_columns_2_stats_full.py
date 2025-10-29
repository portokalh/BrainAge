#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GAUDI FULL ANALYSIS PIPELINE (DATE-STAMPED + CATEGORICAL TRAIT FIX)

Performs:
- UMAP embeddings with clusters (colored by trait)
- Cluster–trait significance tests (age, APOE, sex, risk)
- Violin/boxplots per trait
- Regression of latent embeddings vs age
- Summary heatmaps: –log10(p), |β|, and R²
- Saves all outputs in Graph_data/results_<YYYY-MM-DD>/

Author: Alexandra Badea (with ChatGPT)
Date: 2025-10-29
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import umap
from datetime import date

# ---------------- CONFIG ----------------
BASE = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation/Graph_data"
TODAY = date.today().strftime("%Y-%m-%d")
OUTROOT = os.path.join(BASE, f"results_{TODAY}")
os.makedirs(OUTROOT, exist_ok=True)

MODALITIES = ["MD", "QSM", "joint"]
TRAITS = ["age", "APOE", "sex", "risk_for_ad"]
SEED = 42
sns.set(style="whitegrid", context="talk")
np.random.seed(SEED)


# ---------------- HELPER FUNCTIONS ----------------
def load_data(mod):
    npy_path = os.path.join(BASE, f"latent_epochs_{mod}", f"latent_final_{mod}.npy")
    csv_path = os.path.join(BASE, f"latent_epochs_{mod}", f"{mod}_subject_clusters.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE, f"latent_epochs_{mod}", f"{mod.capitalize()}_subject_clusters.csv")
    emb = np.load(npy_path)
    meta = pd.read_csv(csv_path)
    meta.columns = [c.strip() for c in meta.columns]
    return emb, meta


def safe_stats(df, cluster_col, trait):
    """Automatically select Chi2 for categorical or ANOVA/Kruskal for continuous."""
    if df[trait].dtype == object or df[trait].nunique() <= 6:
        tab = pd.crosstab(df[cluster_col], df[trait])
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        return {"test": "Chi2", "stat": chi2, "p": p}
    else:
        groups = [v.dropna() for _, v in df.groupby(cluster_col)[trait]]
        if all(len(g) > 2 for g in groups):
            f, p = stats.f_oneway(*groups)
            return {"test": "ANOVA", "stat": f, "p": p}
        else:
            h, p = stats.kruskal(*groups)
            return {"test": "Kruskal", "stat": h, "p": p}


def regression_age(df, zcol="z1"):
    X = sm.add_constant(df["age"])
    y = df[zcol]
    model = sm.OLS(y, X, missing="drop").fit()
    beta = model.params["age"]
    r2 = model.rsquared
    pval = model.pvalues["age"]
    return beta, r2, pval


def umap_plot(emb, trait_values, title, outpath, categorical=False):
    """UMAP scatter colored by trait."""
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=SEED)
    proj = reducer.fit_transform(emb)

    plt.figure(figsize=(6, 5))
    if categorical:
        unique_vals = pd.unique(trait_values)
        palette = sns.color_palette("Set2", len(unique_vals))
        for i, val in enumerate(unique_vals):
            mask = np.array(trait_values) == val
            plt.scatter(proj[mask, 0], proj[mask, 1],
                        s=70, alpha=0.8, label=str(val), color=palette[i])
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title=title)
    else:
        sc = plt.scatter(proj[:, 0], proj[:, 1],
                         c=trait_values, cmap="viridis", s=60, alpha=0.8)
        plt.colorbar(sc, label=title)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------- MAIN LOOP ----------------
all_cluster_stats = []
regression_results = []

for mod in MODALITIES:
    print(f"\n[{mod.upper()}] Processing ...")
    emb, meta = load_data(mod)
    df = pd.concat(
        [meta.reset_index(drop=True),
         pd.DataFrame(emb, columns=[f"z{i+1}" for i in range(emb.shape[1])])],
        axis=1
    )

    cluster_col = [c for c in df.columns if "cluster" in c.lower()][0]
    outdir = os.path.join(OUTROOT, f"{mod}")
    os.makedirs(outdir, exist_ok=True)

    # ---------- Cluster-trait visualization ----------
    for trait in TRAITS:
        if trait not in df.columns:
            print(f"⚠️ Missing {trait} in {mod}")
            continue

        trait_vals = df[trait]
        is_cat = (trait in ["APOE", "sex"]) or (trait_vals.dtype == object) or (df[trait].nunique() < 6)

        umap_file = os.path.join(outdir, f"{mod}_embedding_by_{trait}_with_clusters.png")
        umap_plot(emb, np.array(trait_vals), f"{mod.upper()} Embedding colored by {trait}",
                  umap_file, categorical=is_cat)
        print(f"Saved → {umap_file}")

        # Violin/boxplot
        statres = safe_stats(df, cluster_col, trait)
        p = statres["p"]
        plt.figure(figsize=(5, 4))
        sns.violinplot(data=df, x=cluster_col, y=trait, inner="box", palette="Set2")
        sns.stripplot(data=df, x=cluster_col, y=trait, color="k", alpha=0.5)
        plt.title(f"{mod.upper()} {trait} vs cluster (p={p:.4f})")
        plt.tight_layout()
        outbox = os.path.join(outdir, f"{mod}_{trait}_cluster_boxplot.png")
        plt.savefig(outbox, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved → {outbox} (p={p:.4f})")

        statres.update({"modality": mod, "trait": trait})
        all_cluster_stats.append(statres)

    # ---------- Regression vs age ----------
    if "age" in df.columns:
        for zcol in [c for c in df.columns if c.startswith("z")]:
            beta, r2, p = regression_age(df, zcol)
            regression_results.append({
                "modality": mod, "dimension": zcol,
                "beta": beta, "abs_beta": abs(beta),
                "R2": r2, "p": p
            })

# ---------------- SAVE CSVs ----------------
cluster_df = pd.DataFrame(all_cluster_stats)
reg_df = pd.DataFrame(regression_results)

cluster_csv = os.path.join(OUTROOT, "cluster_trait_associations_full.csv")
reg_csv = os.path.join(OUTROOT, "embedding_vs_age_full.csv")
cluster_df.to_csv(cluster_csv, index=False)
reg_df.to_csv(reg_csv, index=False)
print(f"\n✅ Saved CSVs → {OUTROOT}/")

# ---------------- HEATMAPS ----------------
# 1. –log₁₀(p)
heatmap_df = cluster_df.pivot(index="trait", columns="modality", values="p")
heatmap_df = -np.log10(heatmap_df)
plt.figure(figsize=(6, 4))
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="mako", cbar_kws={"label": "–log₁₀(p)"})
plt.title("Cluster–Trait Association Strength (–log₁₀ p)")
plt.tight_layout()
plt.savefig(os.path.join(OUTROOT, "cluster_trait_p_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# 2. |β|
beta_matrix = reg_df.pivot(index="dimension", columns="modality", values="abs_beta")
plt.figure(figsize=(7, 6))
sns.heatmap(beta_matrix, cmap="rocket_r", cbar_kws={"label": "|β| (effect of age)"})
plt.title("Absolute β per Latent Dimension (Age Regression)")
plt.tight_layout()
plt.savefig(os.path.join(OUTROOT, "latent_age_absbeta_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# 3. R²
r2_matrix = reg_df.pivot(index="dimension", columns="modality", values="R2")
plt.figure(figsize=(7, 6))
sns.heatmap(r2_matrix, cmap="viridis", cbar_kws={"label": "R² (variance explained by age)"})
plt.title("R² per Latent Dimension (Age Regression)")
plt.tight_layout()
plt.savefig(os.path.join(OUTROOT, "latent_age_R2_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ All figures and CSVs saved in {OUTROOT}")

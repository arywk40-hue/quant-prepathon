"""Generate Part 4 Feature Forensics figures.

Reads frozen analytical artifacts only.  Does not modify any result files.

Figures produced:
  1. feature_ic_distribution.png   — distribution of feature-level IC values
  2. family_ic_summary.png         — per-family IC summary
  3. top_bottom_features.png       — strongest and weakest features
  4. eligibility_funnel.png        — count reduction through frozen screen
  5. per_family_redundancy.png     — family-level correlation heatmap
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "part4"

AGGREGATE_IC = RESULTS / "predictive" / "aggregate_ic.csv"
FROZEN_SET = RESULTS / "ml" / "features" / "frozen_feature_set.csv"
PAIRWISE = RESULTS / "redundancy" / "pairwise_redundancy.csv"
PCA_SUMMARY = RESULTS / "redundancy" / "pca_summary.csv"
FAMILY_SUMMARY = RESULTS / "features" / "family_summary.csv"

FAMILY_ORDER = ["PB", "BB", "PV", "V", "VB"]
SCREEN_COLS = {
    "pearson_fdr_reject",
    "pearson_pct_same_sign",
    "mean_pearson_ic",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"ERROR: required artifact not found: {label} ({path})")
    return pd.read_csv(path)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    dest = FIGURES / name
    fig.savefig(dest, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {dest.name}")


# ── figure 1: feature IC distribution ────────────────────────────────────────
def _fig1_ic_distribution(agg: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Histogram of mean_pearson_ic, coloured by eligibility and significance."""
    merged = agg.merge(
        frozen[["feature", "horizon_if_applicable", "eligible_for_ml", "development_screen_status"]],
        left_on=["feature", "horizon_seconds"],
        right_on=["feature", "horizon_if_applicable"],
        how="left",
        suffixes=("", "_frozen"),
    )
    merged["eligible_for_ml"] = merged["eligible_for_ml"].fillna(False)

    eligible = merged.loc[merged["eligible_for_ml"], "mean_pearson_ic"]
    excluded = merged.loc[~merged["eligible_for_ml"], "mean_pearson_ic"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-0.15, 0.15, 80)

    ax.hist(excluded, bins=bins, alpha=0.55, color="#4c72b0", label=f"Excluded (n={len(excluded):,})", edgecolor="white", linewidth=0.3)
    ax.hist(eligible, bins=bins, alpha=0.80, color="#dd8452", label=f"Eligible (n={len(eligible):,})", edgecolor="white", linewidth=0.3)

    ax.axvline(0.05, color="black", ls="--", lw=0.8, label="|IC| = 0.05 threshold")
    ax.axvline(-0.05, color="black", ls="--", lw=0.8)

    ax.set_xlabel("Mean Pearson IC (feature–horizon pair)")
    ax.set_ylabel("Count")
    ax.set_title("Feature IC distribution — development period, 70 available days\n"
                 "Frozen screen: |IC| >= 0.05, pct_same_sign >= 70%, FDR-rejected")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "feature_ic_distribution.png")


# ── figure 2: family IC summary ──────────────────────────────────────────────
def _fig2_family_summary(agg: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Per-family bar chart: candidate count, eligible count, mean |IC|."""
    merged = agg.merge(
        frozen[["feature", "horizon_if_applicable", "eligible_for_ml"]],
        left_on=["feature", "horizon_seconds"],
        right_on=["feature", "horizon_if_applicable"],
        how="left",
        suffixes=("", "_frozen"),
    )
    merged["eligible_for_ml"] = merged["eligible_for_ml"].fillna(False)

    families = []
    for fam in FAMILY_ORDER:
        fam_mask = merged["feature"].str.startswith(fam)
        fam_data = merged.loc[fam_mask]
        n_candidates = fam_data["feature"].nunique()
        n_eligible = fam_data.loc[fam_data["eligible_for_ml"], "feature"].nunique()
        mean_ic = fam_data["mean_pearson_ic"].abs().mean()
        std_ic = fam_data["mean_pearson_ic"].std()
        pct_positive = (fam_data["mean_pearson_ic"] > 0).mean() * 100
        families.append({
            "family": fam,
            "candidates": n_candidates,
            "eligible": n_eligible,
            "mean_abs_ic": mean_ic,
            "std_ic": std_ic,
            "pct_positive_ic": pct_positive,
        })
    fam_df = pd.DataFrame(families)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: candidate vs eligible counts
    x = np.arange(len(fam_df))
    w = 0.35
    axes[0].bar(x - w / 2, fam_df["candidates"], w, color="#4c72b0", label="Candidates")
    axes[0].bar(x + w / 2, fam_df["eligible"], w, color="#dd8452", label="Eligible")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fam_df["family"])
    axes[0].set_ylabel("Feature count")
    axes[0].set_title("Candidate vs eligible features")
    axes[0].legend(frameon=False, fontsize=9)

    # Panel B: mean |IC| with error bars
    axes[1].bar(x, fam_df["mean_abs_ic"], 0.5, color="#55a868", yerr=fam_df["std_ic"],
                capsize=3, error_kw={"lw": 0.7})
    axes[1].axhline(0.05, color="black", ls="--", lw=0.8, label="Eligibility threshold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fam_df["family"])
    axes[1].set_ylabel("Mean |Pearson IC|")
    axes[1].set_title("Mean IC magnitude by family")
    axes[1].legend(frameon=False, fontsize=9)

    # Panel C: sign consistency
    axes[2].bar(x, fam_df["pct_positive_ic"], 0.5, color="#c44e52")
    axes[2].axhline(50, color="black", ls="--", lw=0.8, label="50% (random)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(fam_df["family"])
    axes[2].set_ylabel("% features with positive IC")
    axes[2].set_title("IC sign consistency")
    axes[2].set_ylim(0, 100)
    axes[2].legend(frameon=False, fontsize=9)

    fig.suptitle("Feature family IC summary — development period, 70 days", y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "family_ic_summary.png")


# ── figure 3: top / bottom features ──────────────────────────────────────────
def _fig3_top_bottom(agg: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Horizontal bars for strongest and weakest features at 300 s horizon."""
    h300 = agg.loc[agg["horizon_seconds"] == 300].copy()
    h300 = h300.merge(
        frozen[["feature", "horizon_if_applicable", "eligible_for_ml", "pearson_pct_same_sign"]],
        left_on=["feature", "horizon_seconds"],
        right_on=["feature", "horizon_if_applicable"],
        how="left",
        suffixes=("", "_frozen"),
    )
    h300["eligible_for_ml"] = h300["eligible_for_ml"].fillna(False)
    h300["abs_ic"] = h300["mean_pearson_ic"].abs()

    top = h300.nlargest(10, "abs_ic").copy()
    bot = h300.nsmallest(10, "abs_ic").copy()

    for df in (top, bot):
        df["label"] = df.apply(
            lambda r: f"{r['feature']}  (h={int(r['horizon_seconds'])}s, "
                      f"{'ELIGIBLE' if r['eligible_for_ml'] else 'excluded'})",
            axis=1,
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Top features
    colors_top = ["#dd8452" if e else "#4c72b0" for e in top["eligible_for_ml"]]
    ax1.barh(range(len(top)), top["mean_pearson_ic"].values, color=colors_top, edgecolor="white", linewidth=0.3)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(top["label"].values, fontsize=8)
    ax1.set_xlabel("Mean Pearson IC (300 s horizon)")
    ax1.set_title("Top 10 features by |IC|")
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", lw=0.5)

    # Bottom features
    colors_bot = ["#dd8452" if e else "#4c72b0" for e in bot["eligible_for_ml"]]
    ax2.barh(range(len(bot)), bot["mean_pearson_ic"].values, color=colors_bot, edgecolor="white", linewidth=0.3)
    ax2.set_yticks(range(len(bot)))
    ax2.set_yticklabels(bot["label"].values, fontsize=8)
    ax2.set_xlabel("Mean Pearson IC (300 s horizon)")
    ax2.set_title("Bottom 10 features by |IC|")
    ax2.invert_yaxis()
    ax2.axvline(0, color="black", lw=0.5)

    fig.suptitle("Top and bottom features — 300 s horizon, development period", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, "top_bottom_features.png")


# ── figure 4: eligibility funnel ─────────────────────────────────────────────
def _fig4_eligibility_funnel(frozen: pd.DataFrame) -> None:
    """Show how candidate count reduces through each frozen screen criterion."""
    total = len(frozen)

    # Criterion 1: pearson_fdr_reject
    step1 = frozen["pearson_fdr_reject"].astype(str).str.lower().eq("true").sum()

    # Criterion 2: pct_same_sign >= 0.70
    step2 = frozen.loc[
        frozen["pearson_fdr_reject"].astype(str).str.lower().eq("true")
        & (frozen["pearson_pct_same_sign"] >= 0.70)
    ].shape[0]

    # Criterion 3: abs(mean_pearson_ic) >= 0.05
    step3 = frozen["eligible_for_ml"].sum()

    stages = [
        "All feature–horizon pairs",
        "FDR-rejected",
        "pct_same_sign >= 0.70",
        "|mean_IC| >= 0.05\n(all three criteria)",
    ]
    counts = [total, step1, step2, step3]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4c72b0", "#55a868", "#c44e52", "#dd8452"]
    bars = ax.barh(range(len(stages)), counts, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=10)
    ax.set_xlabel("Feature–horizon pairs remaining")
    ax.set_title("Eligibility funnel — frozen Part 4 screen\n"
                 "Development period, 70 available days")
    ax.invert_yaxis()

    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = count / total * 100
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({pct:.1f}%)", va="center", fontsize=9)

    ax.set_xlim(0, total * 1.15)
    fig.tight_layout()
    _save(fig, "eligibility_funnel.png")


# ── figure 5: per-family redundancy heatmap ──────────────────────────────────
def _fig5_redundancy_heatmap(pairwise: pd.DataFrame, pca_summary: pd.DataFrame) -> None:
    """Family-level mean absolute correlation matrix from pairwise redundancy."""
    # parse feature families from feature_i and feature_j
    def _family(name: str) -> str:
        for fam in FAMILY_ORDER:
            if name.startswith(fam):
                return fam
        return "OTHER"

    pw = pairwise.copy()
    pw["fam_i"] = pw["feature_i"].map(_family)
    pw["fam_j"] = pw["feature_j"].map(_family)
    pw = pw.loc[pw["fam_i"] != "OTHER"]
    pw = pw.loc[pw["fam_j"] != "OTHER"]

    # build family-level matrix
    families = FAMILY_ORDER
    n = len(families)
    pearson_mat = np.full((n, n), np.nan)
    spearman_mat = np.full((n, n), np.nan)

    for i, fi in enumerate(families):
        for j, fj in enumerate(families):
            mask = (pw["fam_i"] == fi) & (pw["fam_j"] == fj)
            if i == j:
                mask = mask | ((pw["fam_i"] == fi) & (pw["fam_j"] == fj))
            subset = pw.loc[mask]
            if len(subset) > 0:
                pearson_mat[i, j] = subset["mean_abs_pearson"].mean()
                spearman_mat[i, j] = subset["mean_abs_spearman"].mean()
            if i == j:
                pearson_mat[i, j] = 1.0
                spearman_mat[i, j] = 1.0

    pooled = pca_summary.loc[pca_summary["pca_type"] == "pooled_incremental"]
    pca_note = ""
    if len(pooled) > 0:
        row = pooled.iloc[0]
        pca_note = (f"  Pooled PCA: {int(row['components_50pct'])} components for 50% var, "
                    f"{int(row['components_80pct'])} for 80%, "
                    f"{int(row['components_90pct'])} for 90%  "
                    f"(first component: {row['variance_first_component']:.1%})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Pearson heatmap
    im1 = ax1.imshow(pearson_mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(families, fontsize=10)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(families, fontsize=10)
    for i in range(n):
        for j in range(n):
            val = pearson_mat[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax1.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)
    ax1.set_title("Mean |Pearson correlation|")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Spearman heatmap
    im2 = ax2.imshow(spearman_mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(families, fontsize=10)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(families, fontsize=10)
    for i in range(n):
        for j in range(n):
            val = spearman_mat[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax2.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)
    ax2.set_title("Mean |Spearman correlation|")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    sup = ("Per-family redundancy — development period, 70 available days\n"
           f"238,395 feature pairs; median |Pearson| = 0.132, |Spearman| = 0.158"
           f"  {pca_note}")
    fig.suptitle(sup, y=1.05, fontsize=10)
    fig.tight_layout()
    _save(fig, "per_family_redundancy.png")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading Part 4 artifacts ...")
    agg = _load(AGGREGATE_IC, "aggregate_ic.csv")
    frozen = _load(FROZEN_SET, "frozen_feature_set.csv")
    pairwise = _load(PAIRWISE, "pairwise_redundancy.csv")
    pca = _load(PCA_SUMMARY, "pca_summary.csv")

    print(f"  aggregate_ic.csv:       {len(agg):,} rows")
    print(f"  frozen_feature_set.csv: {len(frozen):,} rows")
    print(f"  pairwise_redundancy.csv: {len(pairwise):,} rows")
    print(f"  pca_summary.csv:        {len(pca):,} rows")

    # validate expected columns
    missing_agg = SCREEN_COLS - set(agg.columns)
    if missing_agg:
        sys.exit(f"ERROR: aggregate_ic.csv missing columns: {sorted(missing_agg)}")
    missing_frozen = {"eligible_for_ml", "pearson_fdr_reject", "pearson_pct_same_sign", "mean_pearson_ic"} - set(frozen.columns)
    if missing_frozen:
        sys.exit(f"ERROR: frozen_feature_set.csv missing columns: {sorted(missing_frozen)}")

    print("\nGenerating Part 4 figures ...")
    _fig1_ic_distribution(agg, frozen)
    _fig2_family_summary(agg, frozen)
    _fig3_top_bottom(agg, frozen)
    _fig4_eligibility_funnel(frozen)
    _fig5_redundancy_heatmap(pairwise, pca)
    print("\nDone. All Part 4 figures written to figures/part4/")


if __name__ == "__main__":
    main()

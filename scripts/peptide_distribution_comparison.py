"""Compare peptide distributions between real and synthetic data.

Identifies peptide columns with a zero fraction > 40%, excludes near-constant
columns (>99% zeros), and plots the best- and worst-matching distributions
ranked by Kolmogorov-Smirnov statistic.

Usage:
    python scripts/peptide_distribution_comparison.py \
        --real resources/merged_peptide_and_clinical.csv \
        --synth resources/merged_synth.csv \
        --output evaluation_output/peptide_comparison \
        --n-show 5
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp
from tqdm import tqdm


# ── helpers ──────────────────────────────────────────────────────────────────


def get_peptide_columns(df: pl.DataFrame) -> list[str]:
    pattern = re.compile("peptide", re.IGNORECASE)
    return [col for col in df.columns if pattern.search(col)]


def zero_fraction(series: pl.Series) -> float:
    return float((series == 0).sum() / len(series))


# ── main logic ───────────────────────────────────────────────────────────────


def run(
    real_path: str,
    synth_path: str,
    output_dir: str,
    n_show: int = 5,
    min_zero_frac: float = 0.40,
    max_zero_frac: float = 0.99,
    log_scale: bool = False,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # load data and drop rows with NaNs
    real_df = pl.read_csv(real_path).drop_nulls()
    synth_df = pl.read_csv(synth_path).drop_nulls()

    # peptide columns present in both datasets
    pep_cols = sorted(
        set(get_peptide_columns(real_df)) & set(get_peptide_columns(synth_df))
    )
    print(f"Total peptide columns in both datasets: {len(pep_cols)}")

    # compute zero fractions on real data
    zf_real = {
        col: zero_fraction(real_df[col])
        for col in tqdm(pep_cols, desc="Computing zero fractions")
    }

    # filter: keep columns with 40% < zero_frac <= 99%
    selected = [
        col for col in pep_cols if min_zero_frac < zf_real[col] <= max_zero_frac
    ]
    print(
        f"After filtering (zero fraction in ({min_zero_frac:.0%}, {max_zero_frac:.0%}]): "
        f"{len(selected)} columns"
    )
    if not selected:
        print("No columns to analyse — exiting.")
        return

    # compute KS statistic for each selected column
    records = []
    for col in tqdm(selected, desc="Computing KS statistics per peptide"):
        real_v = real_df[col].to_numpy().astype(float)
        synth_v = synth_df[col].to_numpy().astype(float)
        ks_stat, ks_pvalue = ks_2samp(real_v, synth_v)
        records.append(
            {
                "column": col,
                "real_zero_pct": zf_real[col],
                "synth_zero_pct": zero_fraction(synth_df[col]),
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
            }
        )

    results_df = pl.DataFrame(records).sort("ks_statistic", descending=True)
    results_df.write_csv(str(out / "peptide_distribution_comparison.csv"))
    print(f"Full results saved to {out / 'peptide_distribution_comparison.csv'}")

    best = results_df.tail(n_show).reverse()  # lowest KS stat = best match
    worst = results_df.head(n_show)  # highest KS stat = worst match

    print(f"\n{'='*60}")
    print(f"TOP {n_show} BEST-MATCHING (lowest KS statistic):")
    print(f"{'='*60}")
    for row in best.iter_rows(named=True):
        print(
            f"  {row['column']:30s}  KS={row['ks_statistic']:.4f}  p={row['ks_pvalue']:.2e}  "
            f"zero%: real={row['real_zero_pct']:.1%}  synth={row['synth_zero_pct']:.1%}"
        )

    print(f"\n{'='*60}")
    print(f"TOP {n_show} WORST-MATCHING (highest KS statistic):")
    print(f"{'='*60}")
    for row in worst.iter_rows(named=True):
        print(
            f"  {row['column']:30s}  KS={row['ks_statistic']:.4f}  p={row['ks_pvalue']:.2e}  "
            f"zero%: real={row['real_zero_pct']:.1%}  synth={row['synth_zero_pct']:.1%}"
        )

    # ── plotting ─────────────────────────────────────────────────────────
    def _plot_group(group_df: pl.DataFrame, title: str, filename: str):
        n = len(group_df)
        n_cols = 2
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(9 * n_cols, 6 * n_rows), squeeze=False
        )
        for i, row in enumerate(
            tqdm(list(group_df.iter_rows(named=True)), desc=f"Plotting {title[:30]}")
        ):
            col = row["column"]
            ax = axes[i // n_cols][i % n_cols]
            real_v = real_df[col].to_numpy().astype(float)
            synth_v = synth_df[col].to_numpy().astype(float)
            lo = min(real_v.min(), synth_v.min())
            hi = max(real_v.max(), synth_v.max())
            bins = (
                np.linspace(lo, hi, 50) if lo < hi else np.array([lo - 0.5, lo + 0.5])
            )

            ax.hist(
                real_v,
                bins=bins,
                density=True,
                alpha=0.55,
                color="darkorange",
                label="Real",
                edgecolor="white",
                linewidth=0.4,
            )
            ax.hist(
                synth_v,
                bins=bins,
                density=True,
                alpha=0.55,
                color="steelblue",
                label="Synthetic",
                edgecolor="white",
                linewidth=0.4,
            )

            ax.set_title(
                f"{col}\n"
                f"KS={row['ks_statistic']:.3f}  p={row['ks_pvalue']:.2e}\n"
                f"zero%: real={row['real_zero_pct']:.1%}  synth={row['synth_zero_pct']:.1%}",
                fontsize=11,
            )
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            if log_scale:
                ax.set_yscale("log")
            ax.legend(fontsize=10)
            ax.tick_params(labelsize=10)

        # hide any unused axes in the last row
        for j in range(n, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].set_visible(False)

        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(str(out / filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out / filename}")

    _plot_group(
        best,
        "Best-Matching Peptide Distributions (Real vs Synthetic)",
        "best_matching.png",
    )
    _plot_group(
        worst,
        "Worst-Matching Peptide Distributions (Real vs Synthetic)",
        "worst_matching.png",
    )

    # ── combined overview figure ─────────────────────────────────────────
    all_ks = results_df["ks_statistic"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(all_ks, bins=60, color="#aec7e8", edgecolor="white", linewidth=0.3)
    for row in worst.iter_rows(named=True):
        ax.axvline(row["ks_statistic"], color="#d62728", ls="--", lw=0.8, alpha=0.7)
    for row in best.iter_rows(named=True):
        ax.axvline(row["ks_statistic"], color="#2ca02c", ls="--", lw=0.8, alpha=0.7)
    ax.plot([], [], color="#d62728", ls="--", lw=1, label=f"Worst {n_show}")
    ax.plot([], [], color="#2ca02c", ls="--", lw=1, label=f"Best {n_show}")
    ax.legend(fontsize=9)
    ax.set_xlabel("KS statistic", fontsize=10)
    ax.set_ylabel("Number of peptides", fontsize=10)
    ax.set_title(
        f"KS Statistic Distribution — {len(selected)} peptides "
        f"(zero fraction in ({min_zero_frac:.0%}, {max_zero_frac:.0%}])",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(out / "divergence_overview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out / 'divergence_overview.png'}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare peptide distributions between real and synthetic data."
    )
    parser.add_argument("--real", required=True, help="Path to real data CSV")
    parser.add_argument("--synth", required=True, help="Path to synthetic data CSV")
    parser.add_argument(
        "--output",
        default="evaluation_output/peptide_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=5,
        help="Number of best/worst peptides to plot (default: 5)",
    )
    parser.add_argument(
        "--min-zero-frac",
        type=float,
        default=0.40,
        help="Minimum zero fraction to include (default: 0.40)",
    )
    parser.add_argument(
        "--max-zero-frac",
        type=float,
        default=0.90,
        help="Maximum zero fraction to include — excludes near-constant columns (default: 0.99)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for the y-axis of density histograms",
    )
    args = parser.parse_args()

    run(
        real_path=args.real,
        synth_path=args.synth,
        output_dir=args.output,
        n_show=args.n_show,
        min_zero_frac=args.min_zero_frac,
        max_zero_frac=args.max_zero_frac,
        log_scale=args.log_scale,
    )

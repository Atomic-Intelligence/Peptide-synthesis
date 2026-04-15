"""Quantify the effect of eigenvalue clipping on the copula correlation matrix.

Loads real data, estimates the correlation matrix using Spearman (or Kendall)
rank-based methods (same as the copula pipeline), applies corr_clipped, and
reports how much the correction distorts the correlation structure.

Metrics reported
----------------
- Eigenvalue spectrum before / after clipping
- Number and magnitude of negative eigenvalues
- Frobenius norm of the correction: ||C_clipped - C_raw||_F
- Relative Frobenius norm: ||C_clipped - C_raw||_F / ||C_raw||_F
- Mean / max absolute change in off-diagonal entries
- Implicit shrinkage factor: mean(|C_clipped_ij| / |C_raw_ij|) for |C_raw_ij| > 0.05
- Per-event breakdown (hf, ckd, no_event) to show dependence on sample size

Usage:
    python scripts/quantify_correlation_clipping.py \
        --real resources/merged_peptide_and_clinical.csv \
        --output evaluation_output/clipping_analysis \
        --methods spearman kendall \
        --event-column event_type \
        --event-types hf ckd no_event
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, kendalltau
from statsmodels.stats.correlation_tools import corr_clipped
from tqdm import tqdm


# ── dataclass for results ────────────────────────────────────────────────────


@dataclass
class ClippingAnalysis:
    label: str
    method: str
    n_samples: int
    n_features: int

    # eigenvalue stats
    eigenvalues_raw: np.ndarray
    eigenvalues_clipped: np.ndarray
    n_negative_eigs: int
    most_negative_eig: float
    sum_negative_eigs: float

    # correction magnitude
    frobenius_correction: float
    relative_frobenius: float
    mean_abs_offdiag_change: float
    max_abs_offdiag_change: float

    # shrinkage
    mean_shrinkage_ratio: float  # mean(|clipped_ij|/|raw_ij|) for |raw_ij|>0.05

    # Marchenko-Pastur law
    mp_lambda_plus: float        # upper bulk edge λ_+ = (1 + sqrt(p/n))²
    mp_lambda_minus: float       # lower bulk edge λ_- = max(0, (1 - sqrt(p/n)))²
    n_signal_eigs: int           # eigenvalues above λ_+  (real factors)
    n_noise_eigs: int            # eigenvalues inside MP bulk (noise)
    signal_eigenvalues: np.ndarray  # the signal eigenvalues (> λ_+)

    def summary_dict(self) -> dict:
        return {
            "label": self.label,
            "method": self.method,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_negative_eigenvalues": self.n_negative_eigs,
            "most_negative_eigenvalue": round(self.most_negative_eig, 6),
            "sum_negative_eigenvalues": round(self.sum_negative_eigs, 6),
            "frobenius_correction": round(self.frobenius_correction, 6),
            "relative_frobenius_pct": round(self.relative_frobenius * 100, 2),
            "mean_abs_offdiag_change": round(self.mean_abs_offdiag_change, 6),
            "max_abs_offdiag_change": round(self.max_abs_offdiag_change, 6),
            "mean_shrinkage_ratio": round(self.mean_shrinkage_ratio, 4),
            "implicit_shrinkage_pct": round((1 - self.mean_shrinkage_ratio) * 100, 2),
            # Marchenko-Pastur
            "mp_lambda_plus": round(self.mp_lambda_plus, 6),
            "mp_lambda_minus": round(self.mp_lambda_minus, 6),
            "n_signal_eigenvalues": self.n_signal_eigs,
            "n_noise_eigenvalues": self.n_noise_eigs,
            "signal_fraction_pct": round(self.n_signal_eigs / self.n_features * 100, 2),
        }


# ── correlation estimation (mirrors gaussian_copula_basic.py) ────────────────


def estimate_correlation_spearman(X: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 0, X)
    rho = np.corrcoef(ranks, rowvar=False)
    return 2.0 * np.sin(np.pi / 6.0 * rho)


def estimate_correlation_kendall(X: np.ndarray) -> np.ndarray:
    n_cols = X.shape[1]
    tau = np.eye(n_cols)
    for i in tqdm(range(n_cols), desc="Kendall tau rows"):
        for j in range(i + 1, n_cols):
            t, _ = kendalltau(X[:, i], X[:, j])
            val = t if np.isfinite(t) else 0.0
            tau[i, j] = tau[j, i] = val
    return np.sin(np.pi / 2.0 * tau)


ESTIMATORS = {
    "spearman": estimate_correlation_spearman,
    "kendall": estimate_correlation_kendall,
}


# ── analysis ─────────────────────────────────────────────────────────────────


def analyse_clipping(
    X: np.ndarray, method: str, label: str, threshold: float = 1e-5
) -> ClippingAnalysis:
    estimator = ESTIMATORS[method]
    C_raw = estimator(X)

    # eigendecomposition of raw matrix
    eigs_raw = np.linalg.eigvalsh(C_raw)
    negative_mask = eigs_raw < 0
    n_neg = int(negative_mask.sum())
    most_neg = float(eigs_raw.min()) if n_neg > 0 else 0.0
    sum_neg = float(eigs_raw[negative_mask].sum()) if n_neg > 0 else 0.0

    # apply clipping (same as pipeline)
    C_clipped = corr_clipped(C_raw, threshold=threshold)
    eigs_clipped = np.linalg.eigvalsh(C_clipped)

    # correction magnitude
    diff = C_clipped - C_raw
    mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
    off_diag_diff = diff[mask]

    frob_correction = float(np.linalg.norm(diff, "fro"))
    frob_raw = float(np.linalg.norm(C_raw, "fro"))
    rel_frob = frob_correction / frob_raw if frob_raw > 0 else 0.0

    mean_abs_change = float(np.abs(off_diag_diff).mean())
    max_abs_change = float(np.abs(off_diag_diff).max())

    # shrinkage ratio for non-trivial correlations
    raw_offdiag = np.abs(C_raw[mask])
    clipped_offdiag = np.abs(C_clipped[mask])
    significant = raw_offdiag > 0.05
    if significant.sum() > 0:
        ratios = clipped_offdiag[significant] / raw_offdiag[significant]
        mean_shrinkage = float(ratios.mean())
    else:
        mean_shrinkage = 1.0

    # ── Marchenko-Pastur law ─────────────────────────────────────────────────
    # For a correlation matrix estimated from n samples of p variables,
    # eigenvalues of a *purely random* matrix are bounded by:
    #   λ_± = (1 ± sqrt(q))²  where q = p / n
    # Eigenvalues above λ_+ are "signal"; inside [λ_-, λ_+] are noise.
    n, p = X.shape
    q = p / n                                      # aspect ratio
    mp_lambda_plus  = (1.0 + np.sqrt(q)) ** 2
    mp_lambda_minus = max(0.0, (1.0 - np.sqrt(q)) ** 2)

    signal_mask   = eigs_raw > mp_lambda_plus
    n_signal      = int(signal_mask.sum())
    n_noise       = int((~signal_mask).sum())
    signal_eigs   = np.sort(eigs_raw[signal_mask])[::-1]  # descending

    return ClippingAnalysis(
        label=label,
        method=method,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        eigenvalues_raw=eigs_raw,
        eigenvalues_clipped=eigs_clipped,
        n_negative_eigs=n_neg,
        most_negative_eig=most_neg,
        sum_negative_eigs=sum_neg,
        frobenius_correction=frob_correction,
        relative_frobenius=rel_frob,
        mean_abs_offdiag_change=mean_abs_change,
        max_abs_offdiag_change=max_abs_change,
        mean_shrinkage_ratio=mean_shrinkage,
        mp_lambda_plus=float(mp_lambda_plus),
        mp_lambda_minus=float(mp_lambda_minus),
        n_signal_eigs=n_signal,
        n_noise_eigs=n_noise,
        signal_eigenvalues=signal_eigs,
    )


# ── plotting ─────────────────────────────────────────────────────────────────


def _mp_density(lam: np.ndarray, q: float) -> np.ndarray:
    """Marchenko-Pastur probability density for aspect ratio q = p/n.

    Only valid for q <= 1 (well-conditioned regime); for q > 1 the bulk
    still exists but a point mass sits at 0 for the (q-1)/q fraction of
    zero eigenvalues — we still return the density over the bulk range.
    """
    lam_plus  = (1.0 + np.sqrt(q)) ** 2
    lam_minus = max(0.0, (1.0 - np.sqrt(q)) ** 2)
    density = np.zeros_like(lam, dtype=float)
    bulk = (lam >= lam_minus) & (lam <= lam_plus) & (lam > 0)
    l = lam[bulk]
    density[bulk] = (
        np.sqrt((lam_plus - l) * (l - lam_minus)) / (2.0 * np.pi * q * l)
    )
    return density


def plot_eigenvalue_spectrum(
    results: list[ClippingAnalysis], save_dir: Path
) -> None:
    """Eigenvalue spectrum before/after clipping with Marchenko-Pastur overlay."""
    for res in results:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        q = res.n_features / res.n_samples
        eigs_sorted = np.sort(res.eigenvalues_raw)[::-1]   # descending

        # ── Panel 1: full spectrum + MP boundary ────────────────────────────
        ax = axes[0]
        idx = np.arange(len(res.eigenvalues_raw))
        ax.plot(idx, eigs_sorted, "o-", ms=2, label="Raw", alpha=0.8)
        ax.plot(
            idx, np.sort(res.eigenvalues_clipped)[::-1],
            "s-", ms=2, label="Clipped", alpha=0.8,
        )
        ax.axhline(res.mp_lambda_plus, color="green", ls="--", lw=1.2,
                   label=f"MP λ+ = {res.mp_lambda_plus:.3f}")
        ax.axhline(res.mp_lambda_minus, color="orange", ls="--", lw=1.2,
                   label=f"MP λ- = {res.mp_lambda_minus:.3f}")
        ax.axhline(0, color="k", ls=":", lw=0.8)
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_xlabel("Eigenvalue index (sorted descending)")
        ax.set_ylabel("Eigenvalue (symlog)")
        ax.set_title(
            f"{res.label} — {res.method}\n"
            f"n={res.n_samples}, d={res.n_features},  q=d/n={q:.2f}\n"
            f"Signal: {res.n_signal_eigs} / {res.n_features} "
            f"({res.n_signal_eigs/res.n_features*100:.1f}%)"
        )
        ax.legend(fontsize=8)

        # ── Panel 2: MP density histogram (bulk eigenvalues only) ───────────
        ax = axes[1]
        bulk_mask = (res.eigenvalues_raw >= res.mp_lambda_minus) & \
                    (res.eigenvalues_raw <= res.mp_lambda_plus)
        bulk_eigs = res.eigenvalues_raw[bulk_mask]
        if len(bulk_eigs) > 0:
            ax.hist(bulk_eigs, bins=50, density=True, alpha=0.6,
                    color="steelblue", label="Bulk eigenvalues (histogram)")
            lam_grid = np.linspace(
                max(res.mp_lambda_minus, 1e-6), res.mp_lambda_plus, 500
            )
            ax.plot(lam_grid, _mp_density(lam_grid, q), "r-", lw=2,
                    label="MP density")
        ax.axvline(res.mp_lambda_plus, color="green", ls="--", lw=1.2,
                   label=f"λ+ = {res.mp_lambda_plus:.3f}")
        ax.set_xlabel("Eigenvalue (log scale)")
        ax.set_ylabel("Density")
        if len(bulk_eigs) > 0 and res.mp_lambda_minus > 0:
            ax.set_xscale("log")
        ax.set_title(
            f"Bulk eigenvalue distribution vs Marchenko-Pastur\n"
            f"({len(bulk_eigs)} bulk eigenvalues)"
        )
        ax.legend(fontsize=8)

        # ── Panel 3: zoom into the negative / near-zero tail ────────────────
        ax = axes[2]
        cutoff = max(20, res.n_negative_eigs + 5)
        raw_sorted_asc  = np.sort(res.eigenvalues_raw)[:cutoff]
        clip_sorted_asc = np.sort(res.eigenvalues_clipped)[:cutoff]
        idx2 = np.arange(len(raw_sorted_asc))
        ax.bar(idx2 - 0.15, raw_sorted_asc,  width=0.3, label="Raw",     alpha=0.7)
        ax.bar(idx2 + 0.15, clip_sorted_asc, width=0.3, label="Clipped", alpha=0.7)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("Smallest eigenvalues (index)")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(
            f"Zoom: {res.n_negative_eigs} negative eigenvalues\n"
            f"Most negative: {res.most_negative_eig:.4f}"
        )
        ax.legend()

        plt.tight_layout()
        fname = f"eigenvalues_{res.label}_{res.method}.png"
        fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_summary_bar(results: list[ClippingAnalysis], save_dir: Path) -> None:
    """Bar chart comparing key metrics across all analyses."""
    labels = [f"{r.label}\n({r.method})" for r in results]
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── helper: place label above short bars, inside tall bars ─────────────────
    # Avoids the label floating into the subplot title area when bars reach
    # near the top of the axis.  "tall" = bar height > 60% of the axis max.
    def _label_bar(ax, bar, text, y_max):
        h = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        if h > 0.60 * y_max:
            # Place inside the bar, near the top, with contrasting white text
            ax.text(cx, h - 0.04 * y_max, text,
                    ha="center", va="top", fontsize=8,
                    color="white", fontweight="bold")
        else:
            # Place above the bar with a small relative offset
            ax.text(cx, h + 0.03 * y_max, text,
                    ha="center", va="bottom", fontsize=8)

    # Implicit shrinkage %
    ax = axes[0, 0]
    shrinkage_pct = [(1 - r.mean_shrinkage_ratio) * 100 for r in results]
    bars = ax.bar(x, shrinkage_pct, width, color="steelblue")
    y_max = max(shrinkage_pct, default=1.0) * 1.30 or 1.0
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Implicit shrinkage (%)")
    ax.set_title("Off-diagonal shrinkage due to clipping")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    for bar, val in zip(bars, shrinkage_pct):
        _label_bar(ax, bar, f"{val:.1f}%", y_max)

    # Relative Frobenius %
    ax = axes[0, 1]
    rel_frob = [r.relative_frobenius * 100 for r in results]
    bars = ax.bar(x, rel_frob, width, color="coral")
    y_max = max(rel_frob, default=1.0) * 1.30 or 1.0
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Relative Frobenius norm (%)")
    ax.set_title("||C_clipped - C_raw||_F / ||C_raw||_F")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    for bar, val in zip(bars, rel_frob):
        _label_bar(ax, bar, f"{val:.1f}%", y_max)

    # Number of negative eigenvalues
    ax = axes[1, 0]
    n_negs = [r.n_negative_eigs for r in results]
    bars = ax.bar(x, n_negs, width, color="mediumpurple")
    y_max = max(n_negs, default=10) * 1.30 or 10.0
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Count")
    ax.set_title("Negative eigenvalues in raw matrix")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    for bar, val in zip(bars, n_negs):
        _label_bar(ax, bar, str(val), y_max)

    # Marchenko-Pastur: signal vs noise eigenvalue counts (stacked bar)
    ax = axes[1, 1]
    n_signal = [r.n_signal_eigs for r in results]
    n_noise  = [r.n_noise_eigs  for r in results]
    n_total  = [s + n for s, n in zip(n_signal, n_noise)]
    bars_sig   = ax.bar(x, n_signal, width, color="seagreen",   label="Signal (> λ+)")
    bars_noise = ax.bar(x, n_noise,  width, bottom=n_signal,
                        color="lightgray", label="Noise / bulk (≤ λ+)")
    y_max_mp = max(n_total, default=100) * 1.12
    ax.set_ylim(0, y_max_mp)
    # Annotate signal count (inside green bar) and λ+ (above full stacked bar)
    for bar_s, bar_n, ns, q_val in zip(
        bars_sig, bars_noise, n_signal,
        [r.n_features / r.n_samples for r in results]
    ):
        lp = (1.0 + np.sqrt(q_val)) ** 2
        cx = bar_s.get_x() + bar_s.get_width() / 2
        # Signal count inside green portion
        ax.text(cx, bar_s.get_height() / 2,
                f"{ns}", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold")
        # λ+ label above the full stacked bar (noise top)
        full_top = bar_s.get_height() + bar_n.get_height()
        ax.text(cx, full_top + 0.015 * y_max_mp,
                f"λ+={lp:.2f}", ha="center", va="bottom",
                fontsize=7, color="darkgreen")
    ax.set_ylabel("Number of eigenvalues")
    ax.set_title("Marchenko-Pastur signal vs noise decomposition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_dir / "clipping_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved clipping_summary.png")


# ── data helpers ─────────────────────────────────────────────────────────────


def get_peptide_columns(df: pl.DataFrame) -> list[str]:
    pattern = re.compile("peptide", re.IGNORECASE)
    return [col for col in df.columns if pattern.search(col)]


def filter_sparse_columns(
    df: pl.DataFrame, columns: list[str], max_zero_frac: float = 0.4
) -> list[str]:
    """Drop columns whose zero fraction exceeds *max_zero_frac*."""
    return [
        c for c in columns
        if float((df[c] == 0).sum() / len(df)) <= max_zero_frac
    ]


# ── main ─────────────────────────────────────────────────────────────────────


def run(
    real_path: str,
    output_dir: str,
    methods: list[str],
    zero_frac: float = 0.4,
    event_column: str | None = None,
    event_types: list[str] | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(real_path).drop_nulls()
    peptide_cols = get_peptide_columns(df)
    peptide_cols = filter_sparse_columns(df, peptide_cols, max_zero_frac=zero_frac)
    print(
        f"Loaded {len(df)} rows, {len(peptide_cols)} peptide columns "
        f"(dropped cols with >{zero_frac*100:.0f}% zeros)"
    )

    # Build list of (label, sub-dataframe) pairs.
    # When event types are specified, analyse only those subsets (not the
    # aggregated "all" group) since the per-event breakdown is what matters.
    subsets: list[tuple[str, pl.DataFrame]] = []
    if event_column and event_types:
        for et in event_types:
            sub = df.filter(pl.col(event_column) == et)
            if len(sub) > 10:
                subsets.append((et, sub))
                print(f"  Event '{et}': {len(sub)} rows")
    # Fall back to the full dataset only when no event filter is provided.
    if not subsets:
        subsets = [("all", df)]

    all_results: list[ClippingAnalysis] = []

    for label, sub_df in subsets:
        X = sub_df.select(peptide_cols).to_numpy().astype(float)
        n, d = X.shape
        print(f"\n{'='*60}")
        print(f"Subset: {label}  (n={n}, d={d}, d/n={d/n:.2f})")
        print(f"{'='*60}")

        for method in methods:
            print(f"\n  Method: {method}")
            res = analyse_clipping(X, method, label)
            all_results.append(res)

            summary = res.summary_dict()
            for k, v in summary.items():
                print(f"    {k}: {v}")
            print(
                f"    [MP] λ+={res.mp_lambda_plus:.4f}, λ-={res.mp_lambda_minus:.4f} "
                f"| signal eigenvalues: {res.n_signal_eigs} / {res.n_features} "
                f"({res.n_signal_eigs / res.n_features * 100:.1f}%)"
            )

    # Save summary table
    import json
    summary_path = out / "clipping_analysis.json"
    with open(summary_path, "w") as f:
        json.dump([r.summary_dict() for r in all_results], f, indent=2)
    print(f"\nSaved metrics to {summary_path}")

    # Plots
    print("\nGenerating plots...")
    plot_eigenvalue_spectrum(all_results, out)
    plot_summary_bar(all_results, out)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify the effect of eigenvalue clipping on correlation matrices."
    )
    parser.add_argument(
        "--real", required=True, help="Path to the real data CSV."
    )
    parser.add_argument(
        "--output", default="evaluation_output/clipping_analysis",
        help="Output directory for results and figures.",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["spearman"],
        choices=["spearman", "kendall"],
        help="Correlation estimation methods to analyse.",
    )
    parser.add_argument(
        "--zero-frac", type=float, default=0.4,
        help="Drop peptide columns with a zero fraction above this threshold (default: 0.4).",
    )
    parser.add_argument(
        "--event-column", default="event_type",
        help="Column for per-event stratification (default: event_type).",
    )
    parser.add_argument(
        "--event-types", nargs="+", default=["hf", "ckd", "no_event"],
        help="Event types to analyse separately (default: hf ckd no_event).",
    )
    args = parser.parse_args()

    run(
        real_path=args.real,
        output_dir=args.output,
        methods=args.methods,
        zero_frac=args.zero_frac,
        event_column=args.event_column,
        event_types=args.event_types,
    )


if __name__ == "__main__":
    main()

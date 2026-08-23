"""Real vs. synthetic peptide abundance distributions, split by outcome subgroup.

Produces, for a set of selected peptides, overlaid real-vs-synthetic density
histograms (with KDE) faceted by outcome subgroup (HF, CKD/MAKE, NE) — matching
the Age/dBP/eGFR supplementary-figure style (real = blue, synthetic = gold).

Peptide selection: by default the N most abundant peptides (highest mean
abundance in the real training data, restricted to reasonably frequent peptides
so the distribution is meaningful). Override with --peptides.

Usage
-----
/data1/anaconda3/envs/kidney/bin/python scripts/subgroup_peptide_distributions.py
    [--n-peptides 6] [--peptides Peptide_1 Peptide_2 ...] [--outdir ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde, kurtosis, skew

REAL_PATH = "/data1/prostrat-ai/data/peptide_and_clinical_data_v2.csv"
SYN_PATH = "/home/fmirkovic/Peptide-synthesis/resources/synthetic_datasets/merged_synthetic.csv"

# Subgroups to show (data label -> panel title). CAD is excluded (as in the paper).
SUBGROUPS = [("hf", "HF event"), ("ckd", "CKD event (MAKE)"), ("no_event", "Non-event (NE)")]

REAL_COLOR = "#3B4CC0"   # blue  (real training)
SYN_COLOR = "#E8C520"    # gold  (synthetic)
MIN_NONZERO_FRACTION = 0.40  # only rank peptides that clear the 40% non-zero threshold


def peptide_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if "peptide" in c.lower()]


def select_peptides(real_lf: pl.LazyFrame, peps: list[str], n: int,
                    rank_by: str) -> list[str]:
    """Rank peptides by mean abundance or by fewest zeros (highest non-zero frac).

    ``rank_by='abundance'`` ranks by mean, restricted to peptides clearing the 40%
    non-zero threshold. ``rank_by='nonzero'`` ranks by non-zero fraction directly.
    """
    stats = (
        real_lf.select(
            [pl.col(c).mean().alias(f"{c}__mean") for c in peps]
            + [(pl.col(c) != 0).mean().alias(f"{c}__nz") for c in peps]
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    rows = [(c, stats[f"{c}__mean"], stats[f"{c}__nz"]) for c in peps]
    if rank_by == "nonzero":
        rows.sort(key=lambda r: (r[2] if r[2] is not None else -1, r[1] or -1),
                  reverse=True)
        criterion = "fewest zeros (highest non-zero fraction)"
    else:
        rows = [r for r in rows if r[2] is not None and r[2] >= MIN_NONZERO_FRACTION]
        rows.sort(key=lambda r: (r[1] if r[1] is not None else -1), reverse=True)
        criterion = "highest mean abundance (>=40% non-zero)"
    top = [r[0] for r in rows[:n]]
    print(f"Peptides selected by {criterion} (name, mean, non-zero frac):")
    for c in top:
        print(f"  {c}: mean={stats[f'{c}__mean']:.4g}, nz={stats[f'{c}__nz']:.2%}")
    return top


def select_best_overlap(peps: list[str], n: int, zero_max: float) -> list[str]:
    """Among peptides with < ``zero_max`` zero fraction, pick the n with the
    highest pooled real-vs-synthetic overlap coefficient."""
    nz = (
        pl.scan_csv(REAL_PATH)
        .select([(pl.col(c) != 0).mean().alias(c) for c in peps])
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    cand = [c for c in peps if nz[c] is not None and (1.0 - nz[c]) < zero_max]
    print(f"{len(cand)} peptides with < {zero_max:.0%} zeros; scoring overlap...")

    real = pl.read_csv(REAL_PATH, columns=cand)
    syn = pl.read_csv(SYN_PATH, columns=cand)
    scored = []
    for c in cand:
        rv = real[c].to_numpy().astype(float)
        sv = syn[c].to_numpy().astype(float)
        lo, hi = robust_xlim(rv, sv)
        scored.append((c, overlap_coefficient(rv, sv, lo, hi), nz[c]))
    scored.sort(key=lambda r: (r[1] if r[1] == r[1] else -1), reverse=True)
    top = [r[0] for r in scored[:n]]
    print("Best real-vs-synthetic overlap (name, overlap, non-zero frac):")
    for c, ovl, f in scored[:n]:
        print(f"  {c}: overlap={ovl:.3f}, nz={f:.2%}")
    return top


def select_gaussian(peps: list[str], n: int, zero_max: float) -> list[str]:
    """Among peptides with < ``zero_max`` zero fraction, pick the n whose *real*
    abundance distribution looks most Gaussian (symmetric, bell-shaped).

    Scored by non-normality = |skewness| + 0.5*|excess kurtosis| on the real
    values (zeros included, so a zero-spike is penalised); lower = more Gaussian.
    """
    nz = (
        pl.scan_csv(REAL_PATH)
        .select([(pl.col(c) != 0).mean().alias(c) for c in peps])
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    cand = [c for c in peps if nz[c] is not None and (1.0 - nz[c]) < zero_max]
    print(f"{len(cand)} peptides with < {zero_max:.0%} zeros; scoring normality...")

    real = pl.read_csv(REAL_PATH, columns=cand)
    scored = []
    for c in cand:
        v = real[c].to_numpy().astype(float)
        v = v[np.isfinite(v)]
        if v.size < 20 or np.ptp(v) == 0:
            continue
        sk = abs(float(skew(v)))
        ku = abs(float(kurtosis(v)))  # Fisher: 0 == normal
        scored.append((c, sk + 0.5 * ku, sk, ku, nz[c]))
    scored.sort(key=lambda r: r[1])  # lowest non-normality first
    top = [r[0] for r in scored[:n]]
    print("Most Gaussian real marginals (name, |skew|, |excess kurt|, non-zero frac):")
    for c, _score, sk, ku, f in scored[:n]:
        print(f"  {c}: |skew|={sk:.3f}, |kurt|={ku:.3f}, nz={f:.2%}")
    return top


def select_gaussian_overlap(peps: list[str], n: int, zero_max: float,
                            gauss_pool: int = 20,
                            zero_source: str = "real") -> list[str]:
    """Among peptides with < ``zero_max`` zero fraction, first keep the
    ``gauss_pool`` most Gaussian-looking real marginals (a normality gate), then
    return the ``n`` of those with the best real-vs-synthetic overlap.

    ``zero_source`` selects which dataset the zero-fraction filter is applied to:
    ``'real'`` (default) or ``'synth'``. Filtering on the synthetic zeros removes
    peptides where the copula's zero-inflated marginal emits a zero-spike, which
    is the artifact that breaks the bell shape in the plotted panels.

    Gate-then-rank rather than an equal-weight rank sum: overlap alone favours
    low-abundance right-skewed peptides, so a plain combination is dominated by
    overlap and yields non-Gaussian shapes. Gating on shape first guarantees the
    selected peptides actually look Gaussian, then overlap picks the best-fitting
    among those.
    """
    zero_path = SYN_PATH if zero_source == "synth" else REAL_PATH
    nz = (
        pl.scan_csv(zero_path)
        .select([(pl.col(c) != 0).mean().alias(c) for c in peps])
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    cand = [c for c in peps if nz[c] is not None and (1.0 - nz[c]) < zero_max]
    print(f"{len(cand)} peptides with < {zero_max:.0%} zeros in {zero_source} data; "
          "scoring normality + overlap...")

    real = pl.read_csv(REAL_PATH, columns=cand)
    syn = pl.read_csv(SYN_PATH, columns=cand)
    metrics = {}
    for c in cand:
        rv = real[c].to_numpy().astype(float)
        rv = rv[np.isfinite(rv)]
        if rv.size < 20 or np.ptp(rv) == 0:
            continue
        sv = syn[c].to_numpy().astype(float)
        lo, hi = robust_xlim(rv, sv)
        non_normality = abs(float(skew(rv))) + 0.5 * abs(float(kurtosis(rv)))
        ovl = overlap_coefficient(rv, sv, lo, hi)
        if ovl != ovl:  # NaN
            continue
        metrics[c] = {"nn": non_normality, "ovl": ovl, "nz": nz[c]}

    names = list(metrics)
    # 1) Gaussian gate: keep the most bell-shaped pool.
    gaussian_pool = sorted(names, key=lambda c: metrics[c]["nn"])[:gauss_pool]
    # 2) Rank that pool by overlap.
    ranked = sorted(gaussian_pool, key=lambda c: metrics[c]["ovl"], reverse=True)
    top = ranked[:n]
    worst_nn = metrics[gaussian_pool[-1]]["nn"] if gaussian_pool else float("nan")
    print(f"Gaussian gate = {len(gaussian_pool)} most bell-shaped "
          f"(|skew|+.5|kurt| <= {worst_nn:.3f}); ranked by overlap:")
    for c in top:
        m = metrics[c]
        print(f"  {c}: overlap={m['ovl']:.3f}, non_normality={m['nn']:.3f}, "
              f"nz={m['nz']:.2%}")
    return top


def robust_xlim(*arrays: np.ndarray) -> tuple[float, float]:
    # Clip the display range to the pooled 1st-98th percentile so a single
    # extreme value (common in small subgroups) does not squash the bulk.
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a.size])
    if vals.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(vals, [1, 98])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def overlap_coefficient(real_vals: np.ndarray, syn_vals: np.ndarray,
                        lo: float, hi: float, bins: int = 45) -> float:
    """Histogram overlap coefficient in [0, 1] (same metric as Table S2)."""
    r = real_vals[np.isfinite(real_vals)]
    s = syn_vals[np.isfinite(syn_vals)]
    if r.size == 0 or s.size == 0:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    rh, _ = np.histogram(r, bins=edges, density=True)
    sh, _ = np.histogram(s, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(rh, sh).sum() * width)


def plot_panel(ax, real_vals: np.ndarray, syn_vals: np.ndarray, title: str):
    lo, hi = robust_xlim(real_vals, syn_vals)
    bins = np.linspace(lo, hi, 45)
    for vals, color, label in [
        (real_vals, REAL_COLOR, "Real Training"),
        (syn_vals, SYN_COLOR, "Synthetic"),
    ]:
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            continue
        ax.hist(v, bins=bins, density=True, color=color, alpha=0.55,
                edgecolor="white", linewidth=0.3, label=label)
        vin = v[(v >= lo) & (v <= hi)]
        if vin.size > 5 and np.ptp(vin) > 0:
            try:
                kde = gaussian_kde(vin)
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, kde(xs), color=color, linewidth=1.6)
            except np.linalg.LinAlgError:
                pass
    ovl = overlap_coefficient(real_vals, syn_vals, lo, hi)
    ax.set_title(title, fontsize=10)
    ax.text(0.97, 0.72, f"overlap = {ovl:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", alpha=0.8))
    ax.set_xlim(lo, hi)
    ax.set_ylabel("Density", fontsize=9)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-peptides", type=int, default=6)
    ap.add_argument("--peptides", nargs="*", default=None,
                    help="Explicit peptide column names (overrides --n-peptides).")
    ap.add_argument("--rank-by",
                    choices=["abundance", "nonzero", "overlap", "gaussian",
                             "gaussian_overlap"],
                    default="abundance",
                    help="Auto-select by highest mean abundance, fewest zeros, "
                         "best real-vs-synthetic overlap, most Gaussian-looking "
                         "real marginal, or jointly best on Gaussian shape AND "
                         "overlap (last three respect --zero-max).")
    ap.add_argument("--zero-max", type=float, default=0.40,
                    help="For --rank-by overlap/gaussian: only consider peptides with "
                         "a zero fraction below this value (default 0.40 = '<40%% "
                         "zeros').")
    ap.add_argument("--gauss-pool", type=int, default=20,
                    help="For --rank-by gaussian_overlap: size of the most-Gaussian "
                         "pool to gate on before ranking by overlap (smaller = "
                         "stricter bell-shape requirement; default 20).")
    ap.add_argument("--zero-source", choices=["real", "synth"], default="real",
                    help="For --rank-by gaussian_overlap: apply the --zero-max filter "
                         "to the zero fraction of the real or synthetic data "
                         "(synth removes copula zero-spike artifacts; default real).")
    ap.add_argument("--outdir", default="reports/subgroup_peptide_distributions")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Identify common peptide columns.
    real_head = pl.read_csv(REAL_PATH, n_rows=1)
    syn_head = pl.read_csv(SYN_PATH, n_rows=1)
    common_peps = sorted(set(peptide_columns(real_head)) & set(peptide_columns(syn_head)))
    print(f"{len(common_peps)} peptide columns common to real and synthetic.")

    # Select peptides.
    if args.peptides:
        selected = [p for p in args.peptides if p in common_peps]
        missing = set(args.peptides) - set(selected)
        if missing:
            print(f"WARNING: not found / not common: {sorted(missing)}")
    elif args.rank_by == "overlap":
        selected = select_best_overlap(common_peps, args.n_peptides, args.zero_max)
    elif args.rank_by == "gaussian":
        selected = select_gaussian(common_peps, args.n_peptides, args.zero_max)
    elif args.rank_by == "gaussian_overlap":
        selected = select_gaussian_overlap(common_peps, args.n_peptides, args.zero_max,
                                           gauss_pool=args.gauss_pool,
                                           zero_source=args.zero_source)
    else:
        real_lf = pl.scan_csv(REAL_PATH).select(common_peps)
        selected = select_peptides(real_lf, common_peps, args.n_peptides, args.rank_by)
    if not selected:
        raise SystemExit("No peptides selected.")

    # Load only what we need.
    cols = selected + ["event_type"]
    real = pl.read_csv(REAL_PATH, columns=cols)
    syn = pl.read_csv(SYN_PATH, columns=cols)

    n_pep, n_sub = len(selected), len(SUBGROUPS)
    fig, axes = plt.subplots(
        n_pep, n_sub, figsize=(4.2 * n_sub, 2.7 * n_pep), squeeze=False
    )

    for i, pep in enumerate(selected):
        for j, (label, title) in enumerate(SUBGROUPS):
            ax = axes[i][j]
            rv = real.filter(pl.col("event_type") == label)[pep].to_numpy().astype(float)
            sv = syn.filter(pl.col("event_type") == label)[pep].to_numpy().astype(float)
            plot_panel(ax, rv, sv, f"{title}  (real n={rv.size}, synth n={sv.size})")
            if j == 0:
                ax.text(-0.32, 0.5, pep, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
            if i == 0 and j == n_sub - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Peptide abundance distributions: real training vs. synthetic, by outcome subgroup",
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    grid_path = outdir / "subgroup_peptide_distributions_grid.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid: {grid_path}")

    # Also one 1x3 figure per peptide (easy to place individually in the manuscript).
    for pep in selected:
        f, axs = plt.subplots(1, n_sub, figsize=(4.2 * n_sub, 3.2), squeeze=False)
        for j, (label, title) in enumerate(SUBGROUPS):
            ax = axs[0][j]
            rv = real.filter(pl.col("event_type") == label)[pep].to_numpy().astype(float)
            sv = syn.filter(pl.col("event_type") == label)[pep].to_numpy().astype(float)
            plot_panel(ax, rv, sv, f"{title}  (real n={rv.size}, synth n={sv.size})")
            if j == n_sub - 1:
                ax.legend(fontsize=8, loc="upper right")
        f.suptitle(f"Distribution of {pep} — Real Training vs Synthetic", fontsize=12)
        f.tight_layout()
        p = outdir / f"dist_{pep}.png"
        f.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(f)
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()

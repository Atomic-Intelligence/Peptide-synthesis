"""Missingness--outcome dependence: real vs synthetic (reviewer comment 6).

Beyond *marginal* missingness fidelity, this quantifies whether the association
between **whether a value is missing** and the **clinical outcome** survives
synthesis.

Operational definition of the missingness indicator (see paper Methods):
  * peptides  -> M = 1[value == 0]   (non-detection / "missing" abundance)
  * survival  -> M = 1[value is null] (field absent, e.g. no time-to-event)
Nothing is imputed; the goal is to reproduce the *pattern* of missingness and
its relationship to outcome, not to fill values.

Outcome mapping (from merge_synthetic_datasets.py): HF=hf, MAKE=ckd, NE=no_event.

Analyses
--------
A1  Conditional missingness table (Supplementary Table S8): missingness
    fraction stratified by outcome group (HF / MAKE / NE), real vs synthetic,
    per indicator, with a per-cell Fisher exact real-vs-synthetic difference
    test and BH (Benjamini-Hochberg) correction.

A2  Association strength: for each binary outcome Y in {HF (hf vs no_event),
    MAKE (ckd vs no_event)} and each indicator M_j, the log odds ratio of
    Y ~ M_j (closed-form 2x2 with Haldane 0.5 correction; identical to the
    logistic-regression coefficient for a single binary predictor) in real and
    synthetic separately, a z-test on the difference in log-OR (pooled SE,
    mirroring the KM-quintile HR z-test), BH correction, and a real-vs-synthetic
    log-OR scatter with Spearman rho (Figure 3C style).

A3  (optional) Multivariable logistic outcome ~ (all indicators) fit in real and
    synthetic on a shared reduced feature set; correlation of the coefficient
    vectors, showing the *joint* mapping is preserved.

Run (kidney env has polars + scipy + statsmodels):
  PYTHONPATH=/home/fmirkovic/Peptide-synthesis \
  /data1/anaconda3/envs/kidney/bin/python scripts/missingness_outcome_dependence.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import fisher_exact, spearmanr
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EVENT_COL = "event_type"
EGFR_COL = "GFR_CKD_EPI_M"  # continuous endpoint (point-biserial vs missingness)
DROP_EVENTS = ("cad",)  # excluded upstream throughout the paper
# outcome label -> (event value, contrast/control value)
OUTCOMES = {
    "HF": ("hf", "no_event"),
    "MAKE": ("ckd", "no_event"),
}
OUTCOME_GROUPS = {"HF": "hf", "MAKE": "ckd", "NE": "no_event"}
# "Informative missingness" band (paper text): peptides whose detectability
# actually varies. Peptides at ~0% missing carry no signal; peptides at >40%
# are the sparse/histogram-sampled branch (dropped from the copula, see
# pca_copula_synthetic_data.py max_zero_frac=0.4) and are the negative control.
INFORMATIVE_LO, INFORMATIVE_HI = 0.05, 0.40
SURVIVAL_COLS = [
    "time-to-CKDevent(to event or last visit)",
    "FU duration_CAD_Hfevent (to event or last visit)",
]
HALDANE = 0.5  # continuity correction so no 2x2 cell is zero
PEPTIDE_RE = re.compile("peptide", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load(real_path: str, synth_path: str):
    real = pl.read_csv(real_path, infer_schema_length=3000)
    synth = pl.read_csv(synth_path, infer_schema_length=3000)
    # drop the copula's echoed "_duplicated_0" columns
    synth = synth.select([c for c in synth.columns if "_duplicated_" not in c])
    for ev in DROP_EVENTS:
        real = real.filter(pl.col(EVENT_COL) != ev)
        synth = synth.filter(pl.col(EVENT_COL) != ev)
    logger.info(f"real rows={real.height}  synth rows={synth.height}")
    return real, synth


def peptide_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if PEPTIDE_RE.search(c)]


def missingness_matrix(df: pl.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    """(n_rows, n_cols) float array of the missingness indicator M (1 = missing)."""
    arr = df.select(cols).to_numpy()
    if kind == "peptide":
        return (arr == 0).astype(np.float64)
    # survival/clinical: null (NaN) means the value was not recorded
    return np.isnan(arr.astype(np.float64)).astype(np.float64)


# ---------------------------------------------------------------------------
# Vectorised 2x2 log-OR for a binary predictor M against binary outcome y
# ---------------------------------------------------------------------------
def log_or_vectorised(M: np.ndarray, y: np.ndarray):
    """Log-OR + SE of y ~ M_j for every column of M (Haldane-corrected).

    a=missing&event, b=missing&control, c=present&event, d=present&control.
    Returns (log_or, se, valid) where valid flags columns that vary in this set.
    """
    y = y.astype(np.float64)
    n = y.shape[0]
    n_event = y.sum()
    m_sum = M.sum(axis=0)  # a + b   (# missing)
    a = M.T @ y  # missing & event
    b = m_sum - a  # missing & control
    c = n_event - a  # present & event
    d = (n - n_event) - b  # present & control
    valid = (m_sum > 0) & (m_sum < n)  # indicator varies in this subset
    a_, b_, c_, d_ = a + HALDANE, b + HALDANE, c + HALDANE, d + HALDANE
    log_or = np.log((a_ * d_) / (b_ * c_))
    se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    return log_or, se, valid


def subset_binary(df: pl.DataFrame, event: str, control: str):
    """Rows for the {event, control} contrast + the binary outcome vector."""
    sub = df.filter(pl.col(EVENT_COL).is_in([event, control]))
    y = (sub.select(pl.col(EVENT_COL) == event).to_numpy().ravel()).astype(np.float64)
    return sub, y


def real_zero_frac(real: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Per-peptide missingness fraction in the REAL data (full cohort)."""
    return missingness_matrix(real, cols, "peptide").mean(axis=0)


def band_masks(real: pl.DataFrame, cols: list[str]):
    """(informative, sparse) boolean masks over `cols`, defined on real data.

    informative: INFORMATIVE_LO <= zero_frac <= INFORMATIVE_HI  (copula branch,
                 detectability varies -> the reviewer's case of interest)
    sparse:      zero_frac > INFORMATIVE_HI  (histogram-sampled, dropped from the
                 copula and excluded downstream -> negative control)
    """
    zf = real_zero_frac(real, cols)
    informative = (zf >= INFORMATIVE_LO) & (zf <= INFORMATIVE_HI)
    sparse = zf > INFORMATIVE_HI
    return informative, sparse


def point_biserial_vectorised(M: np.ndarray, g: np.ndarray):
    """Point-biserial r between each missingness column M_j and continuous g.

    Identical to Pearson r(M_j, g); the sign says whether *non-detection*
    associates with higher/lower g (e.g. eGFR). Returns (r, valid).
    """
    g = g.astype(np.float64)
    gc = g - g.mean()
    denom_g = np.sqrt((gc**2).sum())
    Mc = M - M.mean(axis=0)
    num = Mc.T @ gc
    denom_M = np.sqrt((Mc**2).sum(axis=0))
    valid = (M.sum(axis=0) > 0) & (M.sum(axis=0) < M.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / (denom_M * denom_g)
    return r, valid


# ---------------------------------------------------------------------------
# Analysis 2 -- association strength + scatter
# ---------------------------------------------------------------------------
def analysis2(real, synth, common_pep, out_dir: Path):
    results = {}
    for label, (event, control) in OUTCOMES.items():
        r_sub, r_y = subset_binary(real, event, control)
        s_sub, s_y = subset_binary(synth, event, control)
        Mr = missingness_matrix(r_sub, common_pep, "peptide")
        Ms = missingness_matrix(s_sub, common_pep, "peptide")
        lo_r, se_r, val_r = log_or_vectorised(Mr, r_y)
        lo_s, se_s, val_s = log_or_vectorised(Ms, s_y)
        valid = val_r & val_s
        z = (lo_r - lo_s) / np.sqrt(se_r**2 + se_s**2)
        # two-sided p for the real-vs-synthetic difference
        from scipy.stats import norm

        p_diff = 2 * norm.sf(np.abs(z))
        p_bh = np.full_like(p_diff, np.nan)
        if valid.sum() > 0:
            p_bh[valid] = multipletests(p_diff[valid], method="fdr_bh")[1]

        rho, rho_p = spearmanr(lo_r[valid], lo_s[valid])
        n_sig = int(np.nansum(p_bh[valid] < 0.05))
        # rho restricted to indicators with a *real* signal (BH-significant real
        # association) -- most peptides carry no real missingness->outcome signal,
        # so the all-indicator rho is diluted by noise-vs-noise.
        from scipy.stats import norm as _norm

        p_real = 2 * _norm.sf(np.abs(lo_r / se_r))
        real_sig = valid.copy()
        if valid.sum():
            bh_real = np.zeros_like(p_real, dtype=bool)
            bh_real[valid] = multipletests(p_real[valid], method="fdr_bh")[1] < 0.05
            real_sig = valid & bh_real
        if real_sig.sum() >= 3:
            rho_sig, rho_sig_p = spearmanr(lo_r[real_sig], lo_s[real_sig])
        else:
            rho_sig, rho_sig_p = float("nan"), float("nan")
        logger.info(
            f"[A2 {label}] n={int(valid.sum())}  rho(all)={rho:.3f} (p={rho_p:.2g})  "
            f"rho(real-signal n={int(real_sig.sum())})={rho_sig:.3f} (p={rho_sig_p:.2g})  "
            f"sig real-vs-synth diffs after BH: {n_sig} "
            f"({100 * n_sig / max(valid.sum(), 1):.1f}%)"
        )

        df = pl.DataFrame(
            {
                "peptide": common_pep,
                "log_or_real": lo_r,
                "se_real": se_r,
                "log_or_synth": lo_s,
                "se_synth": se_s,
                "z_diff": z,
                "p_diff": p_diff,
                "p_diff_bh": p_bh,
                "valid": valid,
            }
        ).filter(pl.col("valid"))
        df.write_csv(out_dir / f"table_S9_source_logOR_{label}.csv")
        results[label] = {
            "n_indicators": int(valid.sum()),
            "spearman_rho_all": float(rho),
            "spearman_p_all": float(rho_p),
            "n_real_signal_indicators": int(real_sig.sum()),
            "spearman_rho_real_signal": float(rho_sig),
            "spearman_p_real_signal": float(rho_sig_p),
            "n_sig_diff_after_bh": n_sig,
            "frac_sig_diff_after_bh": float(n_sig / max(valid.sum(), 1)),
            "_lo_r": lo_r[valid],
            "_lo_s": lo_s[valid],
            "_lo_r_sig": lo_r[real_sig],
            "_lo_s_sig": lo_s[real_sig],
        }
    _scatter(results, out_dir)
    return results


def _scatter(results, out_dir: Path):
    labels = list(results.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 5), squeeze=False)
    for ax, label in zip(axes[0], labels):
        a, b = results[label]["_lo_r"], results[label]["_lo_s"]
        lim = np.nanpercentile(np.abs(np.concatenate([a, b])), 99)
        lim = float(max(lim, 0.5))
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        asig, bsig = results[label]["_lo_r_sig"], results[label]["_lo_s_sig"]
        ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=1, zorder=1, label="y = x")
        ax.scatter(
            a, b, s=7, alpha=0.25, color="0.6", edgecolors="none", zorder=2,
            label=f"all ({results[label]['n_indicators']})",
        )
        ax.scatter(
            asig, bsig, s=14, alpha=0.7, color="tab:blue", edgecolors="none", zorder=3,
            label=f"real signal ({results[label]['n_real_signal_indicators']})",
        )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Real log-OR (missingness $\\to$ outcome)", fontsize=12)
        ax.set_ylabel("Synthetic log-OR", fontsize=12)
        ax.set_title(
            f"{label}: $\\rho_{{all}}$ = {results[label]['spearman_rho_all']:.2f}, "
            f"$\\rho_{{signal}}$ = {results[label]['spearman_rho_real_signal']:.2f}",
            fontsize=12,
        )
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Missingness$\\to$outcome association preserved under synthesis", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figure_S15BC_missingness_logOR.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_dir / 'figure_S15BC_missingness_logOR.png'}")


# ---------------------------------------------------------------------------
# Analysis 1 -- conditional missingness table (Supplementary Table S8)
# ---------------------------------------------------------------------------
def _cond_rows(real, synth, name, real_col_idx, synth_col_idx, Mr_full, Ms_full):
    """One indicator: missingness fraction per outcome group, real vs synth,
    with a Fisher exact real-vs-synthetic difference test per stratum."""
    rows = []
    for grp_label, ev in OUTCOME_GROUPS.items():
        r_mask = (real[EVENT_COL] == ev).to_numpy()
        s_mask = (synth[EVENT_COL] == ev).to_numpy()
        r_ind = Mr_full[r_mask, real_col_idx]
        s_ind = Ms_full[s_mask, synth_col_idx]
        r_miss, r_n = int(r_ind.sum()), int(r_ind.shape[0])
        s_miss, s_n = int(s_ind.sum()), int(s_ind.shape[0])
        table = [[r_miss, r_n - r_miss], [s_miss, s_n - s_miss]]
        try:
            _, p = fisher_exact(table)
        except ValueError:
            p = np.nan
        rf = r_miss / r_n if r_n else np.nan
        sf = s_miss / s_n if s_n else np.nan
        rows.append(
            {
                "indicator": name,
                "outcome_group": grp_label,
                "real_frac_missing": rf,
                "real_n": r_n,
                "synth_frac_missing": sf,
                "synth_n": s_n,
                "abs_frac_diff": abs(rf - sf),
                "fisher_p": p,
            }
        )
    return rows


def analysis1(real, synth, common_pep, a2_results, out_dir: Path, n_top: int):
    # Peptides only (per decision): survival-field missingness is not modelled by
    # the copula and is excluded from Table S8.
    #
    # Exemplar peptides = strongest *real* missingness->outcome signal (union of
    # top-|z| under HF and MAKE): the informative-missingness cases the reviewer
    # cares about. Reported with the effect size (|Δfraction|), not just the
    # Fisher p -- with n_real (5201) >> n_synth (2000) the exact test detects
    # tiny, immaterial fraction gaps, so significance alone would mislead.
    top = set()
    for label in OUTCOMES:
        df = pl.read_csv(out_dir / f"table_S9_source_logOR_{label}.csv")
        df = df.with_columns((pl.col("log_or_real") / pl.col("se_real")).abs().alias("zr"))
        top.update(df.sort("zr", descending=True).head(n_top)["peptide"].to_list())
    top_pep = [p for p in common_pep if p in top]

    Mr_pep = missingness_matrix(real, common_pep, "peptide")
    Ms_pep = missingness_matrix(synth, common_pep, "peptide")
    r_idx = {c: i for i, c in enumerate(common_pep)}

    # Aggregate agreement of per-stratum missingness FRACTIONS across all peptides:
    # directly answers "does the missingness rate within each outcome group match
    # between real and synthetic?" without the selection bias of the exemplar table.
    agg = {}
    for grp_label, ev in OUTCOME_GROUPS.items():
        rf = Mr_pep[(real[EVENT_COL] == ev).to_numpy()].mean(axis=0)
        sf = Ms_pep[(synth[EVENT_COL] == ev).to_numpy()].mean(axis=0)
        keep = (rf > 0) & (rf < 1)  # peptides with any variation in real
        rho_g, p_g = spearmanr(rf[keep], sf[keep])
        mae = float(np.mean(np.abs(rf[keep] - sf[keep])))
        agg[grp_label] = {
            "n_peptides": int(keep.sum()),
            "spearman_rho": float(rho_g),
            "spearman_p": float(p_g),
            "mean_abs_frac_diff": mae,
        }
        logger.info(
            f"[A1-agg {grp_label}] missingness-fraction agreement across "
            f"{int(keep.sum())} peptides: rho={rho_g:.3f} (p={p_g:.2g}) "
            f"mean|Δfrac|={mae:.3f}"
        )

    # Table S8 (headline): aggregate conditional-missingness agreement per group.
    s8 = pl.DataFrame(
        [
            {
                "outcome_group": g,
                "n_peptides": v["n_peptides"],
                "spearman_rho_real_vs_synth": v["spearman_rho"],
                "spearman_p": v["spearman_p"],
                "mean_abs_frac_diff": v["mean_abs_frac_diff"],
            }
            for g, v in agg.items()
        ]
    )
    s8.write_csv(out_dir / "table_S8_conditional_missingness.csv")

    # Exemplar peptides (supporting): informative-missingness cases, real vs synth.
    rows = []
    for name in top_pep:
        k = r_idx[name]
        rows += _cond_rows(real, synth, name, k, k, Mr_pep, Ms_pep)
    ex = pl.DataFrame(rows)
    pv = ex["fisher_p"].to_numpy()
    ok = ~np.isnan(pv)
    p_bh = np.full_like(pv, np.nan)
    if ok.sum():
        p_bh[ok] = multipletests(pv[ok], method="fdr_bh")[1]
    ex = ex.with_columns(pl.Series("fisher_p_bh", p_bh))
    ex.write_csv(out_dir / "table_S8b_exemplar_peptides.csv")

    n_sig = int(np.nansum(p_bh < 0.05))
    med_absdiff = float(np.nanmedian(ex["abs_frac_diff"].to_numpy()))
    logger.info(
        f"[A1] Table S8 (aggregate, {s8.height} groups) + exemplars "
        f"({len(top_pep)} peptides x 3 groups = {ex.height} cells); "
        f"exemplar cells with BH-sig real-vs-synth diff: {n_sig}/{ex.height} "
        f"(median |Δfrac|={med_absdiff:.3f}; exemplars are the most extreme "
        f"peptides, so levels shift more here than the ~0.11-0.14 pooled mean)"
    )
    return s8, agg


# ---------------------------------------------------------------------------
# Analysis 2b -- continuous endpoint (eGFR): point-biserial concordance
# ---------------------------------------------------------------------------
def analysis_egfr(real, synth, common_pep, informative, out_dir: Path):
    """Per-peptide missingness<->eGFR point-biserial, real vs synthetic.

    Restricted to the informative-missingness band (paper text). eGFR is defined
    for the whole cohort; rows with a null eGFR are dropped per dataset.
    """
    if EGFR_COL not in real.columns or EGFR_COL not in synth.columns:
        logger.warning(f"eGFR column '{EGFR_COL}' absent; skipping eGFR analysis")
        return None
    gr = real.select(EGFR_COL).to_numpy().ravel().astype(np.float64)
    gs = synth.select(EGFR_COL).to_numpy().ravel().astype(np.float64)
    mr, ms = ~np.isnan(gr), ~np.isnan(gs)
    Mr = missingness_matrix(real, common_pep, "peptide")[mr]
    Ms = missingness_matrix(synth, common_pep, "peptide")[ms]
    r_r, val_r = point_biserial_vectorised(Mr, gr[mr])
    r_s, val_s = point_biserial_vectorised(Ms, gs[ms])
    valid = val_r & val_s & informative & np.isfinite(r_r) & np.isfinite(r_s)
    rho, rho_p = spearmanr(r_r[valid], r_s[valid])
    logger.info(
        f"[eGFR] informative-band peptides n={int(valid.sum())}  "
        f"point-biserial concordance rho={rho:.3f} (p={rho_p:.2g})"
    )
    pl.DataFrame(
        {
            "peptide": [c for c, k in zip(common_pep, valid) if k],
            "pbc_real": r_r[valid],
            "pbc_synth": r_s[valid],
        }
    ).write_csv(out_dir / "table_S9_source_egfr_pbc.csv")

    fig, ax = plt.subplots(figsize=(5.2, 5))
    a, b = r_r[valid], r_s[valid]
    lim = float(max(np.nanpercentile(np.abs(np.concatenate([a, b])), 99), 0.1))
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=1, zorder=1, label="y = x")
    ax.scatter(a, b, s=9, alpha=0.35, color="tab:green", edgecolors="none", zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Real point-biserial (missingness $\\to$ eGFR)", fontsize=12)
    ax.set_ylabel("Synthetic point-biserial", fontsize=12)
    ax.set_title(f"eGFR (S15A): $\\rho$ = {rho:.2f}, n = {int(valid.sum())}", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_S15A_missingness_eGFR.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_informative_peptides": int(valid.sum()),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }


# ---------------------------------------------------------------------------
# Analysis 3 -- missingness-fingerprint classifier (TSTR); Figure S15D
# ---------------------------------------------------------------------------
def analysis_tstr(real, synth, band_pep, out_dir: Path, seed: int = 0):
    """Predict outcome from the missingness-indicator matrix ALONE.

    The strongest single test: it probes the *joint* missingness->outcome
    dependence, not pairwise. For each binary outcome we report AUC for
      TRTR  (train real / test real, 5-fold CV)   -- reference
      TSTS  (train synth / test synth, 5-fold CV) -- internal synthetic utility
      TSTR  (train synth / test real, held out)   -- the claim we must support
    L2 logistic, balanced classes (MAKE is rare). Features = informative-band
    peptide missingness indicators only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    def _clf():
        return LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, class_weight="balanced"
        )

    results, roc_data = {}, {}
    for label, (event, control) in OUTCOMES.items():
        r_sub, r_y = subset_binary(real, event, control)
        s_sub, s_y = subset_binary(synth, event, control)
        Xr = missingness_matrix(r_sub, band_pep, "peptide")
        Xs = missingness_matrix(s_sub, band_pep, "peptide")
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)

        p_trtr = cross_val_predict(_clf(), Xr, r_y, cv=cv, method="predict_proba")[:, 1]
        auc_trtr = roc_auc_score(r_y, p_trtr)
        p_tsts = cross_val_predict(_clf(), Xs, s_y, cv=cv, method="predict_proba")[:, 1]
        auc_tsts = roc_auc_score(s_y, p_tsts)
        p_tstr = _clf().fit(Xs, s_y).predict_proba(Xr)[:, 1]
        auc_tstr = roc_auc_score(r_y, p_tstr)

        results[label] = {
            "n_features": len(band_pep),
            "auc_trtr_real": float(auc_trtr),
            "auc_tsts_synth": float(auc_tsts),
            "auc_tstr": float(auc_tstr),
            "n_real": int(r_y.shape[0]),
            "n_synth": int(s_y.shape[0]),
            "prevalence_real": float(r_y.mean()),
        }
        roc_data[label] = {
            "TRTR": (roc_curve(r_y, p_trtr), auc_trtr),
            "TSTS": (roc_curve(s_y, p_tsts), auc_tsts),
            "TSTR": (roc_curve(r_y, p_tstr), auc_tstr),
        }
        logger.info(
            f"[TSTR {label}] TRTR(real)={auc_trtr:.3f}  TSTS(synth)={auc_tsts:.3f}  "
            f"TSTR(train-synth/test-real)={auc_tstr:.3f}  "
            f"(n_features={len(band_pep)}, prev_real={r_y.mean():.3f})"
        )

    labels = list(roc_data.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(5.2 * len(labels), 5), squeeze=False)
    styles = {"TRTR": ("tab:gray", "-"), "TSTS": ("tab:orange", "--"), "TSTR": ("tab:blue", "-")}
    for ax, label in zip(axes[0], labels):
        ax.plot([0, 1], [0, 1], ":", color="0.7", lw=1)
        for cfg, ((fpr, tpr, _), auc) in roc_data[label].items():
            c, ls = styles[cfg]
            ax.plot(fpr, tpr, ls, color=c, lw=1.8, label=f"{cfg} (AUC={auc:.2f})")
        ax.set_xlabel("False positive rate", fontsize=12)
        ax.set_ylabel("True positive rate", fontsize=12)
        ax.set_title(f"{label}: missingness-only classifier", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
    fig.suptitle("Missingness fingerprint predicts outcome (S15D)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_S15D_missingness_fingerprint_ROC.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {out_dir / 'figure_S15D_missingness_fingerprint_ROC.png'}")
    return results


# ---------------------------------------------------------------------------
# Negative control -- sparse (histogram-sampled) peptides carry ~no signal
# ---------------------------------------------------------------------------
def negative_control(real, synth, common_pep, sparse, out_dir: Path):
    """Sparse peptides are histogram-sampled independently of outcome and
    excluded downstream (>40% zeros -> dropped from the copula). Confirm their
    synthetic missingness->outcome association is ~0 and uncorrelated with real.
    """
    sparse_pep = [c for c, k in zip(common_pep, sparse) if k]
    if not sparse_pep:
        logger.warning("no sparse peptides for negative control")
        return None
    out = {}
    for label, (event, control) in OUTCOMES.items():
        r_sub, r_y = subset_binary(real, event, control)
        s_sub, s_y = subset_binary(synth, event, control)
        Mr = missingness_matrix(r_sub, sparse_pep, "peptide")
        Ms = missingness_matrix(s_sub, sparse_pep, "peptide")
        lo_r, _, val_r = log_or_vectorised(Mr, r_y)
        lo_s, _, val_s = log_or_vectorised(Ms, s_y)
        valid = val_r & val_s & np.isfinite(lo_r) & np.isfinite(lo_s)
        rho, rho_p = spearmanr(lo_r[valid], lo_s[valid]) if valid.sum() >= 3 else (np.nan, np.nan)
        out[label] = {
            "n_sparse_peptides": int(valid.sum()),
            "median_abs_logOR_synth": float(np.nanmedian(np.abs(lo_s[valid]))),
            "median_abs_logOR_real": float(np.nanmedian(np.abs(lo_r[valid]))),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
        }
        logger.info(
            f"[neg-control {label}] sparse n={int(valid.sum())}  "
            f"median|log-OR| synth={out[label]['median_abs_logOR_synth']:.3f} "
            f"(real={out[label]['median_abs_logOR_real']:.3f})  "
            f"real-vs-synth rho={rho:.3f} (p={rho_p:.2g})"
        )
    return out


# ---------------------------------------------------------------------------
# Supplementary Table S9 -- one-row-per-endpoint dependence summary
# ---------------------------------------------------------------------------
def assemble_table_s9(a2, egfr, tstr, out_dir: Path):
    """Compact per-endpoint summary: per-peptide concordance + classifier AUCs."""
    rows = []
    if egfr is not None:
        rows.append(
            {
                "endpoint": "eGFR",
                "association_metric": "point-biserial r",
                "n_peptides": egfr["n_informative_peptides"],
                "concordance_rho": round(egfr["spearman_rho"], 3),
                "concordance_p": egfr["spearman_p"],
                "auc_trtr_real": None,
                "auc_tsts_synth": None,
                "auc_tstr": None,
            }
        )
    for label in OUTCOMES:
        a = a2.get(label, {})
        t = (tstr or {}).get(label, {})
        rows.append(
            {
                "endpoint": label,
                "association_metric": "log odds ratio",
                "n_peptides": a.get("n_real_signal_indicators"),
                "concordance_rho": round(a.get("spearman_rho_real_signal", float("nan")), 3),
                "concordance_p": a.get("spearman_p_real_signal"),
                "auc_trtr_real": round(t.get("auc_trtr_real", float("nan")), 3),
                "auc_tsts_synth": round(t.get("auc_tsts_synth", float("nan")), 3),
                "auc_tstr": round(t.get("auc_tstr", float("nan")), 3),
            }
        )
    s9 = pl.DataFrame(rows)
    s9.write_csv(out_dir / "table_S9_dependence_summary.csv")
    logger.info(f"wrote {out_dir / 'table_S9_dependence_summary.csv'}\n{s9}")
    return s9


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real",
        default="/data1/prostrat-ai/data/peptide_and_clinical_data_v2.csv",
    )
    ap.add_argument(
        "--synth",
        default="resources/synthetic_datasets/merged_synthetic.csv",
    )
    ap.add_argument("--out", default="evaluation_output/missingness_outcome")
    ap.add_argument("--n-top", type=int, default=15, help="top peptides per outcome for Table S8")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    real, synth = load(args.real, args.synth)
    rp, sp = set(peptide_cols(real)), set(peptide_cols(synth))
    common_pep = sorted(rp & sp)
    logger.info(f"common peptides: {len(common_pep)}")

    informative, sparse = band_masks(real, common_pep)
    band_pep = [c for c, k in zip(common_pep, informative) if k]
    logger.info(
        f"informative-missingness band ({INFORMATIVE_LO:.0%}-{INFORMATIVE_HI:.0%}): "
        f"{len(band_pep)} peptides; sparse (negative control): {int(sparse.sum())}"
    )

    a2 = analysis2(real, synth, common_pep, out_dir)
    _, agg = analysis1(real, synth, common_pep, a2, out_dir, args.n_top)
    egfr = analysis_egfr(real, synth, common_pep, informative, out_dir)
    tstr = analysis_tstr(real, synth, band_pep, out_dir)
    negctrl = negative_control(real, synth, common_pep, sparse, out_dir)
    assemble_table_s9(a2, egfr, tstr, out_dir)

    summary = {
        "n_common_peptides": len(common_pep),
        "n_informative_band_peptides": len(band_pep),
        "n_sparse_peptides": int(sparse.sum()),
        "analysis1_aggregate_fraction_agreement": agg,
        "analysis2": {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in a2.items()
        },
        "analysis_egfr_point_biserial": egfr,
        "analysis_tstr_fingerprint": tstr,
        "negative_control_sparse": negctrl,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("summary:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

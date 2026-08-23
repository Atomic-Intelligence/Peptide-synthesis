"""Side-by-side robust-privacy comparison: full feature space vs peptides with <40% zeros.

Reads the two metrics.json files produced by src.evaluation.evaluate and renders a
grouped-bar figure (one panel per metric, three representation groups, two runs each).
Subfeature bars carry p5-p95 whiskers from the 50-draw distribution.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/fmirkovic/Peptide-synthesis")
OLD = json.load(open(ROOT / "evaluation_output" / "metrics.json"))
NEW = json.load(open(ROOT / "evaluation_output_peptides_lt40zeros" / "metrics.json"))
# TabPFN embedding space was computed in a separate GPU-only pass on the filtered
# data; merge its robust/tabpfn/* keys into the filtered-run dict.
TABPFN = json.load(open(ROOT / "evaluation_output_tabpfn_lt40zeros" / "metrics.json"))
NEW.update({k: v for k, v in TABPFN.items() if k.startswith("robust/tabpfn/")})
# UMAP embedding space, computed in a separate pass (peptides env has umap-learn).
UMAP = json.load(open(ROOT / "evaluation_output_umap_lt40zeros" / "metrics.json"))
NEW.update({k: v for k, v in UMAP.items() if k.startswith("robust/umap/")})

RUNS = [("Original (21,571 feats)", OLD, "#4C78A8"),
        ("<40% zeros (1,199 feats)", NEW, "#F58518")]
REPS = ["identity", "pca", "subfeature", "umap", "tabpfn"]

METRICS = [
    ("dcr_median_dcr_ratio", "DCR median ratio\n(1 = safe, <1 = risk)"),
    ("dcr_privacy_at_risk", "DCR privacy-at-risk\n(fraction)"),
    ("reid_reidentification_rate", "Re-identification rate"),
    ("singling_out_univariate_risk", "Singling-out (univariate)"),
    ("linkability_risk", "Linkability risk"),
    ("mia_auc", "MIA AUC (0.5 = no risk)"),
]


def get_vals(metrics, metric):
    """Return (values, lower_err, upper_err) over the three representations."""
    vals, lo, hi = [], [], []
    for rep in REPS:
        if rep == "subfeature":
            m = metrics.get(f"robust/subfeat_{metric}_mean")
            p5 = metrics.get(f"robust/subfeat_{metric}_p5")
            p95 = metrics.get(f"robust/subfeat_{metric}_p95")
            vals.append(m)
            lo.append(None if m is None or p5 is None else max(0.0, m - p5))
            hi.append(None if m is None or p95 is None else max(0.0, p95 - m))
        else:
            vals.append(metrics.get(f"robust/{rep}/{metric}"))
            lo.append(None)
            hi.append(None)
    return vals, lo, hi


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
x = np.arange(len(REPS))
width = 0.38

for ax, (metric, title) in zip(axes.ravel(), METRICS):
    for i, (label, metrics, color) in enumerate(RUNS):
        vals, lo, hi = get_vals(metrics, metric)
        offset = (i - 0.5) * width
        plot_vals = [0 if v is None else v for v in vals]
        # asymmetric whiskers only where present (subfeature)
        yerr = np.array([[0 if l is None else l for l in lo],
                         [0 if h is None else h for h in hi]])
        has_err = any(l is not None for l in lo)
        ax.bar(x + offset, plot_vals, width, label=label, color=color,
               yerr=yerr if has_err else None, capsize=4,
               error_kw={"ecolor": "#333", "lw": 1.2})
        # mark representations not computed for a run (e.g. TabPFN in original).
        # y in axes fraction (via blended transform) so it never falls off-axis.
        for xi, v in zip(x, vals):
            if v is None:
                ax.text(xi + offset, 0.02, "n/a", rotation=90, ha="center",
                        va="bottom", fontsize=7, color="grey",
                        transform=ax.get_xaxis_transform())
    if metric == "dcr_median_dcr_ratio":
        ax.axhline(1.0, ls="--", lw=1, color="grey", zorder=0)
    if metric == "mia_auc":
        ax.axhline(0.5, ls="--", lw=1, color="grey", zorder=0)
        ax.set_ylim(0.45, max(0.56, ax.get_ylim()[1]))
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(REPS, fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.5)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Robust privacy: full feature space vs peptides with <40% zeros\n"
             "(subfeature bars = mean of 50 draws, whiskers = p5-p95)",
             fontsize=13, y=1.06)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out = ROOT / "evaluation_output_peptides_lt40zeros" / "robust_privacy_sidebyside.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")

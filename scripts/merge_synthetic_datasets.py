"""Merge synthetic datasets (HF, MAKE, NO_EVENT) into a single CSV.

For each group the imputed peptide matrix and the clinical table share the same
row order, so they are joined column-wise. The three groups share the same
columns, so they are then stacked row-wise. The clinical ``event_type`` column
(hf / ckd / no_event) identifies the source group.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE = Path("resources/synthetic_datasets")
GROUPS = {
    "HF": "hf_imputed.csv",
    "MAKE": "ckd_imputed.csv",
    "NO_EVENT": "ne_imputed.csv",
}
OUTPUT = BASE / "merged_synthetic.csv"


def merge_group(group: str, imputed_name: str) -> pd.DataFrame:
    """Join a group's imputed peptides and clinical table column-wise."""
    group_dir = BASE / group
    peptides = pd.read_csv(group_dir / imputed_name)
    clinical = pd.read_csv(group_dir / "clinical.csv")

    if len(peptides) != len(clinical):
        raise ValueError(
            f"{group}: row mismatch peptides={len(peptides)} clinical={len(clinical)}"
        )

    merged = pd.concat(
        [peptides.reset_index(drop=True), clinical.reset_index(drop=True)], axis=1
    )
    return merged


def main() -> None:
    frames = []
    for group, imputed_name in tqdm(GROUPS.items(), desc="Merging groups"):
        frames.append(merge_group(group, imputed_name))

    merged = pd.concat(frames, axis=0, ignore_index=True)
    merged.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} with {len(merged)} rows and {merged.shape[1]} columns")


if __name__ == "__main__":
    main()

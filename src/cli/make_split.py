#!/usr/bin/env python
"""
Split router feature parquet into train/val/test buckets.

By default we split on `prompt_id`, stratified by `category` (if present) and
use the 70/15/15 proportions that the legacy script hard-coded. The previous
implementation ignored CLI arguments; this version exposes them explicitly so
automation like `regenerate_and_train.sh` works as expected.

Example:
    python -m src.cli.make_split \
        --input artifacts/router/features_sum_fixed.parquet \
        --output artifacts/router/features_sum_fixed.with_splits.parquet \
        --split_by prompt_id \
        --train 0.7 --val 0.15 --test 0.15 \
        --seed 42
"""
from __future__ import annotations

import argparse
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


def _normalize_key(values: Tuple) -> Tuple:
    """Convert a tuple of values into a hashable key with NaNs mapped to None."""
    if not isinstance(values, tuple):
        values = (values,)
    return tuple(None if pd.isna(v) else v for v in values)


def _assign_splits(
    keys: Sequence[Tuple],
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> Dict[Tuple, str]:
    """Return mapping from key -> split label using floor allocation (legacy behaviour)."""
    n = len(keys)
    if n == 0:
        return {}
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    if n_train + n_val > n:
        raise ValueError(
            f"Invalid ratios: train={train_ratio}, val={val_ratio} produce counts "
            f"that exceed available groups ({n})."
        )
    n_test = n - n_train - n_val

    idx = np.arange(n)
    rng.shuffle(idx)

    boundary_train = n_train
    boundary_val = n_train + n_val

    assignment: Dict[Tuple, str] = {}
    for position, key_idx in enumerate(idx):
        key = keys[key_idx]
        if position < boundary_train:
            assignment[key] = "train"
        elif position < boundary_val:
            assignment[key] = "val"
        else:
            assignment[key] = "test"
    # Sanity: ensure counts line up
    assert sum(1 for v in assignment.values() if v == "train") == n_train
    assert sum(1 for v in assignment.values() if v == "val") == n_val
    assert sum(1 for v in assignment.values() if v == "test") == n_test
    return assignment


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to features parquet.")
    ap.add_argument("--output", required=True, help="Destination parquet path.")
    ap.add_argument(
        "--split_by",
        nargs="+",
        default=["prompt_id"],
        help="Columns whose combinations are kept within the same split.",
    )
    ap.add_argument(
        "--stratify",
        nargs="+",
        default=None,
        help="Optional columns to stratify over (default: ['category'] if present).",
    )
    ap.add_argument("--train", type=float, default=0.70, help="Train proportion.")
    ap.add_argument("--val", type=float, default=0.15, help="Validation proportion.")
    ap.add_argument("--test", type=float, default=0.15, help="Test proportion.")
    ap.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    ratios_sum = args.train + args.val + args.test
    if not np.isclose(ratios_sum, 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0; received train={args.train}, "
            f"val={args.val}, test={args.test} (sum={ratios_sum:.4f})."
        )

    df = pd.read_parquet(args.input)

    missing_split_cols = [c for c in args.split_by if c not in df.columns]
    if missing_split_cols:
        raise KeyError(f"--split_by columns missing from dataframe: {missing_split_cols}")

    if args.stratify is None:
        strat_cols = ["category"] if "category" in df.columns else []
    else:
        strat_cols = args.stratify

    missing_strat_cols = [c for c in strat_cols if c not in df.columns]
    if missing_strat_cols:
        raise KeyError(f"--stratify columns missing from dataframe: {missing_strat_cols}")

    rng = np.random.default_rng(args.seed)

    # Build frame of unique split groups (includes strat columns so we can stratify).
    unique_cols = args.split_by + strat_cols
    groups_df = df[unique_cols].drop_duplicates().reset_index(drop=True)

    assignments: Dict[Tuple, str] = {}
    if strat_cols:
        grouped = groups_df.groupby(strat_cols, dropna=False, sort=False)
    else:
        grouped = [((), groups_df)]

    for _, sub in grouped:
        keys = [
            _normalize_key(values)
            for values in sub[args.split_by].drop_duplicates().itertuples(index=False, name=None)
        ]
        assignments.update(_assign_splits(keys, args.train, args.val, rng))

    # Map back to main dataframe.
    split_keys = [
        _normalize_key(values)
        for values in df[args.split_by].itertuples(index=False, name=None)
    ]
    df["split"] = [assignments[key] for key in split_keys]

    # Sanity: each split group should appear exactly once.
    assert df.groupby(args.split_by)["split"].nunique().max() == 1

    counts = df["split"].value_counts().to_dict()
    print(f"split counts: {counts}")
    df.to_parquet(args.output, index=False)
    print(f"wrote: {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()

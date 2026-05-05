#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a combined features parquet by merging SUM and ATTN aggregations.

- Joins SUM and ATTN parquets on stable ids (prefer ["prompt_id","pair_id"]; fallback to
  ["prompt_id","movieA","movieB"]).
- Preserves all context columns (split, difficulty, category, mix_*, length_*, flags...),
  taking the union from both inputs without dropping anything present.
- Namespaces blocks as:
    sum_dz_alpha ... sum_dz_delta
    attn_dz_alpha ... attn_dz_delta
- Exposes a canonical dz_* (default = ATTN) so downstream MoE/RankNet continue to work.
- Adds “evidence shape” scalars on both families (computed from dz vectors):
    *_l1_abs, *_l2, *_max_abs, *_argmax_abs, *_pos_sum, *_neg_sum, *_entropy, *_concentration
- Adds cross-aggregation features:
    delta_dz_* (attn - sum), sim_attn_sum_dot, sim_attn_sum_cos
- Adds a few simple interactions on canonical dz: dz_axd, dz_axb, dz_bxg
- Leaves test labels/distribution untouched; this script never rebalances.
"""
from __future__ import annotations
import argparse
from typing import List
import numpy as np
import pandas as pd

# ----- config -----
EXPERTS = ["alpha", "beta", "gamma", "delta"]
KNOWN_CONTEXT = [
    "y","split","difficulty","category",
    "movieA","movieB",
    "mix_alpha","mix_beta","mix_gamma","mix_delta",
    "primary_expert","length_bucket","persona_style",
    "multi_intent","cold_user","has_genre_terms","has_negation","has_year",
    "length_words","num_genre_terms","agg_kind"
]
# ------------------


def _ensure_prefixed_dz(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Ensure columns are present as {prefix}_dz_alpha..delta.
    If plain dz_* exist, rename them to {prefix}_dz_* (non-destructive if already present).
    """
    df = df.copy()
    plain = [f"dz_{e}" for e in EXPERTS]
    want  = [f"{prefix}_dz_{e}" for e in EXPERTS]
    # If 'want' already present, keep them. Otherwise, source from 'plain' if available.
    if not all(c in df.columns for c in want):
        if all(c in df.columns for c in plain):
            ren = {f"dz_{e}": f"{prefix}_dz_{e}" for e in EXPERTS}
            df = df.rename(columns=ren)
        else:
            # if neither present, create zeros; fill later anyway for safety
            pass
    return df


def _pick_merge_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    # Prefer prompt+pair_id
    if all(k in df_a.columns and k in df_b.columns for k in ["prompt_id","pair_id"]):
        return ["prompt_id","pair_id"]
    # Fallback to prompt + (movieA,movieB)
    if all(k in df_a.columns and k in df_b.columns for k in ["prompt_id","movieA","movieB"]):
        return ["prompt_id","movieA","movieB"]
    raise ValueError("Cannot find merge keys. Need ['prompt_id','pair_id'] "
                     "or ['prompt_id','movieA','movieB'] in both inputs.")


def _evidence_shapes_block(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute simple shape stats from the per-pair dz vector (no chunk access required)."""
    req = [f"{prefix}_dz_{e}" for e in EXPERTS]
    for c in req:
        if c not in df.columns:
            df[c] = 0.0

    v = df[req].to_numpy(dtype=np.float32)            # [N,4]
    abs_v = np.abs(v)
    l1 = abs_v.sum(axis=1)
    l2 = np.sqrt((v * v).sum(axis=1) + 1e-12)
    max_abs = abs_v.max(axis=1)
    argmax_abs = abs_v.argmax(axis=1).astype(np.int16)
    pos_sum = np.clip(v, 0, None).sum(axis=1)
    neg_sum = np.clip(-v, 0, None).sum(axis=1)

    # entropy over normalized |dz| (add eps for stability)
    p = abs_v / np.clip(l1[:, None], 1e-9, None)
    p = np.clip(p, 1e-12, None)
    ent = -(p * np.log(p)).sum(axis=1)

    conc = max_abs / np.clip(l1, 1e-9, None)

    block = pd.DataFrame({
        f"{prefix}_l1_abs": l1,
        f"{prefix}_l2": l2,
        f"{prefix}_max_abs": max_abs,
        f"{prefix}_argmax_abs": argmax_abs,
        f"{prefix}_pos_sum": pos_sum,
        f"{prefix}_neg_sum": neg_sum,
        f"{prefix}_entropy": ent,
        f"{prefix}_concentration": conc,
    })
    return pd.concat([df.reset_index(drop=True), block], axis=1)


def _safe_union_context(sum_df, attn_df):
    KNOWN_CONTEXT = [
        "y","split","difficulty","category",
        "movieA","movieB",
        "mix_alpha","mix_beta","mix_gamma","mix_delta",
        "primary_expert","length_bucket","persona_style",
        "multi_intent","cold_user","has_genre_terms","has_negation","has_year",
        "length_words","num_genre_terms","agg_kind",
    ]
    present = [c for c in KNOWN_CONTEXT if (c in sum_df.columns) or (c in attn_df.columns)]

    def _meta_cols(df):
        return [c for c in df.columns
                if not (c.startswith("dz_") or c.startswith("sum_dz_") or c.startswith("attn_dz_"))]

    extra = []
    for c in _meta_cols(sum_df) + _meta_cols(attn_df):
        if c not in present and c not in extra:
            extra.append(c)

    return present + extra



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sum", required=True, help="Path to SUM features parquet")
    ap.add_argument("--attn", required=True, help="Path to ATTN features parquet (single τ). "
                                                  "For τ sweeps, run this script multiple times.")
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--canonical", default="attn", choices=["attn","sum"],
                    help="Which family becomes canonical dz_* used by downstream models.")
    ap.add_argument("--attn_tau", type=float, default=1.0,
                help="Softmax temperature for --agg_kind=attn (ignored for sum/max)")
    args = ap.parse_args()

    # Load
    df_sum = pd.read_parquet(args.sum)
    df_attn = pd.read_parquet(args.attn)

    # Normalize column names to have namespaced dz blocks
    df_sum  = _ensure_prefixed_dz(df_sum,  "sum")
    df_attn = _ensure_prefixed_dz(df_attn, "attn")

    # Choose merge keys
    keys = _pick_merge_keys(df_sum, df_attn)

    # Build keep-lists: union of contexts + id keys + dz blocks
    context_cols = _safe_union_context(df_sum, df_attn)
    # Always ensure y, split stick if present
    for must in ["y","split"]:
        if must in df_sum.columns or must in df_attn.columns:
            if must not in context_cols:
                context_cols.append(must)

    sum_block  = [c for c in df_sum.columns  if c.startswith("sum_dz_")]
    attn_block = [c for c in df_attn.columns if c.startswith("attn_dz_")]

    keep_sum  = sorted(set(keys + context_cols + sum_block))
    keep_attn = sorted(set(keys + attn_block))

    df_sum  = df_sum[keep_sum]
    df_attn = df_attn[keep_attn]

    # Merge one-to-one
    merged = df_sum.merge(df_attn, on=keys, how="inner", validate="one_to_one")
    n0, n1, n = len(df_sum), len(df_attn), len(merged)
    print(f"Merged records: SUM={n0}  ATTN={n1}  ->  {n}")

    # Ensure both families present (fill zeros if missing)
    for e in EXPERTS:
        merged.setdefault = None  # placate linters
        if f"sum_dz_{e}" not in merged.columns:
            merged[f"sum_dz_{e}"] = 0.0
        if f"attn_dz_{e}" not in merged.columns:
            merged[f"attn_dz_{e}"] = 0.0

    # Canonical dz_* for downstream
    def _pick_canonical(prefer: str) -> str:
        prefer = prefer if prefer in {"attn", "sum"} else "attn"
        fallback = "sum" if prefer == "attn" else "attn"
        block = [f"{prefer}_dz_{e}" for e in EXPERTS]
        if not all(c in merged.columns for c in block):
            msg = (f"[build_combo_features] Canonical block '{prefer}' missing columns; "
                   f"falling back to '{fallback}'.")
            print(msg)
            return fallback
        block_df = merged[block]
        if float(block_df.abs().to_numpy().sum()) == 0.0:
            msg = (f"[build_combo_features] Canonical block '{prefer}' appears to be all zeros; "
                   f"falling back to '{fallback}'.")
            print(msg)
            return fallback
        return prefer

    canon = _pick_canonical(args.canonical)
    for e in EXPERTS:
        merged[f"dz_{e}"] = merged[f"{canon}_dz_{e}"]

    # Cross-aggregation deltas & similarity
    attn_mat = merged[[f"attn_dz_{e}" for e in EXPERTS]].to_numpy(dtype=np.float32)
    sum_mat  = merged[[f"sum_dz_{e}"  for e in EXPERTS]].to_numpy(dtype=np.float32)
    dot = (attn_mat * sum_mat).sum(axis=1)
    n1 = np.sqrt((attn_mat**2).sum(axis=1) + 1e-12)
    n2 = np.sqrt((sum_mat**2).sum(axis=1) + 1e-12)
    merged["sim_attn_sum_dot"] = dot
    merged["sim_attn_sum_cos"] = dot / (n1 * n2)

    for e in EXPERTS:
        merged[f"delta_dz_{e}"] = merged[f"attn_dz_{e}"] - merged[f"sum_dz_{e}"]

    # Evidence shape features for both families
    merged = _evidence_shapes_block(merged, "attn")
    merged = _evidence_shapes_block(merged, "sum")

    # Simple canonical interactions
    merged["dz_axd"] = merged["dz_alpha"] * merged["dz_delta"]
    merged["dz_axb"] = merged["dz_alpha"] * merged["dz_beta"]
    merged["dz_bxg"] = merged["dz_beta"]  * merged["dz_gamma"]

    # Order columns nicely
    id_cols = [c for c in ["prompt_id","pair_id","movieA","movieB"] if c in merged.columns]
    front = list(dict.fromkeys(id_cols + [c for c in ["y","split","difficulty","category"] if c in merged.columns]))
    dz_cols = [f"dz_{e}" for e in EXPERTS]
    meta_cols = [c for c in KNOWN_CONTEXT if c in merged.columns and c not in front]
    sum_cols  = [c for c in merged.columns if c.startswith("sum_dz_")]
    attn_cols = [c for c in merged.columns if c.startswith("attn_dz_")]
    delta_cols= [c for c in merged.columns if c.startswith("delta_dz_")]
    shape_cols= [c for c in merged.columns if any(
                    c.startswith(p) for p in ["attn_l1_abs","attn_l2","attn_max_abs","attn_argmax_abs","attn_pos_sum",
                                              "attn_neg_sum","attn_entropy","attn_concentration",
                                              "sum_l1_abs","sum_l2","sum_max_abs","sum_argmax_abs","sum_pos_sum",
                                              "sum_neg_sum","sum_entropy","sum_concentration"])]
    sim_cols  = ["sim_attn_sum_dot","sim_attn_sum_cos"]
    inter_cols= ["dz_axd","dz_axb","dz_bxg"]

    ordered = front + dz_cols + meta_cols
    rest = [c for c in (sum_cols + attn_cols + delta_cols + shape_cols + sim_cols + inter_cols) if c not in ordered]
    merged = merged[ordered + rest]

    merged.to_parquet(args.out, index=False)
    print(f"Wrote: {args.out}")
    print(f"Columns: {len(merged.columns)}")
    print(f"Canonical dz source: {canon}")

if __name__ == "__main__":
    main()

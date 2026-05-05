#!/usr/bin/env python
"""Evaluate the trained RouterHead-XGB models."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .logger_utils import log_config, setup_router_logger
from .utils_xgb import (
    EXPERTS,
    build_design_matrix,
    evaluate_pairwise_metrics,
    load_or_build_targets,
    softmax,
    summarize_prompt_metrics,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RouterHead-XGB models.")
    parser.add_argument("--features_parquet", required=True, help="Path to router features parquet.")
    parser.add_argument("--models_dir", default="artifacts/router/xgb", help="Directory with trained XGB models.")
    parser.add_argument("--out_dir", default="artifacts/router/xgb_eval", help="Directory for evaluation outputs.")
    parser.add_argument("--logs_dir", default="logs", help="Where to write logs.")
    return parser.parse_args(argv)


def load_models(models_dir: Path) -> Dict[str, XGBRegressor]:
    models: Dict[str, XGBRegressor] = {}
    for expert in EXPERTS:
        booster = XGBRegressor()
        booster.load_model(models_dir / f"router_xgb_{expert}.json")
        models[expert] = booster
    return models


def gather_group_metrics(
    ids: Sequence,
    true_lookup: pd.DataFrame,
    pred_lookup: pd.DataFrame,
) -> Dict[str, float]:
    if not ids:
        return {}
    subset_true = true_lookup.loc[ids].to_numpy(dtype=np.float32)
    subset_pred = pred_lookup.loc[ids].to_numpy(dtype=np.float32)
    if subset_true.size == 0:
        return {}
    return summarize_prompt_metrics(subset_true, subset_pred)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    logger, log_file = setup_router_logger(args.logs_dir, name="router_xgb_eval")
    cfg = {
        "features_parquet": args.features_parquet,
        "models_dir": args.models_dir,
        "out_dir": args.out_dir,
    }
    log_config(logger, cfg, "Router XGB Eval Config")

    features_path = Path(args.features_parquet)
    pairs_df = pd.read_parquet(features_path)
    logger.info("Loaded %d pairs for evaluation", len(pairs_df))

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_path = models_dir / "scaler_and_schema.pkl"
    with open(schema_path, "rb") as f:
        payload = pickle.load(f)
    schema = payload["schema"]
    feature_cols = payload["feature_columns"]

    targets_df = load_or_build_targets(pairs_df, recompute=False, logger=logger)
    design_df, _, prompt_meta = build_design_matrix(pairs_df, targets_df, schema=schema, logger=logger)

    X = design_df[feature_cols].to_numpy(dtype=np.float32)
    true_weights = design_df[[f"w_{e}" for e in EXPERTS]].to_numpy(dtype=np.float32)

    models = load_models(models_dir)
    logits = np.column_stack([models[e].predict(X) for e in EXPERTS])
    pred_weights = softmax(logits, axis=1)

    pred_df = pd.DataFrame({"prompt_id": design_df["prompt_id"]})
    for idx, expert in enumerate(EXPERTS):
        pred_df[expert] = pred_weights[:, idx]

    prompt_summary = summarize_prompt_metrics(true_weights, pred_weights)
    pair_summary = evaluate_pairwise_metrics(pairs_df, pred_df)

    true_lookup = design_df.set_index("prompt_id")[[f"w_{e}" for e in EXPERTS]]
    pred_lookup = pred_df.set_index("prompt_id")[list(EXPERTS)]

    prompt_by_category: Dict[str, Dict[str, float]] = {}
    pair_by_category: Dict[str, Dict[str, float]] = {}
    for category, group in prompt_meta.groupby("category"):
        ids = list(group["prompt_id"])
        prompt_by_category[category] = gather_group_metrics(ids, true_lookup, pred_lookup)
        pair_by_category[category] = evaluate_pairwise_metrics(
            pairs_df[pairs_df["prompt_id"].isin(ids)],
            pred_df[pred_df["prompt_id"].isin(ids)],
        )

    prompt_by_difficulty: Dict[str, Dict[str, float]] = {}
    pair_by_difficulty: Dict[str, Dict[str, float]] = {}
    for diff, group in prompt_meta.groupby("difficulty_primary"):
        ids = list(group["prompt_id"])
        prompt_by_difficulty[diff] = gather_group_metrics(ids, true_lookup, pred_lookup)
        pair_by_difficulty[diff] = evaluate_pairwise_metrics(
            pairs_df[pairs_df["prompt_id"].isin(ids)],
            pred_df[pred_df["prompt_id"].isin(ids)],
        )

    metrics = {
        "prompt_overall": prompt_summary,
        "prompt_by_category": prompt_by_category,
        "prompt_by_difficulty": prompt_by_difficulty,
        "pair_overall": pair_summary,
        "pair_by_category": pair_by_category,
        "pair_by_difficulty": pair_by_difficulty,
    }
    metrics_path = out_dir / "metrics_eval.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation metrics to %s", metrics_path)

    preds_path = out_dir / "predictions.csv"
    final_df = pred_df.merge(design_df[["prompt_id", "w_alpha", "w_beta", "w_gamma", "w_delta"]], on="prompt_id")
    final_df.to_csv(preds_path, index=False)
    logger.info("Saved per-prompt predictions to %s", preds_path)

    print("\n=== RouterHead-XGB Evaluation ===")
    print(f"Prompts evaluated: {len(pred_df)}")
    print(
        f"Prompt metrics  KL={prompt_summary['kl']:.4f}  L1={prompt_summary['l1']:.4f}  "
        f"Cosine={prompt_summary['cosine']:.4f}"
    )
    print(
        f"Pair metrics    Acc={pair_summary['accuracy']:.3f}  "
        f"AUC={pair_summary['roc_auc']:.3f}  LogLoss={pair_summary['logloss']:.4f}"
    )
    print(f"Artifacts: metrics={metrics_path} predictions={preds_path}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()

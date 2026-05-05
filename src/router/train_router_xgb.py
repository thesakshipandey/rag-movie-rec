#!/usr/bin/env python
"""Train the XGBoost router head."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from tqdm import tqdm
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
    parser = argparse.ArgumentParser(description="Train the Router XGBoost head.")
    parser.add_argument("--features_parquet", required=True, help="Path to pair-level router features parquet.")
    parser.add_argument(
        "--out_dir",
        default="artifacts/router/xgb",
        help="Directory for trained models and design matrix.",
    )
    parser.add_argument("--config", default="configs/router_xgb.yaml", help="YAML file with model hyperparameters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--recompute_targets", action="store_true", help="Force recomputation of oracle weights.")
    parser.add_argument("--logs_dir", default="logs", help="Directory to store router logs.")
    return parser.parse_args(argv)


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"model": {}, "cv": {"n_splits": 5}}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("model", {})
    cfg.setdefault("cv", {"n_splits": 5})
    cfg["cv"].setdefault("n_splits", 5)
    return cfg


def prepare_folds(
    X: np.ndarray,
    meta: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = meta["category"].fillna("unknown").to_numpy()
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    if len(np.unique(groups)) >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X, None, groups=groups))
    else:
        entropy_bins = meta["entropy_bin"].to_numpy()
        if len(np.unique(entropy_bins)) > 1:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, entropy_bins))
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(splitter.split(X))
    return splits


def train_models(
    X: np.ndarray,
    y_logits: np.ndarray,
    params: Dict[str, Any],
    seed: int,
) -> Dict[str, XGBRegressor]:
    models: Dict[str, XGBRegressor] = {}
    for idx, expert in enumerate(EXPERTS):
        model = XGBRegressor(**params)
        model.set_params(random_state=seed + idx)
        model.fit(X, y_logits[:, idx])
        models[expert] = model
    return models


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    logger, log_file = setup_router_logger(args.logs_dir, name="router_xgb_train")
    cfg = load_config(args.config)
    log_config(logger, cfg, "Router XGB Config")

    features_path = Path(args.features_parquet)
    logger.info("Loading router features from %s", features_path)
    pairs_df = pd.read_parquet(features_path)
    logger.info("Loaded %d pairs (%d prompts)", len(pairs_df), pairs_df["prompt_id"].nunique())

    targets_df = load_or_build_targets(pairs_df, recompute=args.recompute_targets, logger=logger)

    design_df, schema, prompt_meta = build_design_matrix(pairs_df, targets_df, logger=logger)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    design_path = out_dir / "design_matrix.parquet"
    design_df.to_parquet(design_path, index=False)
    logger.info("Design matrix saved to %s (shape=%s)", design_path, design_df.shape)

    feature_cols = [c for c in design_df.columns if c.startswith("X_")]
    target_cols = [f"w_{e}" for e in EXPERTS]
    X = design_df[feature_cols].to_numpy(dtype=np.float32)
    Y = design_df[target_cols].to_numpy(dtype=np.float32)
    y_logits = np.log(np.clip(Y, 1e-6, 1.0))
    prompt_ids = design_df["prompt_id"].to_numpy()

    splits = prepare_folds(X, prompt_meta, cfg["cv"]["n_splits"], seed=args.seed)
    logger.info("Prepared %d-fold CV", len(splits))

    cv_prompt_metrics: List[Dict[str, float]] = []
    cv_pair_metrics: List[Dict[str, float]] = []
    weight_records = []
    for fold_id, (train_idx, val_idx) in enumerate(tqdm(splits, desc="folds")):
        logger.info("Training fold %d (%d train / %d val)", fold_id, len(train_idx), len(val_idx))
        models = train_models(X[train_idx], y_logits[train_idx], cfg["model"], seed=args.seed + fold_id * 10)
        val_logits = np.column_stack([models[e].predict(X[val_idx]) for e in EXPERTS])
        val_weights = softmax(val_logits, axis=1)
        true_weights = Y[val_idx]

        prompt_metric = summarize_prompt_metrics(true_weights, val_weights)
        cv_prompt_metrics.append(prompt_metric)

        pred_df = pd.DataFrame({"prompt_id": prompt_ids[val_idx]})
        for idx, expert in enumerate(EXPERTS):
            pred_df[expert] = val_weights[:, idx]

        fold_pairs = pairs_df[pairs_df["prompt_id"].isin(pred_df["prompt_id"])]
        pair_metric = evaluate_pairwise_metrics(fold_pairs, pred_df)
        cv_pair_metrics.append(pair_metric)

        record = pred_df.copy()
        for idx, expert in enumerate(EXPERTS):
            record[f"w_true_{expert}"] = true_weights[:, idx]
        record["fold_id"] = fold_id
        weight_records.append(record)

    metrics_path = out_dir / "metrics_cv.json"
    metrics_payload = {
        "folds": [
            {
                "fold": idx,
                "prompt_metrics": pm,
                "pair_metrics": qm,
            }
            for idx, (pm, qm) in enumerate(zip(cv_prompt_metrics, cv_pair_metrics))
        ],
        "aggregate": {
            "prompt_metrics": aggregate_metrics(cv_prompt_metrics),
            "pair_metrics": aggregate_metrics(cv_pair_metrics),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved CV metrics to %s", metrics_path)

    if weight_records:
        weights_path = out_dir / "weights_pred_val.csv"
        pd.concat(weight_records, ignore_index=True).to_csv(weights_path, index=False)
        logger.info("Saved validation predictions to %s", weights_path)

    # Train final models on all data
    final_models = train_models(X, y_logits, cfg["model"], seed=args.seed + 999)
    for expert, model in final_models.items():
        model.save_model(out_dir / f"router_xgb_{expert}.json")

    schema_path = out_dir / "scaler_and_schema.pkl"
    with open(schema_path, "wb") as f:
        pickle.dump(
            {
                "schema": schema,
                "feature_columns": feature_cols,
                "version": schema.version,
            },
            f,
        )
    logger.info("Persisted schema to %s", schema_path)

    acceptance = {
        "prompts": len(design_df),
        "folds": len(splits),
        "features": len(feature_cols),
        "metrics_prompt": metrics_payload["aggregate"]["prompt_metrics"],
        "metrics_pair": metrics_payload["aggregate"]["pair_metrics"],
        "models_dir": str(out_dir),
        "schema_path": str(schema_path),
        "metrics_path": str(metrics_path),
        "design_matrix": str(design_path),
    }
    logger.info("Acceptance checklist:\n%s", json.dumps(acceptance, indent=2))
    print("\n=== RouterHead-XGB Acceptance ===")
    print(f"Prompts: {acceptance['prompts']} | Folds: {acceptance['folds']} | Features: {acceptance['features']}")
    pm = acceptance["metrics_prompt"]
    qm = acceptance["metrics_pair"]
    print(
        "Prompt metrics  "
        f"KL={pm.get('kl_mean', float('nan')):.4f}±{pm.get('kl_std', 0.0):.4f}  "
        f"L1={pm.get('l1_mean', float('nan')):.4f}±{pm.get('l1_std', 0.0):.4f}  "
        f"Cos={pm.get('cosine_mean', float('nan')):.4f}±{pm.get('cosine_std', 0.0):.4f}"
    )
    print(
        "Pair metrics    "
        f"Acc={qm.get('accuracy_mean', float('nan')):.3f}±{qm.get('accuracy_std', 0.0):.3f}  "
        f"AUC={qm.get('roc_auc_mean', float('nan')):.3f}±{qm.get('roc_auc_std', 0.0):.3f}  "
        f"LogLoss={qm.get('logloss_mean', float('nan')):.4f}±{qm.get('logloss_std', 0.0):.4f}"
    )
    print(f"Artifacts: models={acceptance['models_dir']} metrics={metrics_path} schema={schema_path}")
    print(f"Log file: {log_file}")


def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    aggregated: Dict[str, float] = {}
    if not metric_list:
        return aggregated
    keys = metric_list[0].keys()
    for key in keys:
        vals = np.array([m[key] for m in metric_list if key in m], dtype=float)
        aggregated[f"{key}_mean"] = float(np.mean(vals))
        aggregated[f"{key}_std"] = float(np.std(vals))
    return aggregated


if __name__ == "__main__":
    main()

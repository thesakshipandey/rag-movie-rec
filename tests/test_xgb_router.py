from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.router import eval_router_xgb, train_router_xgb  # noqa: E402
from src.router.utils_xgb import EXPERTS, PLUTCHIK_ORDER  # noqa: E402
from src.router.xgb_router import RouterHeadXGB  # noqa: E402


def build_synthetic_features(path: Path, n_prompts: int = 40, pairs_per_prompt: int = 8) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    categories = ["action", "drama", "family", "sci-fi"]
    difficulties = ["easy", "medium", "hard"]
    for pid in range(n_prompts):
        category = categories[pid % len(categories)]
        difficulty = difficulties[pid % len(difficulties)]
        plutchik = rng.dirichlet(np.ones(8))
        mix = rng.dirichlet(np.ones(4))
        length_bucket = ["short", "medium", "long"][pid % 3]
        persona_style = ["casual", "formal"][pid % 2]
        primary_expert = ["dense", "bm25", "emo", "lgcn"][pid % 4]
        for pair in range(pairs_per_prompt):
            dz = rng.normal(loc=0.2, scale=0.5, size=4)
            score = float(np.dot(mix, dz) + rng.normal(scale=0.1))
            y = int(score > 0)
            rows.append(
                {
                    "prompt_id": pid,
                    "pair_id": f"{pid}_{pair}",
                    "y": y,
                    "dz_alpha": dz[0],
                    "dz_beta": dz[1],
                    "dz_gamma": dz[2],
                    "dz_delta": dz[3],
                    "category": category,
                    "difficulty": difficulty,
                    "plutchik_dist": json.dumps({emo: float(val) for emo, val in zip(PLUTCHIK_ORDER, plutchik)}),
                    "mix_weights.alpha": mix[0],
                    "mix_weights.beta": mix[1],
                    "mix_weights.gamma": mix[2],
                    "mix_weights.delta": mix[3],
                    "length_bucket": length_bucket,
                    "persona_style": persona_style,
                    "primary_expert": primary_expert,
                    "multi_intent": float(pid % 2 == 0),
                    "cold_user": float(pid % 3 == 0),
                    "has_genre_terms": 1.0,
                    "has_negation": float(pair % 2 == 0),
                    "has_year": 1.0,
                    "length_words": 120 + pid,
                    "num_genre_terms": 3 + pid % 4,
                }
            )
    df = pd.DataFrame(rows)
    out_path = path / "features.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def write_config(path: Path) -> Path:
    cfg = {
        "model": {
            "n_estimators": 15,
            "max_depth": 3,
            "learning_rate": 0.2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "min_child_weight": 1.0,
            "tree_method": "hist",
        },
        "cv": {"n_splits": 3},
    }
    cfg_path = path / "router_test.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return cfg_path


def test_router_xgb_pipeline(tmp_path: Path):
    features = build_synthetic_features(tmp_path)
    config = write_config(tmp_path)
    out_dir = tmp_path / "router_models"
    log_dir = tmp_path / "logs"

    train_router_xgb.main(
        [
            "--features_parquet",
            str(features),
            "--out_dir",
            str(out_dir),
            "--config",
            str(config),
            "--seed",
            "7",
            "--recompute_targets",
            "--logs_dir",
            str(log_dir),
        ]
    )

    assert (out_dir / "router_xgb_alpha.json").exists()
    assert (out_dir / "metrics_cv.json").exists()
    assert (out_dir / "design_matrix.parquet").exists()

    eval_dir = tmp_path / "router_eval"
    eval_router_xgb.main(
        [
            "--features_parquet",
            str(features),
            "--models_dir",
            str(out_dir),
            "--out_dir",
            str(eval_dir),
            "--logs_dir",
            str(log_dir),
        ]
    )
    assert (eval_dir / "metrics_eval.json").exists()

    router = RouterHeadXGB.load(out_dir)
    design_df = pd.read_parquet(out_dir / "design_matrix.parquet")
    feature_cols = [c for c in design_df.columns if c.startswith("X_")]
    sample = design_df.head(5)[["prompt_id", *feature_cols]]
    preds = router.predict_many(sample)
    assert np.allclose(preds[list(EXPERTS)].sum(axis=1), 1.0, atol=1e-6)

    single = router.predict_weights(sample.iloc[0:1])
    assert pytest.approx(sum(single.values()), rel=1e-6) == 1.0

    with open(out_dir / "metrics_cv.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    acc_mean = metrics["aggregate"]["pair_metrics"]["accuracy_mean"]
    assert acc_mean >= 0.6

"""Utilities for building prompt-level design matrices for the XGBoost router."""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

EXPERTS = ["alpha", "beta", "gamma", "delta"]
PLUTCHIK_ORDER = [
    "Joy",
    "Trust",
    "Fear",
    "Anticipation",
    "Sadness",
    "Anger",
    "Surprise",
    "Disgust",
]
SCHEMA_VERSION = "xgb_v1"
DEFAULT_TARGETS_PATH = Path("artifacts/router/targets/router_oracle_weights.csv")

CONTEXT_FLAG_COLS = ["multi_intent", "cold_user", "has_genre_terms", "has_negation", "has_year"]
CONTEXT_FLOAT_COLS = ["length_words", "num_genre_terms"]
CATEGORY_COLS = ["category", "length_bucket", "persona_style", "primary_expert"]
META_COLS = ["prompt_id", "category", "length_bucket", "persona_style", "primary_expert", "difficulty_primary"]


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _normalize_str(val: Any, default: str = "unknown") -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return str(val).strip().lower() or default


def _first_non_null(series: pd.Series) -> Any:
    if series is None:
        return None
    for v in series:
        if pd.notna(v):
            return v
    return None


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    arr = np.asarray(arr, dtype=np.float32)
    max_val = np.max(arr, axis=axis, keepdims=True)
    exp = np.exp(arr - max_val)
    denom = np.sum(exp, axis=axis, keepdims=True) + 1e-9
    return exp / denom


def parse_plutchik(value: Any) -> np.ndarray:
    """Parse a plutchik distribution into an 8-dim simplex."""
    if isinstance(value, (tuple, list)) and len(value) == 2 and not isinstance(value[0], (dict, str)):
        value = value[0]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    if isinstance(value, dict):
        if "emotions" in value and isinstance(value["emotions"], dict):
            value = value["emotions"]
        vec = [float(value.get(k, 0.0)) for k in PLUTCHIK_ORDER]
        return _normalize_simplex(np.asarray(vec, dtype=np.float32))
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 8:
            return _normalize_simplex(arr)
    return np.full(8, 1.0 / 8, dtype=np.float32)


def _normalize_simplex(vec: np.ndarray) -> np.ndarray:
    vec = np.clip(vec, 0.0, None)
    denom = float(vec.sum())
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full(8, 1.0 / 8, dtype=np.float32)
    return (vec / denom).astype(np.float32)


def plutchik_entropy(vec: np.ndarray) -> float:
    vec = np.clip(vec, 1e-9, 1.0)
    return float(-np.sum(vec * np.log(vec)))


def infer_difficulty_levels(df: pd.DataFrame) -> List[str]:
    if "difficulty" not in df.columns:
        return ["easy", "medium", "hard", "unknown"]
    series = df["difficulty"].dropna()
    if series.empty:
        levels = ["easy", "medium", "hard"]
    else:
        levels = sorted({_normalize_str(v) for v in series if str(v).strip()})
    if not levels:
        levels = ["easy", "medium", "hard"]
    if "unknown" not in levels:
        levels.append("unknown")
    return levels


def _difficulty_histogram(group: pd.DataFrame, levels: Sequence[str]) -> Tuple[Dict[str, float], str]:
    total = len(group)
    counts = {lvl: 0.0 for lvl in levels}
    if "difficulty" in group.columns and total > 0:
        vals = group["difficulty"].apply(_normalize_str)
        for lvl, cnt in vals.value_counts().items():
            if lvl in counts:
                counts[lvl] += float(cnt) / total
    counts_sum = sum(counts.values())
    if counts_sum < 1.0:
        counts["unknown"] = counts.get("unknown", 0.0) + (1.0 - counts_sum)
    primary = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "unknown"
    return counts, primary


def aggregate_prompt_features(
    df: pd.DataFrame,
    difficulty_levels: Sequence[str],
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pid, group in df.groupby("prompt_id"):
        row: Dict[str, Any] = {"prompt_id": pid}
        for col in CATEGORY_COLS:
            raw = _first_non_null(group[col]) if col in group.columns else None
            row[col] = _normalize_str(raw)

        diff_hist, diff_primary = _difficulty_histogram(group, difficulty_levels)
        for lvl, frac in diff_hist.items():
            row[f"difficulty_frac_{_slugify(lvl)}"] = float(frac)
        row["difficulty_primary"] = diff_primary

        row["pairs_per_prompt"] = float(len(group))
        if "y" in group.columns and len(group):
            row["share_positive_pairs"] = float(group["y"].astype(float).mean())
        else:
            row["share_positive_pairs"] = 0.5

        for flag in CONTEXT_FLAG_COLS:
            if flag in group.columns:
                vals = group[flag].dropna().astype(float)
                row[flag] = float(vals.mean()) if not vals.empty else 0.0
            else:
                row[flag] = 0.0

        for flt in CONTEXT_FLOAT_COLS:
            raw = _first_non_null(group[flt]) if flt in group.columns else 0.0
            row[flt] = float(raw) if raw is not None and not pd.isna(raw) else 0.0

        pemo = None
        if "plutchik_dist" in group.columns:
            pemo = _first_non_null(group["plutchik_dist"])
        vec = parse_plutchik(pemo)
        for idx, emo in enumerate(PLUTCHIK_ORDER):
            row[f"plutchik_{_slugify(emo)}"] = float(vec[idx])
        row["plutchik_entropy"] = plutchik_entropy(vec)

        for expert in EXPERTS:
            col = f"dz_{expert}"
            if col in group.columns:
                values = group[col].astype(float).to_numpy()
            else:
                values = np.zeros(len(group), dtype=np.float32)
            stats = _dz_stats(values)
            for key, val in stats.items():
                row[f"{col}_{key}"] = val

        rows.append(row)

    prompt_df = pd.DataFrame(rows)
    num_cols = prompt_df.select_dtypes(include=[np.number]).columns
    prompt_df[num_cols] = prompt_df[num_cols].fillna(0.0)
    prompt_df.sort_values("prompt_id", inplace=True)
    prompt_df.reset_index(drop=True, inplace=True)
    if logger:
        logger.info("Aggregated %d prompts into design-matrix rows", len(prompt_df))
    return prompt_df


def _dz_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "pos_frac": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
        }
    vals = values.astype(np.float32)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "pos_frac": float(np.mean(vals > 0)),
        "q10": float(np.quantile(vals, 0.10)),
        "q50": float(np.quantile(vals, 0.50)),
        "q90": float(np.quantile(vals, 0.90)),
    }


@dataclass
class RouterFeatureSchema:
    """Holds metadata to transform prompt-level frames into ML-ready tensors."""

    version: str = SCHEMA_VERSION
    category_levels: List[str] | None = None
    length_bucket_levels: List[str] | None = None
    persona_style_levels: List[str] | None = None
    primary_expert_levels: List[str] | None = None
    difficulty_levels: List[str] | None = None
    base_numeric_columns: List[str] | None = None
    feature_columns: List[str] | None = None
    meta_columns: Tuple[str, ...] = field(default_factory=lambda: tuple(META_COLS))

    def fit(self, prompt_df: pd.DataFrame) -> RouterFeatureSchema:
        self.category_levels = self.category_levels or _prepare_levels(prompt_df["category"])
        self.length_bucket_levels = self.length_bucket_levels or _prepare_levels(prompt_df["length_bucket"])
        self.persona_style_levels = self.persona_style_levels or _prepare_levels(prompt_df["persona_style"])
        self.primary_expert_levels = self.primary_expert_levels or _prepare_levels(prompt_df["primary_expert"])

        if self.base_numeric_columns is None:
            exclude = set(self.meta_columns)
            exclude.add("prompt_id")
            numeric = [c for c in prompt_df.columns if c not in exclude]
            self.base_numeric_columns = sorted(numeric)

        feature_cols: List[str] = []
        feature_cols.extend(_one_hot_names("category", self.category_levels))
        feature_cols.extend(_one_hot_names("length_bucket", self.length_bucket_levels))
        feature_cols.extend(_one_hot_names("persona_style", self.persona_style_levels))
        feature_cols.extend(_one_hot_names("primary_expert", self.primary_expert_levels))
        feature_cols.extend([f"X_{col}" for col in self.base_numeric_columns])
        self.feature_columns = feature_cols
        return self

    def transform(self, prompt_df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_columns or not self.base_numeric_columns:
            raise ValueError("Schema must be fitted before calling transform().")

        df = prompt_df.copy()
        out = pd.DataFrame({"prompt_id": df["prompt_id"]})

        out = _append_one_hot(out, df["category"], self.category_levels, "category")
        out = _append_one_hot(out, df["length_bucket"], self.length_bucket_levels, "length_bucket")
        out = _append_one_hot(out, df["persona_style"], self.persona_style_levels, "persona_style")
        out = _append_one_hot(out, df["primary_expert"], self.primary_expert_levels, "primary_expert")

        for col in self.base_numeric_columns:
            name = f"X_{col}"
            if col in df.columns:
                out[name] = df[col].fillna(0.0).astype(np.float32)
            else:
                out[name] = 0.0

        ordered = ["prompt_id"] + self.feature_columns
        return out.reindex(columns=ordered, fill_value=0.0)


def _prepare_levels(series: pd.Series) -> List[str]:
    vals = {_normalize_str(v) for v in series if pd.notna(v)}
    vals.discard("unknown")
    levels = sorted(vals)
    levels.insert(0, "unknown")
    return levels


def _one_hot_names(prefix: str, levels: Sequence[str] | None) -> List[str]:
    if not levels:
        return []
    return [f"X_{prefix}__{_slugify(lvl)}" for lvl in levels]


def _append_one_hot(
    df: pd.DataFrame,
    series: pd.Series,
    levels: Sequence[str] | None,
    prefix: str,
) -> pd.DataFrame:
    if not levels:
        return df
    series = series.apply(_normalize_str)
    for lvl in levels:
        col = f"X_{prefix}__{_slugify(lvl)}"
        df[col] = (series == lvl).astype(np.float32)
    return df


def build_design_matrix(
    pairs_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    schema: RouterFeatureSchema | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, RouterFeatureSchema, pd.DataFrame]:
    schema = schema or RouterFeatureSchema()
    if schema.difficulty_levels is None:
        schema.difficulty_levels = infer_difficulty_levels(pairs_df)
    prompt_df = aggregate_prompt_features(pairs_df, schema.difficulty_levels, logger=logger)
    schema.fit(prompt_df)
    features_df = schema.transform(prompt_df)

    merged = features_df.merge(targets_df, on="prompt_id", how="inner", validate="one_to_one")
    meta = prompt_df[["prompt_id", "category", "difficulty_primary", "plutchik_entropy"]].copy()
    meta["entropy_bin"] = _entropy_bins(meta["plutchik_entropy"])
    return merged, schema, meta


def _entropy_bins(entropy: pd.Series, n_bins: int = 5) -> np.ndarray:
    entropy = entropy.fillna(entropy.mean() if not entropy.empty else 0.0)
    if entropy.nunique() <= 1:
        return np.zeros(len(entropy), dtype=int)
    try:
        bins = pd.qcut(
            entropy,
            q=min(n_bins, entropy.nunique()),
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return np.zeros(len(entropy), dtype=int)
    return bins.astype(int).to_numpy()


def load_or_build_targets(
    pairs_df: pd.DataFrame,
    cache_path: Path = DEFAULT_TARGETS_PATH,
    recompute: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists() and not recompute:
        if logger:
            logger.info("Loading cached router targets from %s", cache_path)
        return pd.read_csv(cache_path)

    targets = compute_prompt_targets(pairs_df, logger=logger)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    targets.to_csv(cache_path, index=False)
    if logger:
        logger.info("Wrote %d prompt targets to %s", len(targets), cache_path)
    return targets


def compute_prompt_targets(
    pairs_df: pd.DataFrame,
    min_pairs: int = 6,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for pid, group in pairs_df.groupby("prompt_id"):
        weights, source, auc = _resolve_prompt_weights(group, min_pairs=min_pairs, logger=logger)
        rows.append(
            {
                "prompt_id": pid,
                "w_alpha": weights[0],
                "w_beta": weights[1],
                "w_gamma": weights[2],
                "w_delta": weights[3],
                "source": source,
                "pairs_used": int(len(group)),
                "auc_fit": auc,
            }
        )
    out = pd.DataFrame(rows).sort_values("prompt_id").reset_index(drop=True)
    return out


def _resolve_prompt_weights(
    group: pd.DataFrame,
    min_pairs: int,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, str, float]:
    mix = _extract_mix_weights(group)
    if mix is not None:
        return mix, "mix", float("nan")

    weights, auc = _fit_prompt_logit(group, C=5.0)
    if _bad_fit(weights, auc, len(group), min_pairs):
        weights, auc = _fit_prompt_logit(group, C=1.0)
    if _bad_fit(weights, auc, len(group), min_pairs):
        if logger:
            logger.debug("Prompt %s fell back to uniform weights.", group["prompt_id"].iloc[0])
        weights = np.full(4, 0.25, dtype=np.float32)
        return weights, "uniform", float(auc) if auc is not None else float("nan")
    return weights, "logit", float(auc) if auc is not None else float("nan")


def _extract_mix_weights(group: pd.DataFrame) -> np.ndarray | None:
    col_sets = [
        [f"mix_weights.{e}" for e in EXPERTS],
        [f"mix_{e}" for e in EXPERTS],
    ]
    for cols in col_sets:
        if all(c in group.columns for c in cols):
            values = []
            for c in cols:
                s = group[c].dropna()
                values.append(float(s.iloc[0]) if not s.empty else None)
            arr = np.array([v if v is not None else 0.0 for v in values], dtype=np.float32)
            arr = np.clip(arr, 0.0, None)
            total = float(arr.sum())
            if total <= 0.0:
                return None
            return arr / total
    return None


def _fit_prompt_logit(group: pd.DataFrame, C: float) -> tuple[np.ndarray | None, float | None]:
    if not all(f"dz_{e}" in group.columns for e in EXPERTS):
        return None, None
    X = group[[f"dz_{e}" for e in EXPERTS]].to_numpy(dtype=np.float32)
    y = group["y"].to_numpy(dtype=np.int8)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        return None, None
    model = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        fit_intercept=False,
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(X, y)
    coef = model.coef_.reshape(-1)
    probs = model.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = None
    weights = _softplus(coef)
    denom = float(weights.sum()) + 1e-8
    weights = (weights / denom).astype(np.float32)
    return weights, auc


def _bad_fit(weights: np.ndarray | None, auc: float | None, pairs: int, min_pairs: int) -> bool:
    if weights is None or not np.all(np.isfinite(weights)):
        return True
    if pairs < min_pairs:
        return True
    if np.allclose(weights, weights.mean()):
        return True
    if auc is None:
        return True
    return auc < 0.55


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


def cosine_similarity(p: np.ndarray, q: np.ndarray) -> float:
    denom = np.linalg.norm(p) * np.linalg.norm(q) + 1e-9
    return float(np.dot(p, q) / denom)


def summarize_prompt_metrics(true_w: np.ndarray, pred_w: np.ndarray) -> Dict[str, float]:
    kls = [kl_divergence(t, p) for t, p in zip(true_w, pred_w)]
    l1s = [l1_distance(t, p) for t, p in zip(true_w, pred_w)]
    coss = [cosine_similarity(t, p) for t, p in zip(true_w, pred_w)]
    return {
        "kl": float(np.mean(kls)),
        "l1": float(np.mean(l1s)),
        "cosine": float(np.mean(coss)),
    }


def evaluate_pairwise_metrics(pairs_df: pd.DataFrame, pred_df: pd.DataFrame) -> Dict[str, float]:
    merged = pairs_df.merge(pred_df, on="prompt_id", how="inner")
    if merged.empty:
        return {"accuracy": float("nan"), "roc_auc": float("nan"), "logloss": float("nan")}
    dz_mat = merged[[f"dz_{e}" for e in EXPERTS]].to_numpy(dtype=np.float32)
    weight_mat = merged[list(EXPERTS)].to_numpy(dtype=np.float32)
    scores = np.sum(dz_mat * weight_mat, axis=1)
    probs = 1.0 / (1.0 + np.exp(-scores))
    y_true = merged["y"].to_numpy(dtype=np.int32)
    preds = (scores > 0).astype(int)
    acc = float(accuracy_score(y_true, preds))
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    logloss_val = float(log_loss(y_true, clipped))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "roc_auc": auc, "logloss": logloss_val}

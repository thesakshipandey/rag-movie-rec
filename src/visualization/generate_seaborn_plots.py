#!/usr/bin/env python
"""Generate polished seaborn plots from evaluation artifacts.

This utility expects the comprehensive evaluation artifacts produced by
``python -m src.evaluations.comprehensive_eval``. It converts the CSV/JSON
outputs into tidy pandas DataFrames and renders publication-ready seaborn
figures summarising model performance, dataset coverage, expert agreement,
error patterns, and feature correlations.

Example (from repo root)::

    /mnt/nas/sakshipandey/main/venvs/rag_recsys/bin/python \
        -m src.visualization.generate_seaborn_plots \
        --evaluation-dir artifacts/evaluation_results \
        --output-dir artifacts/data_evaluation \
        --features-parquet artifacts/router/features_sum_fixed.with_splits.bal.parquet

The script creates the output directory if needed and overwrites plots with the
same name. Figures are saved as PNG files.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


LOGGER = logging.getLogger(__name__)


def _set_plot_theme() -> None:
    """Configure matplotlib/seaborn aesthetics for consistent styling."""

    sns.set_theme(
        style="whitegrid",
        context="talk",
        palette="deep",
    )
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        LOGGER.warning("Missing CSV: %s", path)
        return None
    return pd.read_csv(path)


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        LOGGER.warning("Missing JSON: %s", path)
        return None
    with path.open() as handle:
        return json.load(handle)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved plot → %s", output_path)


def _annotate_barplot(
    ax: plt.Axes,
    *,
    orientation: str,
    labels: Optional[list[str]] = None,
    fmt: str = "{:.3f}",
) -> None:
    bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    if not bars:
        return

    if orientation not in {"h", "v"}:
        raise ValueError("orientation must be 'h' or 'v'")

    if labels is None:
        if orientation == "h":
            values = [bar.get_width() for bar in bars]
        else:
            values = [bar.get_height() for bar in bars]
        labels = [fmt.format(val) for val in values]

    if orientation == "h":
        offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
    else:
        offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01

    for bar, label in zip(bars, labels):
        if orientation == "h":
            x = bar.get_width() + offset
            y = bar.get_y() + bar.get_height() / 2
            ax.text(x, y, label, va="center", ha="left", fontsize=10)
        else:
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height() + offset
            ax.text(x, y, label, va="bottom", ha="center", fontsize=10)


def plot_overall_model_performance(df: pd.DataFrame, output_path: Path) -> None:
    if df is None or df.empty:
        LOGGER.warning("Skipping overall model performance plot (empty dataframe).")
        return

    metric = "agree_ties_0p5"
    plot_df = df.dropna(subset=[metric]).copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    plot_df = plot_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(plot_df))))
    sns.barplot(
        data=plot_df,
        y="model",
        x=metric,
        hue="model",
        ax=ax,
        palette=sns.color_palette("viridis", n_colors=len(plot_df)),
        dodge=False,
        legend=False,
    )

    ax.set_title("Model Agreement (ties=0.5)", fontweight="bold")
    ax.set_xlabel("Agreement Score")
    ax.set_ylabel("Model")
    ax.set_xlim(0.0, 1.0)
    _annotate_barplot(ax, orientation="h")
    baseline = ax.axvline(0.5, color="red", linestyle="--", alpha=0.4, label="Random baseline")
    ax.legend(handles=[baseline], loc="lower right")

    _save(fig, output_path)


def plot_agreement_scatter(df: pd.DataFrame, output_path: Path) -> None:
    if df is None or df.empty:
        LOGGER.warning("Skipping agreement scatter plot (empty dataframe).")
        return

    metrics = ["agree_no_ties", "agree_ties_0p5"]
    plot_df = df.dropna(subset=metrics).copy()
    for metric in metrics:
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=metrics)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="agree_no_ties",
        y="agree_ties_0p5",
        hue="model",
        s=120,
        ax=ax,
    )

    # Annotate each point with the model name for clarity
    for _, row in plot_df.iterrows():
        ax.text(
            row["agree_no_ties"] + 0.002,
            row["agree_ties_0p5"] + 0.002,
            row["model"],
            fontsize=9,
        )

    ax.set_title("Agreement Metrics Correlation", fontweight="bold")
    ax.set_xlabel("Agreement (ties excluded)")
    ax.set_ylabel("Agreement (ties=0.5)")
    ax.set_xlim(0.45, min(1.0, plot_df["agree_no_ties"].max() + 0.05))
    ax.set_ylim(0.45, min(1.0, plot_df["agree_ties_0p5"].max() + 0.05))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.4)
    ax.legend(loc="lower right", title="Model")

    _save(fig, output_path)


def _prepare_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    matrix = df.copy()
    if "model" in matrix.columns:
        mask = matrix["model"].astype(str).str.startswith("dataset_stats")
        matrix = matrix[~mask]
        matrix = matrix.set_index("model")

    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.dropna(how="all", axis=1)
    matrix = matrix.dropna(how="all", axis=0)

    if matrix.empty:
        return None
    return matrix


def plot_category_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    matrix = _prepare_matrix(df)
    if matrix is None:
        LOGGER.warning("Skipping category heatmap (no data).")
        return

    fig_width = max(10, 1.2 * matrix.shape[1])
    fig_height = max(6, 0.6 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="crest",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Agreement"},
        ax=ax,
    )

    ax.set_title("Performance by Prompt Category", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Model")
    plt.xticks(rotation=45, ha="right")

    _save(fig, output_path)


def plot_difficulty_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    matrix = _prepare_matrix(df)
    if matrix is None:
        LOGGER.warning("Skipping difficulty heatmap (no data).")
        return

    fig, ax = plt.subplots(figsize=(8, max(6, 0.6 * matrix.shape[0])))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Agreement"},
        ax=ax,
    )

    ax.set_title("Performance by Difficulty", fontweight="bold")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Model")

    _save(fig, output_path)


def plot_expert_performance(metrics_dir: Path, output_path: Path) -> None:
    if not metrics_dir.exists():
        LOGGER.warning("Skipping expert performance plot (missing directory).")
        return

    rows = []
    for json_path in sorted(metrics_dir.glob("*_metrics.json")):
        metrics = _load_json(json_path)
        if not metrics:
            continue
        overall = metrics.get("overall", {})
        rows.append(
            {
                "expert": json_path.stem.replace("_metrics", ""),
                "agree_ties_0p5": overall.get("agree_ties_0p5"),
                "agree_no_ties": overall.get("agree_no_ties"),
                "correct": overall.get("correct"),
                "incorrect": overall.get("incorrect"),
                "ties": overall.get("ties"),
            }
        )

    if not rows:
        LOGGER.warning("No expert metrics loaded; skipping plot.")
        return

    df = pd.DataFrame(rows).sort_values("agree_ties_0p5", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="agree_ties_0p5",
        y="expert",
        hue="expert",
        palette=sns.color_palette("flare", n_colors=len(df)),
        ax=ax,
        dodge=False,
        legend=False,
    )
    ax.set_title("Expert Agreement (ties=0.5)", fontweight="bold")
    ax.set_xlabel("Agreement Score")
    ax.set_ylabel("Expert")
    ax.set_xlim(0.0, 1.0)
    _annotate_barplot(ax, orientation="h")

    _save(fig, output_path)


def plot_category_distribution(summary: Dict, output_path: Path) -> None:
    if not summary:
        LOGGER.warning("Skipping category distribution plot (missing summary).")
        return

    prompts = summary.get("prompts", {})
    counts = prompts.get("category_counts") or prompts.get("category_distribution")
    if not counts:
        LOGGER.warning("No category counts in summary; skipping plot.")
        return

    df = (
        pd.Series(counts, name="count")
        .reset_index()
        .rename(columns={"index": "category"})
        .sort_values("count", ascending=False)
    )
    df["percentage"] = 100 * df["count"] / df["count"].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df,
        x="category",
        y="count",
        hue="category",
        palette=sns.color_palette("deep", n_colors=len(df)),
        ax=ax,
        dodge=False,
        legend=False,
    )
    ax.set_title("Prompt Category Coverage", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    labels = [f"{c:,.0f}\n({p:.1f}%)" for c, p in zip(df["count"], df["percentage"])]
    _annotate_barplot(ax, orientation="v", labels=labels)
    plt.xticks(rotation=45, ha="right")

    _save(fig, output_path)


def plot_split_distribution(summary: Dict, output_path: Path) -> None:
    if not summary:
        LOGGER.warning("Skipping split distribution plot (missing summary).")
        return

    pairs = summary.get("pairs", {})
    splits = pairs.get("split_distribution")
    if not splits:
        LOGGER.warning("No split distribution available; skipping plot.")
        return

    df = (
        pd.Series(splits, name="count")
        .reset_index()
        .rename(columns={"index": "split"})
        .sort_values("count", ascending=False)
    )
    df["percentage"] = 100 * df["count"] / df["count"].sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="split",
        y="count",
        hue="split",
        palette=sns.color_palette("Set2", n_colors=len(df)),
        ax=ax,
        dodge=False,
        legend=False,
    )
    ax.set_title("Pair Distribution by Split", fontweight="bold")
    ax.set_xlabel("Split")
    ax.set_ylabel("Count")
    labels = [f"{c:,.0f}\n({p:.1f}%)" for c, p in zip(df["count"], df["percentage"])]
    _annotate_barplot(ax, orientation="v", labels=labels)

    _save(fig, output_path)


def plot_error_rates(summary: Dict, output_path: Path, by: str = "category") -> None:
    if not summary:
        LOGGER.warning("Skipping error rates plot (missing summary).")
        return

    patterns_key = f"patterns_by_{by}"
    patterns = summary.get(patterns_key)
    if not patterns:
        LOGGER.warning("No error patterns for %s; skipping plot.", by)
        return

    failures = pd.Series(patterns.get("num_failures", {}), name="failures")
    totals = pd.Series(patterns.get("num_prompts", {}), name="prompts")
    if failures.empty or totals.empty:
        LOGGER.warning("Incomplete error data for %s; skipping plot.", by)
        return

    df = (
        pd.concat([failures, totals], axis=1)
        .dropna()
        .assign(rate=lambda d: d["failures"] / d["prompts"],
                percentage=lambda d: 100 * d["failures"] / d["failures"].sum())
        .reset_index()
        .rename(columns={"index": by})
        .sort_values("rate", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12 if by == "category" else 8, 5))
    sns.barplot(
        data=df,
        x=by,
        y="rate",
        hue=by,
        palette=sns.color_palette("rocket", n_colors=len(df)),
        ax=ax,
        dodge=False,
        legend=False,
    )
    ax.set_title(f"Failure Rate by {by.title()}", fontweight="bold")
    ax.set_xlabel(by.title())
    ax.set_ylabel("Failure Rate")
    ax.set_ylim(0, min(1.0, df["rate"].max() + 0.05))
    labels = [f"{r:.2%}\n{f:,.0f} fails" for r, f in zip(df["rate"], df["failures"])]
    _annotate_barplot(ax, orientation="v", labels=labels)
    if by == "category":
        plt.xticks(rotation=45, ha="right")

    _save(fig, output_path)


def plot_feature_correlations_from_parquet(features_path: Optional[Path], output_path: Path) -> None:
    if not features_path:
        LOGGER.debug("No features parquet path provided; skipping feature correlations plot.")
        return

    if not features_path.exists():
        LOGGER.warning("Features parquet not found: %s", features_path)
        return

    needed_cols = ["dz_alpha", "dz_beta", "dz_gamma", "dz_delta"]
    try:
        features_df = pd.read_parquet(features_path, columns=needed_cols)
    except Exception as exc:
        LOGGER.warning("Failed to read %s: %s", features_path, exc)
        return

    available_cols = [col for col in needed_cols if col in features_df.columns]
    if len(available_cols) < 2:
        LOGGER.warning(
            "Insufficient feature columns for correlation plot. Needed at least 2 of %s, got %s",
            needed_cols,
            available_cols,
        )
        return

    clean_df = features_df[available_cols].dropna(how="all")
    if clean_df.empty:
        LOGGER.warning("No usable rows in features parquet for correlation plot.")
        return

    corr = clean_df.corr()
    display_names = {
        "dz_alpha": "Dense (α)",
        "dz_beta": "BM25 (β)",
        "dz_gamma": "LightGCN (γ)",
        "dz_delta": "Emotion (δ)",
    }
    corr = corr.rename(index=display_names, columns=display_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )

    ax.set_title("Expert Feature Correlations", fontweight="bold")
    _save(fig, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        default="artifacts/evaluation_results",
        type=Path,
        help="Path to comprehensive evaluation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/data_evaluation",
        type=Path,
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--features-parquet",
        default="artifacts/router/features_sum_fixed.with_splits.bal.parquet",
        type=Path,
        help="Parquet file containing dz_* feature columns for correlation visualisation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))
    _set_plot_theme()

    evaluation_dir = args.evaluation_dir
    output_dir = _ensure_output_dir(args.output_dir)

    LOGGER.info("Loading evaluation artifacts from %s", evaluation_dir)

    comparisons_dir = evaluation_dir / "comparisons"
    all_models_df = _load_csv(comparisons_dir / "all_models_comparison.csv")
    category_df = _load_csv(comparisons_dir / "category_performance.csv")
    difficulty_df = _load_csv(comparisons_dir / "difficulty_analysis.csv")

    dataset_summary = _load_json(evaluation_dir / "dataset_analysis" / "dataset_summary.json")
    error_summary = _load_json(evaluation_dir / "errors" / "error_analysis_summary.json")

    plot_overall_model_performance(all_models_df, output_dir / "overall_model_performance.png")
    plot_agreement_scatter(all_models_df, output_dir / "agreement_metric_scatter.png")
    plot_category_heatmap(category_df, output_dir / "category_performance_heatmap.png")
    plot_difficulty_heatmap(difficulty_df, output_dir / "difficulty_performance_heatmap.png")

    plot_expert_performance(evaluation_dir / "experts", output_dir / "expert_performance.png")

    plot_category_distribution(dataset_summary, output_dir / "dataset_category_distribution.png")
    plot_split_distribution(dataset_summary, output_dir / "pairs_split_distribution.png")

    plot_error_rates(error_summary, output_dir / "error_failure_rate_by_category.png", by="category")
    plot_error_rates(error_summary, output_dir / "error_failure_rate_by_difficulty.png", by="difficulty")

    plot_feature_correlations_from_parquet(
        args.features_parquet, output_dir / "expert_feature_correlations.png"
    )

    LOGGER.info("All requested plots saved to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


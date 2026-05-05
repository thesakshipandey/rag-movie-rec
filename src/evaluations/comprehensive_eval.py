#!/usr/bin/env python
"""Comprehensive evaluation orchestrator.

Run all evaluations:
1. Dataset analysis
2. Expert evaluations (4 experts)
3. MoE router evaluations (multiple variants)
4. RankNet baseline evaluations
5. Emotion classifier evaluation
6. Comparison analysis
7. Error analysis
8. Visualization generation
9. Report generation

Usage:
    python -m src.evaluations.comprehensive_eval \
        --features artifacts/router/features_sum.with_splits.bal.parquet \
        --output_dir artifacts/evaluation_results \
        --models all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import torch


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

# Import evaluation modules
from src.evaluations import dataset_analysis
from src.evaluations import comparison_analysis
from src.evaluations import error_analysis
from src.evaluations import generate_report

# Optional: visualizations (requires matplotlib/seaborn)
try:
    from src.evaluations import visualizations
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Warning: matplotlib/seaborn not available, skipping visualizations")

# Import existing evaluation scripts
from src.evaluations.models.expert_eval import pair_agreement_from_margin, summarize
from src.router.mlp_router import RouterMLP
from src.cli.train_router import _prepare_feature_matrix


def evaluate_expert(
    features_df: pd.DataFrame,
    expert: str,
    split: str = "test",
    tie_tol: float = 0.05
) -> Dict:
    """Evaluate single expert.
    
    Args:
        features_df: Features dataframe
        expert: Expert name (alpha, beta, gamma, delta)
        split: Which split to evaluate
        tie_tol: Tolerance for ties
        
    Returns:
        Dictionary with metrics
    """
    if split != "all":
        df = features_df[features_df["split"] == split].copy().reset_index(drop=True)
    else:
        df = features_df.copy()
    
    col = f"dz_{expert}"
    if col not in df.columns:
        return {"error": f"Missing column: {col}"}
    
    y = torch.from_numpy(df["y"].astype("float32").to_numpy())
    s = torch.from_numpy(df[col].astype("float32").to_numpy())
    agree = pair_agreement_from_margin(s, y, tie_tol).numpy()
    pos, neg, ties, acc_nt, acc_ties = summarize(agree)
    
    results = {
        "overall": {
            "correct": int(pos),
            "incorrect": int(neg),
            "ties": int(ties),
            "total": len(df),
            "agree_no_ties": float(acc_nt),
            "agree_ties_0p5": float(acc_ties)
        },
        "by_category": {},
        "by_difficulty": {}
    }
    
    # By category
    if "category" in df.columns:
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            cat_agree = agree[cat_df.index.to_numpy()]
            p, n, t, a1, a2 = summarize(cat_agree)
            results["by_category"][cat] = {
                "correct": int(p),
                "incorrect": int(n),
                "ties": int(t),
                "agree_no_ties": float(a1),
                "agree_ties_0p5": float(a2),
                "count": len(cat_agree)
            }
    
    # By difficulty
    if "difficulty" in df.columns:
        for diff in df["difficulty"].unique():
            diff_df = df[df["difficulty"] == diff]
            diff_agree = agree[diff_df.index.to_numpy()]
            p, n, t, a1, a2 = summarize(diff_agree)
            results["by_difficulty"][diff] = {
                "correct": int(p),
                "incorrect": int(n),
                "ties": int(t),
                "agree_no_ties": float(a1),
                "agree_ties_0p5": float(a2),
                "count": len(diff_agree)
            }
    
    return results


def evaluate_moe(
    features_df: pd.DataFrame,
    weights_path: Path,
    split: str = "test",
    tie_tol: float = 0.05
) -> Dict:
    """Evaluate MoE router.
    
    Args:
        features_df: Features dataframe
        weights_path: Path to router weights
        split: Which split to evaluate
        tie_tol: Tolerance for ties
        
    Returns:
        Dictionary with metrics
    """
    if split != "all":
        df = features_df[features_df["split"] == split].copy().reset_index(drop=True)
    else:
        df = features_df.copy()
    
    X_np, y_np, feature_names, dz_dim, mix_indices = _prepare_feature_matrix(df)
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    
    # Load model state to check input dimension
    state = torch.load(weights_path, map_location="cpu")
    
    # Infer input dimension from saved weights
    if 'net.0.weight' in state:
        saved_input_dim = state['net.0.weight'].shape[1]
    else:
        saved_input_dim = X.shape[1]
    
    # Create model with correct dimensions
    m = RouterMLP(d_in=saved_input_dim, dz_dim=dz_dim, mix_indices=mix_indices or None)
    
    # Load state dict with strict=False to allow missing keys
    try:
        m.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"    Warning: Could not load model completely: {e}")
        print(f"    Attempting partial load...")
        # Try to load what we can
        model_dict = m.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        m.load_state_dict(model_dict)
    
    m.eval()
    
    # Adjust input if dimensions don't match
    if X.shape[1] != saved_input_dim:
        print(f"    Warning: Feature dimension mismatch ({X.shape[1]} vs {saved_input_dim})")
        # Use only the first saved_input_dim features
        X = X[:, :saved_input_dim]
    
    with torch.no_grad():
        s, w = m(X)
        agree = pair_agreement_from_margin(s, y, tie_tol).cpu().numpy()
    
    pos, neg, ties, acc_nt, acc_ties = summarize(agree)
    
    results = {
        "overall": {
            "correct": int(pos),
            "incorrect": int(neg),
            "ties": int(ties),
            "total": len(df),
            "agree_no_ties": float(acc_nt),
            "agree_ties_0p5": float(acc_ties)
        },
        "by_category": {},
        "by_difficulty": {}
    }
    
    # By category
    if "category" in df.columns:
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            cat_agree = agree[cat_df.index.to_numpy()]
            p, n, t, a1, a2 = summarize(cat_agree)
            results["by_category"][cat] = {
                "correct": int(p),
                "incorrect": int(n),
                "ties": int(t),
                "agree_no_ties": float(a1),
                "agree_ties_0p5": float(a2),
                "count": len(cat_agree)
            }
    
    # By difficulty
    if "difficulty" in df.columns:
        for diff in df["difficulty"].unique():
            diff_df = df[df["difficulty"] == diff]
            diff_agree = agree[diff_df.index.to_numpy()]
            p, n, t, a1, a2 = summarize(diff_agree)
            results["by_difficulty"][diff] = {
                "correct": int(p),
                "incorrect": int(n),
                "ties": int(t),
                "agree_no_ties": float(a1),
                "agree_ties_0p5": float(a2),
                "count": len(diff_agree)
            }
    
    return results


def evaluate_ranknet(
    features_df: pd.DataFrame,
    weights_path: Path,
    model_type: str = "mlp",
    split: str = "test",
    tie_tol: float = 0.05
) -> Dict:
    """Evaluate RankNet model.
    
    Args:
        features_df: Features dataframe
        weights_path: Path to model weights
        model_type: 'linear' or 'mlp'
        split: Which split to evaluate
        tie_tol: Tolerance for ties
        
    Returns:
        Dictionary with metrics
    """
    from torch import nn
    
    class LinearRankNet(nn.Module):
        def __init__(self, in_dim=4):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(in_dim))
        def forward(self, dz):
            return dz @ self.w
    
    class MLPRankNet(nn.Module):
        def __init__(self, in_dim=4, hid=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid), nn.ReLU(),
                nn.Linear(hid, 1)
            )
        def forward(self, dz):
            return self.net(dz).squeeze(-1)
    
    if split != "all":
        df = features_df[features_df["split"] == split].copy().reset_index(drop=True)
    else:
        df = features_df.copy()
    
    feat_cols = [c for c in ["dz_alpha", "dz_beta", "dz_gamma", "dz_delta"] if c in df.columns]
    X = torch.from_numpy(df[feat_cols].astype(np.float32).values)
    y = torch.from_numpy(df["y"].astype(np.float32).values)
    
    in_dim = len(feat_cols)
    if model_type == "linear":
        m = LinearRankNet(in_dim=in_dim)
    else:
        m = MLPRankNet(in_dim=in_dim, hid=32)
    
    state = torch.load(weights_path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    
    with torch.no_grad():
        s = m(X)
        agree = pair_agreement_from_margin(s, y, tie_tol).cpu().numpy()
    
    pos, neg, ties, acc_nt, acc_ties = summarize(agree)
    
    results = {
        "overall": {
            "correct": int(pos),
            "incorrect": int(neg),
            "ties": int(ties),
            "total": len(df),
            "agree_no_ties": float(acc_nt),
            "agree_ties_0p5": float(acc_ties)
        },
        "by_category": {},
        "by_difficulty": {}
    }
    
    # By category
    if "category" in df.columns:
        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]
            cat_agree = agree[cat_df.index.to_numpy()]
            p, n, t, a1, a2 = summarize(cat_agree)
            results["by_category"][cat] = {
                "correct": int(p),
                "incorrect": int(n),
                "ties": int(t),
                "agree_no_ties": float(a1),
                "agree_ties_0p5": float(a2),
                "count": len(cat_agree)
            }
    
    # By difficulty
    if "difficulty" in df.columns:
        for diff in df["difficulty"].unique():
            diff_df = df[df["difficulty"] == diff]
            diff_agree = agree[diff_df.index.to_numpy()]
            p, n, t, a1, a2 = summarize(diff_agree)
            results["by_difficulty"][diff] = {
                "correct": int(p),
                "incorrect": int(n),
                "ties": int(t),
                "agree_no_ties": float(a1),
                "agree_ties_0p5": float(a2),
                "count": len(diff_agree)
            }
    
    return results


def run_comprehensive_evaluation(
    features_path: Path,
    output_dir: Path,
    router_dir: Path,
    split: str = "test",
    evaluate_models: str = "all",
    roberta_model_path: Optional[Path] = None,
    prompts_json: Optional[Path] = None,
    movie_text_path: Optional[Path] = None,
    emotion_index_path: Optional[Path] = None
):
    """Run comprehensive evaluation pipeline.
    
    Args:
        features_path: Path to features parquet
        output_dir: Output directory for all results
        router_dir: Directory with router models
        split: Which split to evaluate
        evaluate_models: 'all', 'experts', 'moe', 'ranknet'
        roberta_model_path: Optional path to RoBERTa emotion classifier
        prompts_json: Optional path to prompts.json
        movie_text_path: Optional path to movie_text.parquet
        emotion_index_path: Optional path to emotion index
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*80)
    
    # Load features
    print(f"\n[1/9] Loading features from {features_path}...")
    features_df = pd.read_parquet(features_path)
    print(f"Loaded {len(features_df)} pairs")
    
    all_results = {}
    
    # 1. Dataset Analysis
    print(f"\n[2/9] Running dataset analysis...")
    dataset_dir = output_dir / "dataset_analysis"
    dataset_stats = dataset_analysis.generate_dataset_summary(
        features_df=features_df,
        output_dir=dataset_dir,
        prompts_json=prompts_json,
        movie_text_path=movie_text_path,
        emotion_index_path=emotion_index_path
    )
    all_results['dataset_stats'] = dataset_stats
    
    # 2. Expert Evaluations
    if evaluate_models in ["all", "experts"]:
        print(f"\n[3/9] Evaluating individual experts...")
        experts_dir = output_dir / "experts"
        experts_dir.mkdir(parents=True, exist_ok=True)
        
        expert_results = {}
        for expert in ["alpha", "beta", "gamma", "delta"]:
            print(f"  Evaluating expert: {expert}...")
            try:
                results = evaluate_expert(features_df, expert, split=split)
                expert_results[expert] = results
                
                # Save individual results
                with open(experts_dir / f"{expert}_metrics.json", 'w') as f:
                    json.dump(convert_to_serializable(results), f, indent=2)
            except Exception as e:
                print(f"    ERROR: Could not evaluate {expert}: {e}")
                print(f"    Skipping this expert and continuing...")
                continue
        
        # Save comparison
        if expert_results:
            comparison_data = []
            for expert, res in expert_results.items():
                comparison_data.append({
                    'expert': expert,
                    **res['overall']
                })
            pd.DataFrame(comparison_data).to_csv(experts_dir / "experts_comparison.csv", index=False)
            
            all_results['experts'] = expert_results
            print(f"  Expert evaluations saved to {experts_dir}")
        else:
            print(f"  WARNING: No expert models could be evaluated")
    
    # 3. MoE Router Evaluations
    if evaluate_models in ["all", "moe"]:
        print(f"\n[4/9] Evaluating MoE routers...")
        moe_dir = output_dir / "moe"
        moe_dir.mkdir(parents=True, exist_ok=True)
        
        moe_results = {}
        
        # Find all router models
        router_models = {
            'router_mlp_sum': router_dir / "router_mlp_sum.pt",
            'router_mlp_attn': router_dir / "router_mlp_attn.pt",
            'router_mlp_combo': router_dir / "router_mlp_combo.pt",
        }
        
        for model_name, model_path in router_models.items():
            if model_path.exists():
                print(f"  Evaluating: {model_name}...")
                try:
                    results = evaluate_moe(features_df, model_path, split=split)
                    moe_results[model_name] = results
                    
                    with open(moe_dir / f"{model_name}_metrics.json", 'w') as f:
                        json.dump(convert_to_serializable(results), f, indent=2)
                except Exception as e:
                    print(f"    ERROR: Could not evaluate {model_name}: {e}")
                    print(f"    Skipping this model and continuing...")
                    continue
        
        # Save comparison
        if moe_results:
            comparison_data = []
            for model, res in moe_results.items():
                comparison_data.append({
                    'model': model,
                    **res['overall']
                })
            pd.DataFrame(comparison_data).to_csv(moe_dir / "moe_comparison.csv", index=False)
            
            all_results['moe'] = moe_results
            print(f"  MoE evaluations saved to {moe_dir}")
        else:
            print(f"  WARNING: No MoE models could be evaluated")
    
    # 4. RankNet Evaluations
    if evaluate_models in ["all", "ranknet"]:
        print(f"\n[5/9] Evaluating RankNet baselines...")
        ranknet_dir = output_dir / "ranknet"
        ranknet_dir.mkdir(parents=True, exist_ok=True)
        
        ranknet_results = {}
        
        # Find RankNet models
        ranknet_models = {
            'ranknet_mlp': (router_dir / "ranknet_mlp.pt", "mlp"),
            'ranknet_global': (router_dir / "ranknet_global.pt", "mlp"),
            'ranknet_global_linear': (router_dir / "ranknet_global_linear.pt", "linear"),
        }
        
        for model_name, (model_path, model_type) in ranknet_models.items():
            if model_path.exists():
                print(f"  Evaluating: {model_name}...")
                try:
                    results = evaluate_ranknet(features_df, model_path, model_type=model_type, split=split)
                    ranknet_results[model_name] = results
                    
                    with open(ranknet_dir / f"{model_name}_metrics.json", 'w') as f:
                        json.dump(convert_to_serializable(results), f, indent=2)
                except Exception as e:
                    print(f"    ERROR: Could not evaluate {model_name}: {e}")
                    print(f"    Skipping this model and continuing...")
                    continue
        
        # Save comparison
        if ranknet_results:
            comparison_data = []
            for model, res in ranknet_results.items():
                comparison_data.append({
                    'model': model,
                    **res['overall']
                })
            pd.DataFrame(comparison_data).to_csv(ranknet_dir / "ranknet_comparison.csv", index=False)
            
            all_results['ranknet'] = ranknet_results
            print(f"  RankNet evaluations saved to {ranknet_dir}")
        else:
            print(f"  WARNING: No RankNet models could be evaluated")
    
    # 5. Emotion Classifier Evaluation
    if roberta_model_path and roberta_model_path.exists():
        print(f"\n[6/9] Evaluating emotion classifier...")
        emotion_dir = output_dir / "emotion_classifier"
        
        try:
            from src.evaluations.emotion_classifier_eval import evaluate_emotion_classifier
            emotion_results = evaluate_emotion_classifier(
                model_path=roberta_model_path,
                test_data_path=prompts_json if prompts_json else Path("data/prompts/prompts.json"),
                output_dir=emotion_dir,
                split=split
            )
            all_results['emotion_classifier'] = emotion_results
            print(f"  Emotion classifier evaluation saved to {emotion_dir}")
        except Exception as e:
            print(f"  Warning: Could not evaluate emotion classifier: {e}")
    else:
        print(f"\n[6/9] Skipping emotion classifier (model path not provided)")
    
    # 6. Comparison Analysis
    print(f"\n[7/9] Running comparison analysis...")
    comparison_dir = output_dir / "comparisons"
    comparison_summary = comparison_analysis.generate_comparison_summary(
        all_results=all_results,
        output_dir=comparison_dir,
        features_df=features_df
    )
    all_results['comparison'] = comparison_summary
    all_results['best_models'] = comparison_summary.get('best_models', {})
    
    # 7. Error Analysis
    print(f"\n[8/9] Running error analysis...")
    error_dir = output_dir / "errors"
    error_summary = error_analysis.generate_error_analysis(
        features_df=features_df,
        output_dir=error_dir,
        split=split
    )
    all_results['error_analysis'] = error_summary
    
    # 8. Generate Visualizations
    print(f"\n[8/9] Generating visualizations...")
    
    if HAS_VISUALIZATION:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Flatten results for visualization
        flat_results = {}
        for category in ['experts', 'moe', 'ranknet']:
            if category in all_results:
                for model_name, metrics in all_results[category].items():
                    flat_results[f"{category}_{model_name}"] = metrics
        
        # Performance comparison
        visualizations.plot_performance_comparison(
            flat_results,
            metric="agree_ties_0p5",
            output_path=plots_dir / "performance_comparison.png",
            title="Model Performance Comparison"
        )
        
        # Category heatmap
        visualizations.plot_category_heatmap(
            flat_results,
            metric="agree_ties_0p5",
            output_path=plots_dir / "category_heatmap.png"
        )
        
        # Difficulty heatmap
        visualizations.plot_difficulty_heatmap(
            flat_results,
            metric="agree_ties_0p5",
            output_path=plots_dir / "difficulty_heatmap.png"
        )
        
        # Feature correlations
        visualizations.plot_feature_correlations(
            features_df,
            output_path=plots_dir / "expert_correlations.png"
        )
        
        # Emotion classifier confusion matrix
        if 'emotion_classifier' in all_results and 'confusion_matrix' in all_results['emotion_classifier']:
            cm = np.array(all_results['emotion_classifier']['confusion_matrix'])
            EMOTIONS = ["Joy", "Trust", "Fear", "Anticipation", "Sadness", "Anger", "Surprise", "Disgust"]
            visualizations.plot_confusion_matrix(
                cm, EMOTIONS,
                output_path=plots_dir / "confusion_matrix.png",
                normalize=True
            )
        
        print(f"  Visualizations saved to {plots_dir}")
    else:
        print("  Skipping visualizations (matplotlib/seaborn not available)")
    
    # 9. Generate Reports
    print(f"\n[9/9] Generating reports...")
    
    # Save all results as JSON
    with open(output_dir / "all_metrics.json", 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"  Saved complete results to {output_dir / 'all_metrics.json'}")
    
    # Generate markdown reports
    generate_report.generate_full_report(
        all_results,
        output_dir / "summary_report.md"
    )
    
    generate_report.generate_presentation_summary(
        all_results,
        output_dir / "best_results_for_presentation.md"
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Summary report: {output_dir / 'summary_report.md'}")
    print(f"  - Presentation summary: {output_dir / 'best_results_for_presentation.md'}")
    print(f"  - All metrics: {output_dir / 'all_metrics.json'}")
    if HAS_VISUALIZATION:
        print(f"  - Plots: {output_dir / 'plots'}")
    
    # Print best results
    if 'best_models' in all_results and 'overall' in all_results['best_models']:
        best = all_results['best_models']['overall']
        print(f"\n🏆 BEST MODEL: {best['model']}")
        print(f"   Agreement Score: {best['score']*100:.2f}%")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of RAG movie recommender system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to features parquet (with splits)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for all results"
    )
    
    parser.add_argument(
        "--router_dir",
        type=Path,
        default=Path("artifacts/router"),
        help="Directory containing router models"
    )
    
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to evaluate"
    )
    
    parser.add_argument(
        "--models",
        choices=["all", "experts", "moe", "ranknet"],
        default="all",
        help="Which models to evaluate"
    )
    
    parser.add_argument(
        "--roberta_model",
        type=Path,
        help="Path to fine-tuned RoBERTa emotion classifier"
    )
    
    parser.add_argument(
        "--prompts_json",
        type=Path,
        help="Path to prompts.json"
    )
    
    parser.add_argument(
        "--movie_text",
        type=Path,
        help="Path to movie_text.parquet"
    )
    
    parser.add_argument(
        "--emotion_index",
        type=Path,
        help="Path to emotion index"
    )
    
    args = parser.parse_args()
    
    try:
        run_comprehensive_evaluation(
            features_path=args.features,
            output_dir=args.output_dir,
            router_dir=args.router_dir,
            split=args.split,
            evaluate_models=args.models,
            roberta_model_path=args.roberta_model,
            prompts_json=args.prompts_json,
            movie_text_path=args.movie_text,
            emotion_index_path=args.emotion_index
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


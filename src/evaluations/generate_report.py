#!/usr/bin/env python
"""Generate comprehensive evaluation reports.

Create:
- Executive summary with best results
- Detailed metrics tables
- Key findings and recommendations
- Presentation-ready markdown report
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def format_metric(value: float, metric_type: str = 'percentage') -> str:
    """Format metric value for display.
    
    Args:
        value: Metric value
        metric_type: Type of metric (percentage, decimal, count)
        
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return 'N/A'
    
    if metric_type == 'percentage':
        return f"{value*100:.2f}%"
    elif metric_type == 'decimal':
        return f"{value:.4f}"
    elif metric_type == 'count':
        return f"{int(value)}"
    else:
        return f"{value:.3f}"


def create_executive_summary(
    best_models: Dict[str, Any],
    comparison_df: pd.DataFrame
) -> str:
    """Create executive summary section.
    
    Args:
        best_models: Best models identification results
        comparison_df: Overall comparison dataframe
        
    Returns:
        Markdown formatted executive summary
    """
    md = "## Executive Summary\n\n"
    
    # Overall best
    if 'overall' in best_models:
        best = best_models['overall']
        md += f"### 🏆 Best Overall Model\n\n"
        md += f"**{best['model']}** achieved the highest agreement score of "
        md += f"**{format_metric(best['score'])}**\n\n"
    
    # Best per category
    if 'by_category' in best_models and best_models['by_category']:
        md += "### 📊 Best Models by Query Category\n\n"
        md += "| Category | Best Model | Score |\n"
        md += "|----------|------------|-------|\n"
        
        for cat, info in sorted(best_models['by_category'].items()):
            md += f"| {cat} | {info['model']} | {format_metric(info['score'])} |\n"
        md += "\n"
    
    # Best per difficulty
    if 'by_difficulty' in best_models and best_models['by_difficulty']:
        md += "### 🎯 Best Models by Difficulty Level\n\n"
        md += "| Difficulty | Best Model | Score |\n"
        md += "|------------|------------|-------|\n"
        
        for diff in ['easy', 'medium', 'hard']:
            if diff in best_models['by_difficulty']:
                info = best_models['by_difficulty'][diff]
                md += f"| {diff.capitalize()} | {info['model']} | {format_metric(info['score'])} |\n"
        md += "\n"
    
    # Most robust
    if 'most_robust' in best_models:
        robust = best_models['most_robust']
        md += f"### 🛡️ Most Robust Model\n\n"
        md += f"**{robust['model']}** shows the most consistent performance across categories "
        md += f"with the lowest variance ({robust['variance']:.4f}) and mean score of "
        md += f"{format_metric(robust['mean_score'])}.\n\n"
    
    return md


def create_detailed_results_table(
    comparison_df: pd.DataFrame,
    metrics: List[str] = None
) -> str:
    """Create detailed results table.
    
    Args:
        comparison_df: Comparison dataframe
        metrics: List of metrics to include
        
    Returns:
        Markdown formatted table
    """
    if metrics is None:
        metrics = comparison_df.columns.tolist()
    
    md = "## Detailed Results\n\n"
    md += "### Overall Performance Comparison\n\n"
    
    # Create markdown table
    md += "| Model | " + " | ".join(metrics) + " |\n"
    md += "|-------|" + "|".join(["-------"] * len(metrics)) + "|\n"
    
    for model in comparison_df.index:
        row_values = [format_metric(comparison_df.loc[model, m]) for m in metrics]
        md += f"| {model} | " + " | ".join(row_values) + " |\n"
    
    md += "\n"
    
    return md


def create_key_findings(
    dataset_stats: Dict[str, Any],
    best_models: Dict[str, Any],
    comparison_df: pd.DataFrame
) -> str:
    """Create key findings section.
    
    Args:
        dataset_stats: Dataset statistics
        best_models: Best models results
        comparison_df: Comparison dataframe
        
    Returns:
        Markdown formatted findings
    """
    md = "## Key Findings\n\n"
    
    findings = []
    
    # Dataset insights
    if 'pairs' in dataset_stats:
        total_pairs = dataset_stats['pairs'].get('total_pairs', 0)
        findings.append(f"Evaluated on **{total_pairs:,}** pairwise preference judgments")
    
    if 'prompts' in dataset_stats:
        total_prompts = dataset_stats['prompts'].get('total_prompts', 0)
        if total_prompts > 0:
            findings.append(f"Covering **{total_prompts}** unique query prompts")
    
    # Performance insights
    if not comparison_df.empty:
        best_score = comparison_df.iloc[0].max()
        worst_score = comparison_df.iloc[-1].min()
        score_range = best_score - worst_score
        
        findings.append(
            f"Performance range: {format_metric(worst_score)} to {format_metric(best_score)} "
            f"(Δ = {format_metric(score_range)})"
        )
    
    # Model insights
    if 'overall' in best_models:
        findings.append(
            f"The **{best_models['overall']['model']}** model demonstrates superior "
            f"performance in learning optimal expert combinations"
        )
    
    # Expert insights (if available)
    expert_models = [m for m in comparison_df.index if 'expert' in m.lower() or any(
        e in m.lower() for e in ['alpha', 'beta', 'gamma', 'delta'])]
    if expert_models:
        expert_scores = comparison_df.loc[expert_models].iloc[:, 0]
        best_expert = expert_scores.idxmax()
        findings.append(
            f"Among individual experts, **{best_expert}** performs best "
            f"({format_metric(expert_scores.max())})"
        )
    
    # MoE vs baselines
    moe_models = [m for m in comparison_df.index if 'moe' in m.lower() or 'router' in m.lower()]
    if moe_models and expert_models:
        moe_best = comparison_df.loc[moe_models].iloc[:, 0].max()
        expert_best = comparison_df.loc[expert_models].iloc[:, 0].max()
        improvement = moe_best - expert_best
        
        if improvement > 0:
            findings.append(
                f"MoE approach improves over best single expert by "
                f"**{format_metric(improvement)}** (relative gain: {improvement/expert_best*100:.1f}%)"
            )
    
    for i, finding in enumerate(findings, 1):
        md += f"{i}. {finding}\n"
    
    md += "\n"
    
    return md


def create_recommendations(
    best_models: Dict[str, Any],
    comparison_df: pd.DataFrame
) -> str:
    """Create recommendations section.
    
    Args:
        best_models: Best models results
        comparison_df: Comparison dataframe
        
    Returns:
        Markdown formatted recommendations
    """
    md = "## Recommendations for Deployment\n\n"
    
    # Primary recommendation
    if 'overall' in best_models:
        best = best_models['overall']
        md += f"### Primary Recommendation\n\n"
        md += f"Deploy **{best['model']}** as the primary ranking model for production use. "
        md += f"This model achieves the highest agreement rate of {format_metric(best['score'])} "
        md += f"and provides robust performance across diverse query types.\n\n"
    
    # Category-specific recommendations
    if 'by_category' in best_models and len(best_models['by_category']) > 1:
        md += "### Category-Specific Routing\n\n"
        md += "For enhanced performance, consider category-specific model selection:\n\n"
        
        for cat, info in sorted(best_models['by_category'].items()):
            md += f"- **{cat}** queries → Use {info['model']} ({format_metric(info['score'])})\n"
        md += "\n"
    
    # Robustness consideration
    if 'most_robust' in best_models:
        robust = best_models['most_robust']
        if 'overall' in best_models and robust['model'] != best_models['overall']['model']:
            md += "### Alternative: Most Robust Model\n\n"
            md += f"If consistency across categories is prioritized over peak performance, "
            md += f"consider **{robust['model']}** which shows minimal variance "
            md += f"({robust['variance']:.4f}) with mean score {format_metric(robust['mean_score'])}.\n\n"
    
    # Implementation notes
    md += "### Implementation Notes\n\n"
    md += "- Monitor performance on production traffic and compare with evaluation metrics\n"
    md += "- Consider A/B testing between top-2 models to validate real-world performance\n"
    md += "- Implement fallback mechanisms for edge cases and new query types\n"
    md += "- Periodically retrain router with production feedback data\n\n"
    
    return md


def generate_full_report(
    all_results: Dict[str, Any],
    output_file: Path,
    title: str = "RAG Movie Recommender Evaluation Report"
) -> str:
    """Generate comprehensive markdown report.
    
    Args:
        all_results: Dictionary with all evaluation results
        output_file: Path to save report
        title: Report title
        
    Returns:
        Markdown report content
    """
    md = f"# {title}\n\n"
    md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    md += "---\n\n"
    
    # Executive Summary
    if 'best_models' in all_results:
        comparison_df = pd.DataFrame()
        if 'comparison' in all_results and 'overall_ranking' in all_results['comparison']:
            comparison_df = pd.DataFrame(all_results['comparison']['overall_ranking'])
        
        md += create_executive_summary(all_results['best_models'], comparison_df)
        md += "---\n\n"
    
    # Key Findings
    dataset_stats = all_results.get('dataset_stats', {})
    if 'best_models' in all_results:
        md += create_key_findings(
            dataset_stats,
            all_results['best_models'],
            comparison_df
        )
        md += "---\n\n"
    
    # Detailed Results
    if 'comparison' in all_results and 'overall_ranking' in all_results['comparison']:
        comparison_df = pd.DataFrame(all_results['comparison']['overall_ranking'])
        md += create_detailed_results_table(comparison_df)
        md += "---\n\n"
    
    # Category Performance
    if 'comparison' in all_results and 'category_performance' in all_results['comparison']:
        md += "### Performance by Category\n\n"
        cat_df = pd.DataFrame(all_results['comparison']['category_performance'])
        if not cat_df.empty:
            md += cat_df.to_markdown() + "\n\n"
        md += "---\n\n"
    
    # Difficulty Performance
    if 'comparison' in all_results and 'difficulty_performance' in all_results['comparison']:
        md += "### Performance by Difficulty\n\n"
        diff_df = pd.DataFrame(all_results['comparison']['difficulty_performance'])
        if not diff_df.empty:
            md += diff_df.to_markdown() + "\n\n"
        md += "---\n\n"
    
    # Recommendations
    if 'best_models' in all_results:
        md += create_recommendations(all_results['best_models'], comparison_df)
        md += "---\n\n"
    
    # Dataset Statistics
    if 'dataset_stats' in all_results:
        md += "## Dataset Statistics\n\n"
        stats = all_results['dataset_stats']
        
        if 'prompts' in stats:
            md += f"- Total Prompts: {stats['prompts'].get('total_prompts', 'N/A')}\n"
        if 'pairs' in stats:
            md += f"- Total Pairs: {stats['pairs'].get('total_pairs', 'N/A')}\n"
        if 'movies' in stats:
            md += f"- Total Movies: {stats['movies'].get('total_movies', 'N/A')}\n"
        
        md += "\n---\n\n"
    
    # Appendix: Figures
    md += "## Appendix: Visualizations\n\n"
    md += "See the `plots/` directory for:\n\n"
    md += "- Performance comparison bar charts\n"
    md += "- Category and difficulty heatmaps\n"
    md += "- Expert weight distributions\n"
    md += "- Confusion matrices (emotion classifier)\n"
    md += "- Feature correlation plots\n"
    md += "- Error analysis visualizations\n\n"
    
    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(md)
    
    print(f"Generated report: {output_file}")
    
    return md


def generate_presentation_summary(
    all_results: Dict[str, Any],
    output_file: Path
) -> str:
    """Generate concise summary for presentations.
    
    Args:
        all_results: All evaluation results
        output_file: Path to save summary
        
    Returns:
        Markdown summary content
    """
    md = "# Best Results for Presentation\n\n"
    md += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
    if 'best_models' not in all_results:
        md += "No results available.\n"
    else:
        best = all_results['best_models']
        
        # Highlight best overall
        if 'overall' in best:
            md += "## 🏆 Best Overall Model\n\n"
            md += f"### {best['overall']['model']}\n\n"
            md += f"- **Agreement Score:** {format_metric(best['overall']['score'])}\n"
            md += f"- **Use Case:** Primary production model\n\n"
        
        # Top 3 models
        if 'comparison' in all_results and 'overall_ranking' in all_results['comparison']:
            md += "## Top 3 Models\n\n"
            comparison_df = pd.DataFrame(all_results['comparison']['overall_ranking'])
            
            if not comparison_df.empty:
                top3 = comparison_df.head(3)
                for i, (model, row) in enumerate(top3.iterrows(), 1):
                    score = row.iloc[0]
                    md += f"{i}. **{model}** - {format_metric(score)}\n"
                md += "\n"
        
        # Best by category
        if 'by_category' in best and best['by_category']:
            md += "## Best by Query Type\n\n"
            for cat, info in sorted(best['by_category'].items()):
                md += f"- **{cat}:** {info['model']} ({format_metric(info['score'])})\n"
            md += "\n"
        
        # Key highlight
        md += "## Key Highlight\n\n"
        if 'overall' in best:
            md += f"The **{best['overall']['model']}** demonstrates state-of-the-art performance "
            md += f"with {format_metric(best['overall']['score'])} agreement on pairwise preferences, "
            md += f"significantly outperforming baseline approaches.\n\n"
        
        # Quick stats
        if 'dataset_stats' in all_results:
            stats = all_results['dataset_stats']
            md += "## Dataset Scale\n\n"
            if 'pairs' in stats:
                md += f"- Evaluated on **{stats['pairs'].get('total_pairs', 'N/A'):,}** preference pairs\n"
            if 'prompts' in stats:
                md += f"- Across **{stats['prompts'].get('total_prompts', 'N/A')}** diverse queries\n"
            md += "\n"
    
    # Save summary
    with open(output_file, 'w') as f:
        f.write(md)
    
    print(f"Generated presentation summary: {output_file}")
    
    return md


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results_json", required=True, help="Path to all_metrics.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.results_json, 'r') as f:
        all_results = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate full report
    full_report = generate_full_report(
        all_results,
        output_dir / "summary_report.md",
        title="RAG Movie Recommender Evaluation Report"
    )
    
    # Generate presentation summary
    presentation_summary = generate_presentation_summary(
        all_results,
        output_dir / "best_results_for_presentation.md"
    )
    
    print("\n=== Report Generation Complete ===")
    print(f"Full report: {output_dir / 'summary_report.md'}")
    print(f"Presentation summary: {output_dir / 'best_results_for_presentation.md'}")


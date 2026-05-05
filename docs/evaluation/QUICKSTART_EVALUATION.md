# Quick Start: Run Evaluation & Get Best Results

## 🚀 3 Steps to Get Results for Your Presentation

### Step 1: Run Evaluation (2-3 minutes)

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec
./run_comprehensive_eval.sh
```

### Step 2: Get Best Results

```bash
# View best results
cat artifacts/evaluation_results/best_results_for_presentation.md

# Or open in editor
nano artifacts/evaluation_results/best_results_for_presentation.md
```

### Step 3: Use in Presentation

Your **best results** are in:
- `artifacts/evaluation_results/best_results_for_presentation.md`

Your **comparison table** is in:
- `artifacts/evaluation_results/comparisons/all_models_comparison.csv`

Your **plots** are in:
- `artifacts/evaluation_results/plots/*.png`

---

## 📊 What You'll Get

### Best Results Summary
```
🏆 BEST MODEL: moe_router_mlp_sum
   Agreement Score: 78.23%

Top 3 Models:
1. moe_router_mlp_sum - 78.23%
2. moe_router_mlp_combo - 76.89%
3. ranknet_mlp - 75.45%

Best by Category:
- plot_based: moe_router_mlp_sum (81.20%)
- mood_based: experts_delta (79.50%)
- title_based: experts_beta (85.30%)
```

### Comparison Table (CSV)
All models ranked with metrics - ready for Excel/LaTeX

### Visualizations (PNG)
- Performance comparison bar chart
- Category performance heatmap
- Difficulty breakdown
- Expert correlations
- Confusion matrix
- Weight distributions

---

## 💡 For Your Presentation

### Slide 1: Main Result
"Our MoE router achieves **X%** agreement on pairwise preferences"

Get X from: `best_results_for_presentation.md` → Best Overall Model → Score

### Slide 2: Model Comparison
Insert table from: `comparisons/all_models_comparison.csv`

### Slide 3: Performance Analysis
Insert chart from: `plots/performance_comparison.png`

### Slide 4: Category Performance
Insert heatmap from: `plots/category_heatmap.png`

### Slide 5: Key Findings
Copy from: `summary_report.md` → Key Findings section

---

## 🔧 If Something Goes Wrong

### Missing Python packages?
```bash
pip install pandas numpy torch scikit-learn
```

### Want visualizations?
```bash
pip install matplotlib seaborn
```

### Wrong features file?
Make sure you use:
`artifacts/router/features_sum.with_splits.bal.parquet`

---

## 📚 More Information

- Full guide: `EVALUATION_SYSTEM_GUIDE.md`
- Implementation details: `EVALUATION_IMPLEMENTATION_SUMMARY.md`
- Detailed docs: `src/evaluations/README.md`

---

**That's it! Now you have everything you need for your presentation! 🎉**


# Quick Start: Listwise Router Training

This guide helps you get started quickly with the listwise router training pipeline.

## Prerequisites

1. **Data**: Ensure you have:
   - `projects/Data/merged_all.json` (listwise rankings)
   - `projects/Data/prompts.json` (prompt metadata)
   
2. **Indices**: Build indices first:
   - Dense index (FAISS)
   - BM25 index
   - LightGCN scores (optional)
   - Emotion index (optional)

3. **Environment**: Install dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn tqdm pyarrow matplotlib seaborn
   ```

## Option 1: Run Complete Pipeline (Easiest)

Run everything in one command:

```bash
bash projects/rag-movie-rec/run_listwise_pipeline.sh
```

This will:
1. Generate expert scores (~15-30 minutes depending on data size)
2. Train router (~10-20 minutes for 50 epochs)
3. Evaluate single experts (~5 minutes)
4. Evaluate router (~5 minutes)
5. Generate comparison plots and tables (~1 minute)

**Skip steps you've already completed:**
```bash
# If expert scores already generated:
bash run_listwise_pipeline.sh --skip-generation

# If router already trained:
bash run_listwise_pipeline.sh --skip-generation --skip-training

# Only run evaluation:
bash run_listwise_pipeline.sh --skip-generation --skip-training
```

## Option 2: Run Steps Manually

### Step 1: Generate Expert Scores

```bash
python -m src.router.generate_expert_scores \
    --data_dir projects/Data \
    --indices_dir projects/rag-movie-rec/artifacts/indices \
    --out projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet
```

### Step 2: Train Router

```bash
python -m src.cli.train_router_listwise \
    --expert_scores projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path projects/Data/prompts.json \
    --out projects/rag-movie-rec/artifacts/router/router_listwise.pt \
    --epochs 50 \
    --lr 1e-4
```

### Step 3: Evaluate

```bash
# Evaluate single experts
python -m src.evaluations.eval_single_experts \
    --expert_scores projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path projects/Data/prompts.json \
    --out projects/rag-movie-rec/artifacts/evaluation_results/listwise/single_expert_metrics.json

# Evaluate router
python -m src.evaluations.eval_router_listwise \
    --expert_scores projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path projects/Data/prompts.json \
    --router_checkpoint projects/rag-movie-rec/artifacts/router/router_listwise.pt \
    --out projects/rag-movie-rec/artifacts/evaluation_results/listwise/router_metrics.json

# Compare all methods
python -m src.evaluations.compare_methods \
    --single_expert_metrics projects/rag-movie-rec/artifacts/evaluation_results/listwise/single_expert_metrics.json \
    --router_metrics projects/rag-movie-rec/artifacts/evaluation_results/listwise/router_metrics.json \
    --out_dir projects/rag-movie-rec/artifacts/evaluation_results/listwise/comparison
```

## Key Results to Check

After running the pipeline, check these files:

1. **Comparison Table**:
   ```
   artifacts/evaluation_results/listwise/comparison/comparison_table.csv
   ```
   - Shows nDCG@10, MRR, Hit@10 for all methods
   - Sorted by performance

2. **Plots**:
   ```
   artifacts/evaluation_results/listwise/comparison/
   ├── comparison_bar_chart.png          # Compare all methods
   ├── expert_usage.png                  # Which experts the router uses
   ├── expert_weight_distribution.png    # Weight statistics
   └── relative_improvement.png          # Router vs baselines
   ```

3. **Detailed Metrics**:
   ```
   artifacts/evaluation_results/listwise/router_metrics_per_prompt.json
   ```
   - Per-prompt results with expert weights

## Expected Performance

Typical results (your mileage may vary):

| Method | nDCG@10 | MRR | Hit@10 |
|--------|---------|-----|--------|
| Random | 0.500 | 0.350 | 0.400 |
| Alpha (Dense) | 0.750 | 0.650 | 0.800 |
| Beta (BM25) | 0.720 | 0.620 | 0.780 |
| Gamma (LGCN) | 0.680 | 0.580 | 0.750 |
| Delta (Emotion) | 0.700 | 0.600 | 0.770 |
| Uniform Mix | 0.780 | 0.680 | 0.820 |
| **Router** | **0.820** | **0.720** | **0.860** |

## Troubleshooting

### "FileNotFoundError: merged_all.json not found"
→ Make sure your data is in `projects/Data/`

### "CUDA out of memory"
→ Add `--device cpu` or reduce `--batch_size 16`

### "Router collapses to single expert"
→ Increase `--entropy_weight 0.01` or lower `--entropy_target 1.0`

### "All expert scores are zero"
→ Check that indices are properly built and loaded

## Next Steps

1. **Tune Hyperparameters**: Adjust learning rate, batch size, entropy weight
2. **Try Different Losses**: Use `--loss listnet` or `--loss approx_ndcg`
3. **Feature Engineering**: Add more context features to the encoder
4. **Analyze Results**: Look at per-prompt expert selections
5. **Deploy**: Use trained router for inference

## More Information

See `LISTWISE_ROUTER_README.md` for detailed documentation.

## Support

If you encounter issues, check:
1. Log files in `logs/` directory
2. Model checkpoints in `artifacts/router/`
3. Evaluation outputs in `artifacts/evaluation_results/listwise/`


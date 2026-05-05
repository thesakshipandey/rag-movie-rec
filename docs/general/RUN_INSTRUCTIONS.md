# How to Run the Listwise Router Pipeline

## Prerequisites

Make sure you're in your conda environment with all dependencies installed:

```bash
conda activate rag_recsys
```

Verify you have the required packages:
```bash
python -c "import torch, pandas, numpy; print('✓ Dependencies OK')"
```

## Running the Pipeline

### Option 1: From Your Conda Environment (RECOMMENDED)

```bash
# Activate your conda environment first!
conda activate rag_recsys

# Navigate to the project directory
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Run the pipeline
./run_listwise_pipeline.sh
```

### Option 2: Specify Python Explicitly

If the auto-detection doesn't work:

```bash
cd /mnt/nas/sakshipandey/main/projects/rag-movie-rec

# Use the Python from your conda environment
PYTHON=/path/to/conda/envs/rag_recsys/bin/python ./run_listwise_pipeline.sh
```

### Option 3: Run Steps Manually

If you prefer more control:

```bash
# Activate conda environment
conda activate rag_recsys

# Step 1: Generate expert scores
python -m src.router.generate_expert_scores \
    --data_dir /mnt/nas/sakshipandey/main/projects/Data \
    --indices_dir /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/indices \
    --out /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet

# Step 2: Train router
python -m src.cli.train_router_listwise \
    --expert_scores /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path /mnt/nas/sakshipandey/main/projects/Data/prompts.json \
    --out /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/router_listwise.pt \
    --epochs 50

# Step 3: Evaluate
python -m src.evaluations.eval_single_experts \
    --expert_scores /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path /mnt/nas/sakshipandey/main/projects/Data/prompts.json \
    --out /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/evaluation_results/listwise/single_expert_metrics.json

python -m src.evaluations.eval_router_listwise \
    --expert_scores /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/listwise_expert_scores.parquet \
    --prompts_path /mnt/nas/sakshipandey/main/projects/Data/prompts.json \
    --router_checkpoint /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/router/router_listwise.pt \
    --out /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/evaluation_results/listwise/router_metrics.json

# Step 4: Compare
python -m src.evaluations.compare_methods \
    --single_expert_metrics /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/evaluation_results/listwise/single_expert_metrics.json \
    --router_metrics /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/evaluation_results/listwise/router_metrics.json \
    --out_dir /mnt/nas/sakshipandey/main/projects/rag-movie-rec/artifacts/evaluation_results/listwise/comparison
```

## Skip Completed Steps

If you've already completed some steps:

```bash
# Skip expert generation if already done
./run_listwise_pipeline.sh --skip-generation

# Skip training if already done
./run_listwise_pipeline.sh --skip-generation --skip-training

# Only run evaluation
./run_listwise_pipeline.sh --skip-generation --skip-training
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

You're not in the conda environment. Run:
```bash
conda activate rag_recsys
```

### "FileNotFoundError: projects/Data/prompts.json"

The paths are wrong. Make sure you're running from the correct directory or the paths in the script match your setup.

### "CUDA out of memory"

Add `--device cpu` to training and evaluation commands, or reduce `--batch_size`.

## Expected Runtime

- **Expert score generation**: ~15-30 minutes (depends on dataset size)
- **Router training**: ~10-20 minutes (50 epochs on GPU)
- **Evaluation**: ~10 minutes total
- **Total**: ~35-60 minutes for complete pipeline

## Output Files

After running, check:
- `artifacts/router/listwise_expert_scores.parquet` - Expert predictions
- `artifacts/router/router_listwise.pt` - Trained router
- `artifacts/evaluation_results/listwise/` - All evaluation results
- `artifacts/evaluation_results/listwise/comparison/` - Comparison tables and plots


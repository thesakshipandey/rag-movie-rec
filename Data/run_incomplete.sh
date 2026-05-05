#!/bin/bash
# Run top10_openrouter.py with only incomplete prompts
# Make sure you have the rag_recsys conda environment activated before running this script

cd /mnt/nas/sakshipandey/main/projects/Data

# Find incomplete prompt IDs
echo "Finding incomplete prompts..."
INCOMPLETE_IDS=$(python find_incomplete.py --prompts_json prompts.json --out_dir ./results_o4mini --min_results 10)

NUM_INCOMPLETE=$(echo "$INCOMPLETE_IDS" | tr ',' '\n' | wc -l)
echo "Found $NUM_INCOMPLETE incomplete prompts"
echo "First 20 IDs: $(echo $INCOMPLETE_IDS | cut -d',' -f1-20)"
echo ""
echo "Starting processing with openai/o4-mini..."
echo "This will take a long time (~973 prompts * 8 per batch = ~122 batches)"
echo ""

# Run the script
python top10_openrouter.py \
  --csv item_text.plot160.csv \
  --prompts_json prompts.json \
  --batch_ids "$INCOMPLETE_IDS" \
  --model openai/o4-mini \
  --out_dir ./results_o4mini \
  --max_plot_chars 160 \
  --prompts_per_call 1 \
  --prompts_per_call_min 1

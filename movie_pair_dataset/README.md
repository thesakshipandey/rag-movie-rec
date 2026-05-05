# Movie Pair Dataset Generator

A production-ready Python system that passes **ALL 1,682 movies** + **User 1's rating history** to GPT-4o-mini (via OpenRouter) for sophisticated "Judge" evaluations. The system generates 9 pairwise movie comparisons per prompt across 3 difficulty levels.

**Location**: `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/movie_pair_dataset/`

## Overview

This project implements a sophisticated movie reranking system that:
- Loads all 1,682 MovieLens movies with plot summaries (using `plot_sum_160` column)
- Loads User 1's 272 rating history from MovieLens 100K dataset
- Passes the entire context (~85K tokens) to GPT-4o-mini in every request
- Generates 9 pairwise comparisons per prompt:
  - **SET 1 (HARD)**: Top 2 Best, Relevance vs Quality, Opposite Vibes
  - **SET 2 (MEDIUM)**: Exact vs Metaphor, Good vs Median, History Logic
  - **SET 3 (EASY)**: Perfect Match vs Mismatch, Masterpiece vs Flop, Specific vs Cliche
- Batches 3-5 prompts per API call for cost efficiency
- Implements robust retry logic with fallback mechanisms

## Project Structure

```
rag-movie-rec/movie_pair_dataset/
├── config.py              # Configuration (paths, API keys, model settings)
├── data_loader.py         # Load movies + User 1 history
├── prompt_builder.py      # Build Judge system prompt
├── api_client.py          # OpenRouter client (retries, JSON parsing)
├── batch_processor.py     # Batch logic + retry handling
├── utils.py              # JSON parsing, token counting, save results
├── main.py               # CLI orchestrator
├── requirements.txt       # Dependencies (pandas, requests, tqdm)
└── README.md             # This file
```

## Installation

### Dependencies

```bash
pip3 install --break-system-packages pandas requests tqdm
```

Or if you prefer using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Files

The system expects the following data files (absolute paths configured in [config.py](config.py)):

| File | Path | Description |
|------|------|-------------|
| Movies CSV | `/mnt/nas/sakshipandey/main/projects/Data/item_text.plot160.csv` | 1,682 movies with plot_sum_160 column |
| Ratings | `/mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/ml-100k/u.data` | MovieLens 100K ratings (User 1 has 272 ratings) |
| Prompts | `/mnt/nas/sakshipandey/main/projects/Data/prompts.json` | Prompts with metadata (prompt_id, combo_type, text) |

## Usage

### Basic Usage

Process all prompts:

```bash
python3 main.py
```

### Process Specific Prompts

Process only selected prompts:

```bash
python3 main.py --prompt_ids "0001,0002,0003"
```

### Custom Output Directories

```bash
python3 main.py \
  --prompt_ids "0001,0002,0003" \
  --output_dir ./results \
  --debug_dir ./debug
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompts` | From config | Path to prompts.json |
| `--prompt_ids` | `all` | Comma-separated prompt IDs or 'all' |
| `--output_dir` | `./results` | Output directory for results |
| `--debug_dir` | `./debug` | Debug directory for raw API responses |

## Output

### Directory Structure

```
results/
├── 0001.json                        # Per-prompt results (JSON)
├── 0001.csv                         # Per-prompt results (CSV)
├── 0002.json
├── 0002.csv
├── ...
└── all_pairwise_comparisons.json    # Consolidated output

debug/
├── batch_0001_to_0003_attempt1.raw.txt  # Raw API responses
├── ...
```

### Output Schema

Each prompt generates 9 pairwise comparisons:

```json
{
  "prompt_id": {
    "pairs": [
      {
        "set": "hard",
        "type": "top_2_best",
        "movie1_id": 123,
        "movie1_title": "The Matrix (1999)",
        "movie2_id": 456,
        "movie2_title": "Inception (2010)",
        "winner": 123,
        "reasoning": "The Matrix more directly addresses the philosophical investigation..."
      },
      ... (8 more pairs)
    ]
  }
}
```

## Performance

### Token Budget

- **Candidates**: ~85K tokens (1,682 movies with plot_sum_160)
- **User History**: ~2K tokens (272 ratings)
- **System Prompt**: ~3K tokens
- **Prompts**: ~1K tokens (3-5 prompts batched)
- **Total Input**: ~95K tokens per batch
- **Output**: ~12K tokens (9 pairs × 3-5 prompts)
- **TOTAL**: ~107K tokens **✓ Fits in 400K window**

### Estimated Costs & Time

- **Batch Size**: 3-5 prompts (dynamic based on prompt length)
- **Total API Calls**: 1,000 prompts ÷ 4 = ~250 calls
- **Time per Call**: ~30s (large context)
- **Total Time**: ~2.5 hours for 1,000 prompts
- **Cost (GPT-4o-mini)**: ~$0.020/call × 250 = **~$5.00 total**

### Test Results

```bash
python3 main.py --prompt_ids "0001,0002,0003"
```

Output:
```
[INFO] Loaded 272 ratings for User 1
[INFO] Loaded 1682 movies
[INFO] Estimated candidate tokens: ~84,973
[INFO] Loaded 3 prompts to process

Processing prompts: 100%|████████████████████| 3/3 [01:11<00:00, 23.84s/it]

Summary:
  Total prompts: 3
  Successful: 3
  Failed: 0
  Success rate: 100.0%
```

## Judge Logic

The system implements a sophisticated "Judge" that creates 9 pairwise comparisons per prompt:

### SET 1: HARD (Difficult Trade-offs)

1. **Top 2 Best Matches**: Compare #1 vs #2 best matches
2. **Relevance vs. Quality**: Rank ~5 (High Relevance) vs. Rank ~20 (High Quality/User Favorite)
3. **Opposite Vibes**: Two fitting movies with clashing tones (Dark vs. Funny)

### SET 2: MEDIUM (Moderate Trade-offs)

4. **Exact vs. Metaphor**: Literal match vs. Thematic match
5. **Good vs. Median**: Rank ~10 (Good) vs. Rank ~25 (Median)
6. **History Logic** (conditional on `combo_type`):
   - If `combo_type` contains 'history': Good Match vs. Anti-History
   - If `combo_type` excludes 'history': Good Match vs. History Favorite (5★)

### SET 3: EASY (Clear Winners)

7. **Perfect Match vs. Mismatch**: Right genre, wrong plot
8. **Masterpiece vs. Flop**: High Rated vs. Low Rated (<2.5★)
9. **Specific vs. Cliche**: Unique fit vs. Generic trope

## Configuration

Edit [config.py](config.py) to customize:

```python
# API Configuration
OPENROUTER_API_KEY = "your-api-key-here"
MODEL = "openai/gpt-4o-mini"

# Batch configuration
MIN_PROMPTS_PER_BATCH = 3
MAX_PROMPTS_PER_BATCH = 5

# Retry configuration
MAX_RETRIES = 5
BACKOFF_MULTIPLIER = 2.0
```

## Error Handling

The system implements a robust 3-tier retry strategy:

1. **Attempt 1**: Batch processing (3-5 prompts)
2. **Attempt 2**: Retry batch once
3. **Attempt 3**: Fallback to individual prompts

All errors are logged, and partial results are saved incrementally.

## Key Features

- **Context Efficiency**: Uses `plot_sum_160` (160 chars) instead of full plot text to fit within 400K context window
- **Inline User Ratings**: Annotates movies with User 1's ratings directly in the candidate list
- **Dynamic Batch Sizing**: Adjusts batch size (3-5 prompts) based on prompt length
- **Robust JSON Parsing**: Handles code fences, reasoning output, and various JSON formats
- **Atomic Saves**: Saves results immediately after each successful batch
- **Debug Logging**: Saves raw API responses to debug directory for troubleshooting
- **Progress Tracking**: Real-time progress bar with tqdm

## Example Output

Prompt: "I'm looking for a narrative structured around a central investigation..."

Generated Pairs:
1. **Top 2 Best**: Twelve Monkeys (1995) vs. The Sixth Sense (1999) → Winner: Twelve Monkeys
2. **Relevance vs Quality**: The Usual Suspects (1995) vs. Dead Man Walking (1995) → Winner: The Usual Suspects
3. **Opposite Vibes**: Copycat (1995) vs. Get Shorty (1995) → Winner: Copycat
4. ...

## Reference Code

The implementation is modeled after [top10_openrouter.py](../Data/top10_openrouter.py):
- OpenRouter API integration (lines 184-274)
- JSON parsing with regex (lines 77-132)
- Retry logic with exponential backoff
- Debug output to files

## Troubleshooting

### Missing Dependencies

```bash
pip3 install --break-system-packages pandas requests tqdm
```

### API Key Issues

Set your OpenRouter API key in [config.py](config.py):

```python
OPENROUTER_API_KEY = "sk-or-v1-..."
```

### Rate Limiting

The system automatically handles 429 errors with exponential backoff. If you encounter persistent rate limiting, reduce `MAX_PROMPTS_PER_BATCH` in config.py.

### JSON Parse Errors

Check `debug/` directory for raw API responses. The system uses robust regex-based extraction that handles most formats.

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the documentation in the code comments.

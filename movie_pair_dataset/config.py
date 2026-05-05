"""
Configuration for Context-Heavy Movie Reranker
"""
import os

# Data paths
MOVIES_CSV = "/mnt/nas/sakshipandey/main/projects/Data/item_text.plot160.csv"
RATINGS_FILE = "/mnt/nas/sakshipandey/main/projects/rag-movie-rec/data/ml-100k/u.data"
PROMPTS_JSON = "/mnt/nas/sakshipandey/main/projects/Data/prompts.json"

# Output paths
OUTPUT_DIR = "./results"
DEBUG_DIR = "./debug"
FINAL_OUTPUT = "./results/all_pairwise_comparisons.json"

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "openai/gpt-5-mini"  # Reasoning model - compact GPT-5 for lighter reasoning tasks
API_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Model parameters
MAX_TOKENS_OUTPUT = 16000
TEMPERATURE = 0
TOP_P = 1.0
SEED = 42
TIMEOUT_SECONDS = 600

# Batch configuration
TARGET_INPUT_TOKENS = 320000  # Conservative target (400k - 80k output buffer)
MIN_PROMPTS_PER_BATCH = 1
MAX_PROMPTS_PER_BATCH = 3  # Process up to 3 prompts per batch

# Retry configuration
MAX_RETRIES = 5
BACKOFF_MULTIPLIER = 2.0

# User configuration
USER_ID = 1

# Token estimation
CHARS_PER_TOKEN = 4  # Conservative estimate

# Judge task configuration
PAIRS_PER_PROMPT = 9
COMPARISON_SETS = ["hard", "medium", "easy"]

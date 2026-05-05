#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, subprocess
from typing import List, Optional, Tuple
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm     # <-- NEW

# ---------------- GPU selection helpers ---------------- #
def _read_nvidia_smi() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None

def _parse_visible(devs_env: Optional[str]) -> Optional[list[int]]:
    if not devs_env:
        return None
    cleaned = [d.strip() for d in devs_env.split(",") if d.strip() not in ("", "NUL", "None")]
    try:
        return [int(x) for x in cleaned]
    except Exception:
        return None

def pick_best_cuda_device(min_free_gb: float = 1.0) -> Optional[int]:
    """Return GPU index with maximum free memory (>= min_free_gb), respecting CUDA_VISIBLE_DEVICES."""
    if not torch.cuda.is_available():
        return None
    vis = _parse_visible(os.environ.get("CUDA_VISIBLE_DEVICES"))
    smi = _read_nvidia_smi()
    if not smi:
        if vis:
            return vis[0]
        return 0
    candidates: list[Tuple[int, float]] = []
    for line in smi.strip().splitlines():
        try:
            idx_str, total_str, used_str = [x.strip() for x in line.split(",")]
            idx = int(idx_str)
            if vis is not None and idx not in vis:
                continue
            total = float(total_str)
            used = float(used_str)
            free_gb = (total - used) / 1024.0
            candidates.append((idx, free_gb))
        except Exception:
            continue
    if not candidates:
        return None
    best = max(candidates, key=lambda t: t[1])
    return best[0] if best[1] >= min_free_gb else None

# ---------------- utils ---------------- #
def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def truncate_chars(s: str, max_chars: int) -> str:
    s = normalize_spaces(s)
    return (s[:max_chars-3] + "...") if len(s) > max_chars else s

def pick_source_text(row: pd.Series, plot_col: str | None) -> str:
    """Prefer plot; if missing/empty, fall back to overview."""
    if plot_col and plot_col in row and isinstance(row[plot_col], str) and row[plot_col].strip():
        return row[plot_col]
    if "plot" in row and isinstance(row["plot"], str) and row["plot"].strip():
        return row["plot"]
    if "overview" in row and isinstance(row["overview"], str) and row["overview"].strip():
        return row["overview"]
    return ""

# ---------------- model loading ---------------- #
def resolve_device(device_arg: str) -> torch.device:
    device_arg = (device_arg or "auto").lower()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if device_arg == "cuda":
            return torch.device("cuda:0")
        try:
            idx = int(device_arg.split(":")[1])
            return torch.device(f"cuda:{idx}")
        except Exception:
            return torch.device("cuda:0")
    best = pick_best_cuda_device(min_free_gb=1.0)
    if best is not None:
        return torch.device(f"cuda:{best}")
    return torch.device("cpu")

def load_model(model_name: str, models_root: str, device: torch.device):
    local_path = os.path.join(models_root, model_name)
    load_id = local_path if os.path.isdir(local_path) else model_name
    print(f"Loading model from: {load_id}")
    tok = AutoTokenizer.from_pretrained(load_id)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(load_id, torch_dtype=dtype)
    model.to(device)
    return tok, model

@torch.no_grad()
def summarize_batch(
    texts: List[str],
    tok: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_new_tokens: int,
    min_new_tokens: int,
    src_max_len: int,
    num_beams: int = 4,
) -> List[str]:
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=src_max_len,
    ).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        length_penalty=1.0,
        early_stopping=True,
    )
    return [tok.decode(o, skip_special_tokens=True) for o in out]

# ---------------- main ---------------- #
def main():
    ap = argparse.ArgumentParser(description="Summarize plots once and cache to CSV (<=160 chars).")
    ap.add_argument("--in_csv",  default="item_text.csv")
    ap.add_argument("--out_csv", default="item_text.plot160.csv")
    ap.add_argument("--plot_col", default=None)
    ap.add_argument("--model", default="facebook/bart-large-cnn")
    ap.add_argument("--models_root", default="/mnt/nas/sakshipandey/main/models")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_chars", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--src_max_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--min_new_tokens", type=int, default=20)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    tok, model = load_model(args.model, args.models_root, device)

    df = pd.read_csv(args.in_csv)
    if "plot_sum_160" not in df.columns:
        df["plot_sum_160"] = ""

    mask = df["plot_sum_160"].astype(str).str.strip().eq("") if args.skip_existing else pd.Series([True]*len(df), index=df.index)
    idxs = df.index[mask].tolist()
    total_rows = len(idxs)
    print(f"Summarizing {total_rows} rows (out of {len(df)})…")

    # --- progress bar integration --- #
    with tqdm(total=total_rows, desc="Summarizing", ncols=100) as pbar:
        for start in range(0, total_rows, args.batch_size):
            end = min(start + args.batch_size, total_rows)
            batch_idx = idxs[start:end]
            src_texts = [pick_source_text(df.loc[i], args.plot_col) for i in batch_idx]
            to_summarize = [normalize_spaces(t) if t else " " for t in src_texts]

            summaries = summarize_batch(
                to_summarize, tok, model, device,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                src_max_len=args.src_max_len,
                num_beams=4,
            )
            summaries = [
                truncate_chars(s if src_texts[j] else "", args.max_chars)
                for j, s in enumerate(summaries)
            ]
            df.loc[batch_idx, "plot_sum_160"] = summaries
            pbar.update(len(batch_idx))

    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote: {args.out_csv}")

if __name__ == "__main__":
    main()

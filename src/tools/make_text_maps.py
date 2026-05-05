#!/usr/bin/env python
"""
Build the text maps required by bge_reranker_eval.py

Examples:
  python -m src.tools.make_text_maps \
    --prompts_dir /path/to/prompts_dir \
    --movies_corpus /path/to/movies.parquet \
    --out_prompts artifacts/prompts/prompt_text.parquet \
    --out_movies  artifacts/movies/movie_text.parquet
"""
import argparse, os, json
import pandas as pd

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".jsonl", ".json"}:
        # support JSONL and JSON list
        with open(path, "r") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":
                return pd.DataFrame(json.load(f))
            else:
                return pd.read_json(path, lines=True)
    if ext in {".csv", ".tsv"}:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file type: {path}")

def guess_text_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns: {list(df.columns)[:20]}...")

def build_prompt_map(prompts_dir: str) -> pd.DataFrame:
    # Look for prompts.json (or parquet/csv) with prompt_id & prompt_text
    cand_files = [
        "prompts.parquet","prompts.jsonl","prompts.json","prompts.csv",
        "prompt_text.parquet","prompt_text.jsonl","prompt_text.csv",
    ]
    found = None
    for f in cand_files:
        p = os.path.join(prompts_dir, f)
        if os.path.exists(p):
            found = p; break
    if found is None:
        raise FileNotFoundError(f"Could not find prompts file in {prompts_dir}. "
                                f"Tried: {cand_files}")

    df = read_table(found)
    id_col   = "prompt_id" if "prompt_id" in df.columns else "id"
    text_col = guess_text_col(df, ["prompt_text","text","query","q_text","prompt"])
    out = df[[id_col, text_col]].dropna().drop_duplicates().rename(
        columns={id_col:"prompt_id", text_col:"text"}
    )
    return out

# inside build_movie_map(...)
def build_movie_map(csv_path, id_col=None, text_cols=None):
    import pandas as pd
    df = pd.read_csv(csv_path)

    if id_col is None:
        id_candidates = [
            "movie_id","mid","doc_id","id",  # old
            "movieId","iid","TMDbID"         # <-- add these
        ]
        id_col = next((c for c in id_candidates if c in df.columns), None)
        if id_col is None:
            raise ValueError(f"Could not find a movie id column among {id_candidates}")

    if text_cols is None:
        # pick whatever exists
        candidates = ["title","overview","plot","genres","genres_tmdb","tagline"]
        text_cols = [c for c in candidates if c in df.columns]
        if not text_cols:
            raise ValueError("No text columns found in movie table")

    df["_text"] = df[text_cols].astype(str).replace("nan","").agg(" ".join, axis=1)
    out = (df[[id_col, "_text"]]
           .dropna()
           .drop_duplicates(subset=[id_col])
           .rename(columns={id_col: "movie_id", "_text": "movie_text"}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_dir", required=True,
                    help="Folder containing prompts.{parquet|jsonl|csv} with prompt_id & prompt_text")
    ap.add_argument("--movies_corpus", required=True,
                    help="File (parquet/jsonl/csv) with movie_id & text/overview/plot")
    ap.add_argument("--out_prompts", default="artifacts/prompts/prompt_text.parquet")
    ap.add_argument("--out_movies", default="artifacts/movies/movie_text.parquet")
    ap.add_argument("--movie_id_col", default=None)
    ap.add_argument("--movie_text_cols", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prompts), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_movies), exist_ok=True)

    pmap = build_prompt_map(args.prompts_dir)
    mmap = build_movie_map(args.movies_corpus,
                       id_col=args.movie_id_col,
                       text_cols=args.movie_text_cols)

    pmap.to_parquet(args.out_prompts, index=False)
    mmap.to_parquet(args.out_movies, index=False)

    print(f"Wrote prompts map: {args.out_prompts}  (rows={len(pmap)})")
    print(f"Wrote movies  map: {args.out_movies}   (rows={len(mmap)})")

if __name__ == "__main__":
    main()

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import re

# ---------- simple cleaners/splitters ----------
_CITE_LINE = re.compile(r"^\s*\^.*$", re.MULTILINE)
_CITE_ERR  = re.compile(r"(?i)Cite error:.*")
_BRACKETS  = re.compile(r"\[[0-9]+\]")
_TEMPLATES = re.compile(r"\{\{.*?\}\}")
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

def clean_text(s: str) -> str:
    s = s or ""
    s = _CITE_LINE.sub("", s)
    s = _CITE_ERR.sub("", s)
    s = _BRACKETS.sub("", s)
    s = _TEMPLATES.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    sents: List[str] = []
    for p in paras:
        if len(p.split()) < 60:
            sents.append(p)
        else:
            sents.extend([s.strip() for s in _SENT_SPLIT.split(p) if s.strip()])
    return sents

def _year_from_date(s: str) -> str:
    s = str(s or "")
    return s[:4] if len(s) >= 4 else ""

# ---------- chunker ----------
META_KEEP = ["movieId","title","original_title","original_language","release_date","TMDbID","runtime",
             "type","adult","production_countries","spoken_languages","poster_path"]

@dataclass
class ChunkSpec:
    """Single full-movie chunk (overview + plot)."""
    min_words: int = 0  # 0 = keep all

def _combined_text_content_only(r: pd.Series) -> str:
    """Overview + Plot only (no metadata header in text)."""
    ov = clean_text(str(r.get("overview", "") or ""))
    pl = clean_text(str(r.get("plot", "") or ""))
    parts = []
    if ov:
        parts.append("Overview:\n" + ov)
    if pl:
        parts.append("Plot:\n" + pl)
    return "\n\n".join(parts).strip()

def _base_meta(r: pd.Series) -> Dict:
    out: Dict = {}
    for c in META_KEEP:
        if c in r:
            out[c] = r[c]
    return out

def build_chunks_for_record(r: pd.Series, spec: ChunkSpec) -> List[Dict]:
    if "movieId" not in r:
        raise KeyError("movieId column is required")
    mid = int(r["movieId"])
    # normalize text cols
    for col in ("overview", "plot"):
        if col in r:
            r[col] = str(r[col] or "")

    text = _combined_text_content_only(r)
    if not text:
        return []

    n_words = len(text.split())
    if spec.min_words > 0 and n_words < spec.min_words:
        return []

    row = {
        **_base_meta(r),
        "chunkId": str(mid),        # single id field, equals movieId as string
        "movieId": mid,
        "text": text,               # content only (no duplicate metadata header)
        "n_words": n_words,
        "n_chars": len(text),
        "n_tokens": int(round(len(text) / 4.0)),
    }
    return [row]

def build_chunks(df: pd.DataFrame, spec: ChunkSpec) -> pd.DataFrame:
    for col in ("overview", "plot"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    out: List[Dict] = []
    for _, r in df.iterrows():
        out.extend(build_chunks_for_record(r, spec))
    return pd.DataFrame(out)

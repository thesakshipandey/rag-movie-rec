# src/chunking/splitters.py
import re
from typing import List

# --- light cleanup for wiki-ish noise ---
_CITE_LINE = re.compile(r"^\s*\^.*$", re.MULTILINE)
_CITE_ERR  = re.compile(r"(?i)Cite error:.*")
_BRACKETS  = re.compile(r"\[[0-9]+\]")
_TEMPLATES = re.compile(r"\{\{.*?\}\}")

def clean_text(s: str) -> str:
    s = s or ""
    s = _CITE_LINE.sub("", s)
    s = _CITE_ERR.sub("", s)
    s = _BRACKETS.sub("", s)
    s = _TEMPLATES.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# src/embeddings/backends.py
from __future__ import annotations
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import os
import torch
import numpy as np
import math


@dataclass
class EncodeResult:
    vectors: List[List[float]]
    dim: int
    model_id: str


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                         # [B,1]
    return summed / counts


# ---------------- MiniLM (SentenceTransformers) ----------------

class MiniLMEncoder:
    def __init__(self, model: str, device: Optional[str] = None, local_files_only: bool = True):
        from sentence_transformers import SentenceTransformer
        self.model_id = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # SentenceTransformer doesn't expose local_files_only everywhere; rely on local path
        self.model = SentenceTransformer(model, device=self.device)
        try:
            self.dim = int(getattr(self.model, "get_sentence_embedding_dimension")())
        except Exception:
            self.dim = 384

    def encode(self, texts: List[str], batch_size: int = 256, normalize: bool = True) -> EncodeResult:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32)
        if normalize:
            vecs = _l2_normalize(vecs)
        return EncodeResult([v.tolist() for v in vecs], self.dim, self.model_id)


# ---------------- Gemma (HF Transformers) ----------------

class GemmaEncoder:
    """
    google/embeddinggemma-300m (or local folder) using transformers.
    We force the slow SentencePiece tokenizer to avoid fast tokenizer issues.
    """
    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        max_length: int = 2048,
        local_files_only: bool = True,
    ):
        os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"

        try:
            from transformers.models.gemma import GemmaTokenizer  # slow
        except Exception:
            from transformers import GemmaTokenizer

        from transformers import AutoModel

        self.model_id = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self.tok = GemmaTokenizer.from_pretrained(
            model,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model,
            local_files_only=local_files_only
        ).to(self.device)
        self.model.eval()

        self.dim = int(getattr(self.model.config, "hidden_size",
                               getattr(self.model.config, "hidden_dim", 1024)))

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> EncodeResult:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            hidden = out.last_hidden_state  # [B, T, H]
            mask = enc.attention_mask.unsqueeze(-1)  # [B, T, 1]
            summed = (hidden * mask).sum(dim=1)      # [B, H]
            counts = mask.sum(dim=1).clamp(min=1)    # [B, 1]
            mean_pooled = summed / counts
            vecs = mean_pooled.detach().cpu().numpy().astype(np.float32)
            if normalize:
                vecs = _l2_normalize(vecs)
            all_vecs.append(vecs)

        vecs = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, self.dim), dtype=np.float32)
        return EncodeResult([v.tolist() for v in vecs], self.dim, self.model_id)


# ---------------- Qwen3-Embedding-8B (SentenceTransformers) ----------------

class QwenEncoder:
    """
    Qwen3-Embedding-8B via Hugging Face Transformers.
    - mean pooling over last_hidden_state
    - bfloat16 by default
    - optional token-window streaming for long texts
    """
    def __init__(
        self,
        model: str,
        device: Optional[str] = None,          # e.g., "cuda:0" or None
        max_length: int = 8192,                # per-window token cap
        local_files_only: bool = True,
        window_tokens: Optional[int] = None,   # if set, split inputs by this many tokens (e.g., 8192, 12000)
        stride_tokens: int = 256,              # overlap between windows
        dtype: str = "bfloat16",               # "bfloat16" or "float16" or "float32"
    ):
        self.model_id = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.window_tokens = window_tokens     # if None => single pass truncate
        self.stride_tokens = stride_tokens

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(dtype, torch.bfloat16)

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(
            model,
            use_fast=False,                # safer with long context
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # model
        if self.device.startswith("cuda"):
            # put whole model on this CUDA device
            self.model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            ).to(self.device)
        else:
            # CPU fallback
            self.model = AutoModel.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )
        self.model.eval()

        # hidden size
        self.dim = int(getattr(self.model.config, "hidden_size",
                               getattr(self.model.config, "hidden_dim", 4096)))

    @torch.no_grad()
    def _encode_windowed(self, texts: List[str], batch_size: int, normalize: bool) -> EncodeResult:
        """Tokenize per-sample, slide over long sequences in windows, average the window embeddings."""
        outs = []
        for txt in texts:
            # tokenize once to ids
            enc_all = self.tok(txt, return_tensors="pt", add_special_tokens=True)
            input_ids = enc_all["input_ids"][0]  # [T]
            attn_full = enc_all.get("attention_mask", torch.ones_like(input_ids))

            # split into windows
            T = input_ids.size(0)
            win = self.window_tokens or self.max_length
            step = max(1, win - self.stride_tokens)
            vecs = []
            for start in range(0, T, step):
                end = min(start + win, T)
                ids = input_ids[start:end].unsqueeze(0)       # [1, t]
                am  = attn_full[start:end].unsqueeze(0)
                ids = ids.to(self.device)
                am  = am.to(self.device)

                out = self.model(input_ids=ids, attention_mask=am)
                pooled = _mean_pool(out.last_hidden_state, am)   # [1,H]
                vecs.append(pooled.detach().cpu())

                if end >= T: break

            if len(vecs) == 0:
                # empty? produce zeros
                outs.append(torch.zeros(self.dim))
            else:
                v = torch.stack(vecs, dim=0).mean(dim=0).squeeze(0)  # [H]
                outs.append(v.to(torch.float32))

        mat = torch.stack(outs, dim=0).numpy().astype(np.float32)
        if normalize:
            mat = _l2_normalize(mat)
        return EncodeResult([v.tolist() for v in mat], self.dim, self.model_id)

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 4, normalize: bool = True) -> EncodeResult:
        if self.window_tokens and self.window_tokens > 0:
            # per-sample windowing (no batching to keep memory stable)
            return self._encode_windowed(texts, batch_size=1, normalize=normalize)

        # simple batched path with truncation to max_length
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])  # [B,H]
            vecs = pooled.detach().to(torch.float32).cpu().numpy()
            if normalize:
                vecs = _l2_normalize(vecs)
            all_vecs.append(vecs)

        if all_vecs:
            mat = np.concatenate(all_vecs, axis=0)
        else:
            mat = np.zeros((0, self.dim), dtype=np.float32)
        return EncodeResult([v.tolist() for v in mat], self.dim, self.model_id)

# ---------------- Factory ----------------

def load_encoder(encoder: str, model: str, device: Optional[str] = None, **kwargs):
    encoder = (encoder or "").lower()
    if encoder == "qwen":
        return QwenEncoder(
            model=model,
            device=device,
            max_length=kwargs.get("max_length", 8192),
            local_files_only=kwargs.get("local_files_only", True),
            window_tokens=kwargs.get("window_tokens", None),  # e.g., 12000 for long texts
            stride_tokens=kwargs.get("stride_tokens", 256),
            dtype=kwargs.get("dtype", "bfloat16"),
        )
    if encoder == "gemma":
        return GemmaEncoder(
            model=model,
            device=device,
            max_length=kwargs.get("max_length", 2048),
            local_files_only=kwargs.get("local_files_only", True),
        )
    if encoder == "minilm":
        return MiniLMEncoder(
            model=model,
            device=device,
            local_files_only=kwargs.get("local_files_only", True),
        )
    raise ValueError(f"Unknown encoder '{encoder}'. Use 'qwen', 'gemma', or 'minilm'.")

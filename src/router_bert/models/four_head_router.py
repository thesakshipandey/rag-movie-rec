"""
Four-head router model with per-expert attention.

Text → BERT encoder → 4 expert queries → attention → weights
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class FourHeadRouter(nn.Module):
    """
    Text-conditioned router with 4 expert-specific attention heads.
    
    Architecture:
    1. BERT/DistilBERT encoder produces token embeddings H [B, T, d]
    2. Learn 4 expert queries Q [4, d]
    3. Compute attention per expert: scores_k = H @ q_k / sqrt(d)
    4. Get context vectors: context_k = sum(attn_k * H)
    5. Score contexts to logits → softmax → weights [B, 4]
    """
    
    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        d_model: Optional[int] = None,
        freeze_encoder: bool = True,
        max_length: int = 256,
    ):
        """
        Initialize router model.
        
        Args:
            encoder_name: HuggingFace model name
            d_model: Hidden dimension (inferred from encoder if None)
            freeze_encoder: Whether to freeze encoder weights
            max_length: Max sequence length for tokenization
        """
        super().__init__()
        
        self.encoder_name = encoder_name
        self.max_length = max_length
        self.freeze_encoder = freeze_encoder
        
        # Load encoder and tokenizer
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Infer hidden dimension
        if d_model is None:
            d_model = self.encoder.config.hidden_size
        self.d_model = d_model
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"Encoder '{encoder_name}' frozen (not trainable)")
        else:
            print(f"Encoder '{encoder_name}' will be fine-tuned")
        
        # Learnable expert queries [4, d]
        # Initialize with small random values
        self.expert_queries = nn.Parameter(
            torch.randn(4, d_model) * 0.02
        )
        
        # Scorer: maps context [d] to logit [1]
        self.scorer = nn.Linear(d_model, 1)
        
        print(f"FourHeadRouter initialized: d_model={d_model}, freeze_encoder={freeze_encoder}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: [B, T] token ids
            attention_mask: [B, T] attention mask
            return_attn: Whether to return attention weights
        
        Returns:
            weights: [B, 4] expert weights (softmax)
            attn_weights: [B, 4, T] attention per expert (if return_attn=True)
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use last hidden state [B, T, d]
        H = encoder_outputs.last_hidden_state
        B, T, d = H.shape
        
        # Compute attention scores for each expert
        # scores: [B, 4, T] = einsum('btd,kd->bkt', H, Q) / sqrt(d)
        scores = torch.einsum('btd,kd->bkt', H, self.expert_queries) / math.sqrt(d)
        
        # Mask out padding tokens
        # attention_mask: [B, T] → [B, 1, T]
        mask = attention_mask.unsqueeze(1)  # [B, 1, T]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over tokens to get attention weights [B, 4, T]
        attn = F.softmax(scores, dim=-1)
        
        # Handle NaN from all-padding sequences (shouldn't happen but be safe)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        # Compute context vectors: [B, 4, d] = einsum('bkt,btd->bkd', attn, H)
        context = torch.einsum('bkt,btd->bkd', attn, H)
        
        # Score each context to get logits [B, 4]
        logits = self.scorer(context).squeeze(-1)  # [B, 4]
        
        # Softmax to get weights
        weights = F.softmax(logits, dim=-1)  # [B, 4]
        
        if return_attn:
            return weights, attn
        else:
            return weights, None
    
    def forward_texts(
        self,
        texts: List[str],
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convenience method to encode text strings.
        
        Args:
            texts: List of prompt strings
            return_attn: Whether to return attention weights
        
        Returns:
            weights: [B, 4] expert weights
            attn_weights: [B, 4, T] attention per expert (if return_attn=True), else None
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward
        weights, attn = self.forward(input_ids, attention_mask, return_attn=return_attn)
        
        return weights, attn
    
    def compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of weight distribution.
        
        Args:
            weights: [B, 4] expert weights
        
        Returns:
            entropy: [B] entropy per sample
        """
        # H(w) = -sum(w * log(w))
        # Add small epsilon for numerical stability
        log_weights = torch.log(weights + 1e-10)
        entropy = -torch.sum(weights * log_weights, dim=-1)
        return entropy
    
    def save_pretrained(self, save_dir: str):
        """Save model, tokenizer, and config."""
        import os
        import json
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, 'tokenizer'))
        
        # Save config
        config = {
            'encoder_name': self.encoder_name,
            'd_model': self.d_model,
            'freeze_encoder': self.freeze_encoder,
            'max_length': self.max_length,
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = 'cpu'):
        """Load model from directory."""
        import os
        import json
        
        # Load config
        with open(os.path.join(load_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(
            encoder_name=config['encoder_name'],
            d_model=config['d_model'],
            freeze_encoder=config['freeze_encoder'],
            max_length=config['max_length'],
        )
        
        # Load state dict
        state_dict = torch.load(
            os.path.join(load_dir, 'pytorch_model.bin'),
            map_location=device,
        )
        model.load_state_dict(state_dict)
        
        # Load tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(load_dir, 'tokenizer')
        )
        
        model.to(device)
        print(f"Model loaded from {load_dir}")
        
        return model


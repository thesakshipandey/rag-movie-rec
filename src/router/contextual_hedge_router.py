"""
Contextual Hedge Router for expert selection.

Implements an MLP-based router that takes contextual features (prompt metadata,
emotion distributions, etc.) and outputs a softmax distribution over 4 experts:
- Alpha (dense/semantic retrieval)
- Beta (BM25/lexical retrieval)
- Gamma (LGCN/collaborative filtering)
- Delta (emotion-based ranking)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
import numpy as np


class ContextualHedgeRouter(nn.Module):
    """
    MLP-based router that predicts expert weights from contextual features.
    
    Architecture:
        context_features -> feature_encoder -> MLP -> logits -> softmax -> weights
    
    The router learns to select/combine experts based on:
    - Plutchik emotion distribution (8 dimensions)
    - Ground truth mix weights (4 dimensions, optional as input)
    - Context features (length, category, difficulty, etc.)
    """
    
    def __init__(
        self,
        d_context: int = 128,
        d_hidden: int = 256,
        num_experts: int = 4,
        dropout: float = 0.2,
        temperature: float = 1.0,
        use_mix_prior: bool = False,
        mix_prior_strength: float = 1.0
    ):
        """
        Args:
            d_context: Dimension of encoded context features
            d_hidden: Hidden layer size for MLP
            num_experts: Number of experts (default 4)
            dropout: Dropout probability
            temperature: Initial softmax temperature (learnable)
            use_mix_prior: Whether to incorporate ground truth mix weights as prior
            mix_prior_strength: Strength of mix prior (if used)
        """
        super().__init__()
        
        self.d_context = d_context
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.use_mix_prior = use_mix_prior
        self.mix_prior_strength = mix_prior_strength
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(d_context, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(d_hidden // 2, num_experts)
        )
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        
    def forward(
        self,
        context_features: torch.Tensor,
        mix_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            context_features: [B, d_context] encoded context features
            mix_prior: [B, 4] optional ground truth mix weights (for training with prior)
            
        Returns:
            expert_weights: [B, num_experts] softmax distribution over experts
        """
        # Get logits from MLP
        logits = self.mlp(context_features)  # [B, num_experts]
        
        # Optionally incorporate mix prior
        if self.use_mix_prior and mix_prior is not None:
            # Add log-prior as bias (assuming mix_prior is already normalized)
            log_prior = torch.log(mix_prior.clamp(min=1e-6))
            logits = logits + self.mix_prior_strength * log_prior
        
        # Apply temperature and softmax
        temp = self.temperature.clamp(min=0.1, max=10.0)
        expert_weights = F.softmax(logits / temp, dim=-1)
        
        return expert_weights
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return float(self.temperature.item())


class ContextFeatureEncoder(nn.Module):
    """
    Encodes various context features into a fixed-dimensional vector.
    
    Handles:
    - Plutchik emotion distribution (8D)
    - Mix weights (4D, optional)
    - Categorical features (category, difficulty, etc.)
    - Numerical features (length, num_genre_terms, etc.)
    """
    
    def __init__(
        self,
        d_out: int = 128,
        d_emotion: int = 8,
        d_mix: int = 4,
        category_vocab: Optional[List[str]] = None,
        difficulty_vocab: Optional[List[str]] = None,
        primary_expert_vocab: Optional[List[str]] = None,
        length_bucket_vocab: Optional[List[str]] = None,
        persona_style_vocab: Optional[List[str]] = None,
        include_mix_features: bool = False,
        dropout: float = 0.1
    ):
        """
        Args:
            d_out: Output dimension
            d_emotion: Emotion vector dimension (default 8 for Plutchik)
            d_mix: Mix weights dimension (default 4 for alpha/beta/gamma/delta)
            category_vocab: List of categories for embedding
            difficulty_vocab: List of difficulties for embedding
            include_mix_features: Whether to include mix weights as input features
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_out = d_out
        self.d_emotion = d_emotion
        self.d_mix = d_mix
        self.include_mix_features = include_mix_features
        
        # Embedding dimensions
        d_embed = 16
        
        # Categorical embeddings
        self.category_vocab = category_vocab or ["mood", "theme", "genre", "plot", "UNK"]
        self.difficulty_vocab = difficulty_vocab or ["easy", "medium", "hard", "UNK"]
        self.primary_expert_vocab = primary_expert_vocab or ["emotion", "semantic", "cf", "lexical", "UNK"]
        self.length_bucket_vocab = length_bucket_vocab or ["short", "medium", "long", "UNK"]
        self.persona_style_vocab = persona_style_vocab or ["casual", "formal", "humorous", "conversational", "technical", "UNK"]
        
        self.category_embed = nn.Embedding(len(self.category_vocab), d_embed)
        self.difficulty_embed = nn.Embedding(len(self.difficulty_vocab), d_embed)
        self.primary_expert_embed = nn.Embedding(len(self.primary_expert_vocab), d_embed)
        self.length_bucket_embed = nn.Embedding(len(self.length_bucket_vocab), d_embed)
        self.persona_style_embed = nn.Embedding(len(self.persona_style_vocab), d_embed)
        
        # Calculate input dimension
        # Emotion (8) + numerical features + categorical embeddings + optional mix weights
        num_numerical = 7  # length_words, num_genre_terms, has_negation, has_year, has_actor, mentions_movie, multi_intent
        num_categorical_embed = 5 * d_embed  # 5 categorical features
        
        d_in = d_emotion + num_numerical + num_categorical_embed
        if include_mix_features:
            d_in += d_mix
        
        # Projection to output dimension
        self.projector = nn.Sequential(
            nn.Linear(d_in, d_out * 2),
            nn.LayerNorm(d_out * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_dict: Dictionary of feature tensors
                - 'emotion': [B, 8] Plutchik distribution
                - 'mix_weights': [B, 4] mix weights (optional)
                - 'category_idx': [B] category indices
                - 'difficulty_idx': [B] difficulty indices
                - 'primary_expert_idx': [B] primary expert indices
                - 'length_bucket_idx': [B] length bucket indices
                - 'persona_style_idx': [B] persona style indices
                - 'numerical': [B, num_numerical] numerical features
                
        Returns:
            context_encoding: [B, d_out]
        """
        feature_list = []
        
        # Emotion distribution
        if 'emotion' in features_dict:
            feature_list.append(features_dict['emotion'])
        else:
            # Fallback: uniform distribution
            B = next(iter(features_dict.values())).shape[0]
            feature_list.append(torch.ones(B, self.d_emotion, device=next(iter(features_dict.values())).device) / self.d_emotion)
        
        # Categorical embeddings
        if 'category_idx' in features_dict:
            feature_list.append(self.category_embed(features_dict['category_idx']))
        if 'difficulty_idx' in features_dict:
            feature_list.append(self.difficulty_embed(features_dict['difficulty_idx']))
        if 'primary_expert_idx' in features_dict:
            feature_list.append(self.primary_expert_embed(features_dict['primary_expert_idx']))
        if 'length_bucket_idx' in features_dict:
            feature_list.append(self.length_bucket_embed(features_dict['length_bucket_idx']))
        if 'persona_style_idx' in features_dict:
            feature_list.append(self.persona_style_embed(features_dict['persona_style_idx']))
        
        # Numerical features
        if 'numerical' in features_dict:
            feature_list.append(features_dict['numerical'])
        
        # Optional mix weights
        if self.include_mix_features and 'mix_weights' in features_dict:
            feature_list.append(features_dict['mix_weights'])
        
        # Concatenate all features
        x = torch.cat(feature_list, dim=-1)
        
        # Project to output dimension
        context_encoding = self.projector(x)
        
        return context_encoding


class ContextualHedgeRouterWithEncoder(nn.Module):
    """
    Complete router system: feature encoder + contextual hedge router.
    """
    
    def __init__(
        self,
        d_context: int = 128,
        d_hidden: int = 256,
        num_experts: int = 4,
        dropout: float = 0.2,
        temperature: float = 1.0,
        use_mix_prior: bool = False,
        mix_prior_strength: float = 1.0,
        encoder_kwargs: Optional[Dict] = None
    ):
        """
        Args:
            d_context: Context encoding dimension
            d_hidden: Router MLP hidden size
            num_experts: Number of experts
            dropout: Dropout probability
            temperature: Softmax temperature
            use_mix_prior: Whether to use mix prior
            mix_prior_strength: Strength of mix prior
            encoder_kwargs: Additional kwargs for ContextFeatureEncoder
        """
        super().__init__()
        
        encoder_kwargs = encoder_kwargs or {}
        encoder_kwargs['d_out'] = d_context
        encoder_kwargs['dropout'] = dropout
        
        self.encoder = ContextFeatureEncoder(**encoder_kwargs)
        self.router = ContextualHedgeRouter(
            d_context=d_context,
            d_hidden=d_hidden,
            num_experts=num_experts,
            dropout=dropout,
            temperature=temperature,
            use_mix_prior=use_mix_prior,
            mix_prior_strength=mix_prior_strength
        )
        
    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        mix_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features_dict: Dictionary of feature tensors (see ContextFeatureEncoder)
            mix_prior: [B, 4] optional ground truth mix weights
            
        Returns:
            expert_weights: [B, num_experts] softmax distribution
        """
        context_encoding = self.encoder(features_dict)
        expert_weights = self.router(context_encoding, mix_prior=mix_prior)
        return expert_weights
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.router.get_temperature()



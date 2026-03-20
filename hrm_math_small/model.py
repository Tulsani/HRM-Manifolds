"""
model.py
--------
HyperbolicReasoningStudent — the small model we are training.

Architecture overview:
  ┌─────────────────────────────────────────────────────┐
  │  Input: question + reasoning step tokens            │
  │                                                     │
  │  Token embedding (Euclidean, d_model)               │
  │       │                                             │
  │       ▼                                             │
  │  expmap0  ──► Poincaré ball                         │
  │       │                                             │
  │  N × HyperbolicTransformerLayer                     │
  │    ├── HyperbolicAttention  (geometry-aware)        │
  │    ├── residual via Möbius add                      │
  │    └── HyperbolicMLP        (on-manifold FFN)       │
  │       │                                             │
  │  logmap0  ──► Euclidean                             │
  │       │                                             │
  │  LM head (linear → vocab logits)                    │
  └─────────────────────────────────────────────────────┘

Key design choices:
  - Residuals use Möbius addition (stays on manifold)
  - LayerNorm operates in tangent space (logmap → norm → expmap)
  - The curvature c is a learned scalar — the model finds the right
    geometry for the task rather than us hardcoding it
  - Final projection back to Euclidean for the LM head (practical
    necessity — cross-entropy loss is Euclidean)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperbolic import (
    HypLinear,
    HyperbolicMLP,
    HyperbolicAttention,
    TangentSpaceAttention,
    expmap0,
    logmap0,
    mobius_add,
    clamp_to_ball,
)


# ---------------------------------------------------------------------------
# Hyperbolic Layer Norm
# ---------------------------------------------------------------------------

class HyperbolicLayerNorm(nn.Module):
    """
    LayerNorm adapted for the Poincaré ball.
    Strategy: lift to tangent space → normalize → project back.
    This preserves the manifold structure while stabilizing training.
    """
    def __init__(self, d_model: int, c: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.c    = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: points on ball → lift to tangent → normalize → project back
        x_tan = logmap0(x, self.c)
        x_tan = self.norm(x_tan)
        return expmap0(x_tan, self.c)


# ---------------------------------------------------------------------------
# Single Transformer Layer (fully hyperbolic)
# ---------------------------------------------------------------------------

class HyperbolicTransformerLayer(nn.Module):
    """
    attn_type: 'tangent' (default, fast, no OOM) or 'geodesic' (exact, slow for long seqs)
    """
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, c: float = 1.0,
                 dropout: float = 0.1, attn_type: str = 'tangent'):
        super().__init__()
        AttnClass = TangentSpaceAttention if attn_type == 'tangent' else HyperbolicAttention
        self.attn  = AttnClass(d_model, n_heads, c=c, dropout=dropout)
        self.ffn   = HyperbolicMLP(d_model, ffn_dim, c=c, dropout=dropout)
        self.norm1 = HyperbolicLayerNorm(d_model, c=c)
        self.norm2 = HyperbolicLayerNorm(d_model, c=c)
        self.drop  = nn.Dropout(dropout)
        self.c     = c

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, d_model) — points on ball

        # Self-attention with Möbius residual
        attn_out = self.attn(self.norm1(x), mask=mask)
        # Möbius add for residual (stays on manifold)
        x = mobius_add(x, attn_out, self.c)
        x = clamp_to_ball(x, self.c)

        # FFN with Möbius residual
        ffn_out = self.ffn(logmap0(self.norm2(x), self.c))  # FFN expects Euclidean input
        ffn_out = expmap0(ffn_out, self.c)
        x = mobius_add(x, ffn_out, self.c)
        x = clamp_to_ball(x, self.c)

        return x


# ---------------------------------------------------------------------------
# Full Student Model
# ---------------------------------------------------------------------------

class HyperbolicReasoningStudent(nn.Module):
    """
    ~100M parameter hyperbolic reasoning model.

    Config for ~100M params:
      vocab_size = 32000  (match teacher tokenizer)
      d_model    = 512
      n_heads    = 8
      n_layers   = 8
      ffn_dim    = 2048
      max_seq_len= 512

    Parameter count:
      embedding : 32000 × 512       = 16.4M
      per layer : attn(~1M) + ffn(~2M) + norms(~4K) ≈ 3M × 8 layers = 24M
      lm_head   : 512 × 32000       = 16.4M
      total     ≈ 57M  (tune n_layers/d_model to hit 100M target)
    """

    def __init__(
        self,
        vocab_size:   int   = 32000,
        d_model:      int   = 512,
        n_heads:      int   = 8,
        n_layers:     int   = 8,
        ffn_dim:      int   = 2048,
        max_seq_len:  int   = 512,
        dropout:      float = 0.1,
        c_init:       float = 1.0,   # initial curvature — learned during training
        tie_weights:  bool  = True,  # tie embedding and lm_head weights
    ):
        super().__init__()

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len

        # Learnable curvature — let the model find its own geometry
        # Constrained positive via softplus in forward
        self.log_c = nn.Parameter(torch.tensor(math.log(c_init)))

        # Token embedding (starts Euclidean, lifted to ball in forward)
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            HyperbolicTransformerLayer(
                d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim,
                c=c_init, dropout=dropout, attn_type='tangent',
            )
            for _ in range(n_layers)
        ])

        # Final norm + LM head (back in Euclidean)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.embed.weight  # weight tying saves 16M params

        self._init_weights()

    @property
    def c(self) -> float:
        """Current curvature — always positive."""
        return F.softplus(self.log_c).item()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)

    def forward(
        self,
        input_ids:      torch.Tensor,               # (B, T)
        attention_mask: torch.Tensor | None = None, # (B, T)  1=attend, 0=ignore
        labels:         torch.Tensor | None = None, # (B, T)  for LM loss
    ) -> dict:
        B, T = input_ids.shape
        device = input_ids.device

        # Current curvature (learned, always positive)
        c = F.softplus(self.log_c)

        # Embeddings (Euclidean)
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        x = self.embed_drop(x)

        # Lift to Poincaré ball
        x = expmap0(x * 0.1, c.item())   # scale down before lifting for stability

        # Causal mask
        causal_mask = self.get_causal_mask(T, device)   # (1,1,T,T)

        # Padding mask (broadcast over heads)
        if attention_mask is not None:
            # Convert 0/1 mask to additive -inf mask
            pad_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            mask = causal_mask + pad_mask
        else:
            mask = causal_mask

        # Transformer layers (all on manifold)
        # Update c in each layer dynamically
        for layer in self.layers:
            layer.attn.c = c.item()
            layer.ffn.c  = c.item()
            layer.norm1.c = c.item()
            layer.norm2.c = c.item()
            x = layer(x, mask=mask)

        # Project back to Euclidean for LM head
        x = logmap0(x, c.item())
        x = self.final_norm(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        output = {"logits": logits, "curvature": c}

        # Language modelling loss (next-token prediction)
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    def param_count(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total: {total/1e6:.1f}M  Trainable: {trainable/1e6:.1f}M"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    print("Building model...")
    model = HyperbolicReasoningStudent(
        vocab_size=32000, d_model=256, n_heads=4, n_layers=4, ffn_dim=512
    )
    print(f"Parameters: {model.param_count()}")
    print(f"Initial curvature c = {model.c:.4f}")

    B, T = 2, 32
    ids  = torch.randint(0, 32000, (B, T))
    mask = torch.ones(B, T)
    labels = ids.clone()
    labels[:, :5] = -100   # ignore first 5 tokens

    out = model(ids, attention_mask=mask, labels=labels)
    print(f"Logits shape : {out['logits'].shape}")
    print(f"Loss         : {out['loss'].item():.4f}")
    print(f"Curvature c  : {out['curvature'].item():.4f}")
    print("Model forward pass OK.")
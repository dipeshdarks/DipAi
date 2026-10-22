"""DIPAI scratch-mode architecture: a small decoder-only Transformer (GPT-style).

Memory-conscious choices:
  * F.scaled_dot_product_attention (PyTorch SDPA) - fused, no explicit
    attention matrix materialization in the common path.
  * Optional weight tying between token embedding and LM head.
  * All dims come from config.ModelConfig; nothing is hard-coded.

This is an ordinary next-token language model. Trained from scratch on a
small local dataset it will produce limited, dataset-shaped text - it is NOT
a chat assistant and must not be represented as one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dipai.config import ModelConfig


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch SDPA."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # Single fused qkv projection (GPT-2 style) - fewer kernels.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask registration (square, block_size).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
            .view(1, 1, cfg.block_size, cfg.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # SDPA handles the causal mask internally when is_causal=True and no
        # explicit mask is passed; dropout applies only in training.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


# ---------------------------------------------------------------------------
# Block / MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Position-wise feed-forward with GELU and 4x expansion."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN Transformer block: x + attn(ln(x)), x + mlp(ln(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DipaiGPT(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # type: ignore[assignment]

        self.apply(self._init_weights)
        # GPT-2 residual-scaled init for output projections.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("lm_head.weight"):
                if p.dim() >= 2:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        self.num_params = sum(p.numel() for p in self.parameters())
        # Non-embedding count reported by nanoGPT-style printouts.
        self.num_params_nonembed = self.num_params - (
            self.tok_emb.weight.numel()
            + (0 if cfg.tie_weights else self.lm_head.weight.numel())
        )

        # Optional activation recomputation (memory/speed trade-off, off by
        # default). Enabled by the trainer via set_gradient_checkpointing().
        self._grad_ckpt = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Recompute transformer-block activations during backward (VRAM saver).

        Uses torch.utils.checkpoint with use_reentrant=False: RNG state for
        dropout inside the checkpointed blocks is recorded and restored, so
        results match the non-checkpointed path.
        """
        self._grad_ckpt = bool(enabled)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """idx: (B, T) int64 token ids. targets: optional (B, T) next-token ids."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.cfg.block_size}"
            )

        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        use_ckpt = self._grad_ckpt and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            # Inference: only compute logits for the last position when the
            # caller only needs the next token (cheap path used by generate()).
            logits = self.lm_head(x)
            loss = None
        return ModelOutput(logits=logits, loss=loss)

    # ------------------------------------------------------------------

    def crop_block_size(self, block_size: int) -> None:
        """Shrink the context window in place (e.g. to fit less VRAM)."""
        assert block_size <= self.cfg.block_size
        self.cfg.block_size = block_size
        self.pos_emb.weight = nn.Parameter(self.pos_emb.weight[:block_size])
        for block in self.blocks:
            block.attn.causal_mask = torch.tril(
                torch.ones(block_size, block_size, dtype=torch.bool)
            ).view(1, 1, block_size, block_size)

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """AdamW with decay applied only to 2D+ params (no decay on biases/LN)."""
        decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # fused AdamW is available on CUDA; fall back gracefully.
        doc = getattr(torch.optim.AdamW.__init__, "__doc__", "") or ""
        fused_ok = device_type == "cuda" and "fused" in doc
        try:
            optimizer = torch.optim.AdamW(
                groups, lr=learning_rate, betas=betas,
                fused=True if fused_ok else None,
            )
        except (TypeError, ValueError):
            # Older/newer torch builds may not accept fused=None.
            optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=betas)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Returns the extended sequence (B, T+n).

        temperature : softmax temperature (<=0 -> greedy argmax)
        top_k       : keep only the k most likely tokens (0/None = disabled)
        top_p       : nucleus sampling threshold in (0, 1] (None = disabled)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to block_size (positional embedding limit).
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            out = self.forward(idx_cond)
            logits = out.logits[:, -1, :]

            if temperature is None or temperature <= 0.0:
                # Greedy.
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    logits = _top_p_filter(logits, top_p)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).all()):
                break
        return idx

    # Serialization helpers (checkpoint manager owns file IO).

    def state_dict_full(self) -> dict:
        return {
            "model": self.state_dict(),
            "model_config": self.cfg.__dict__,
            "num_params": self.num_params,
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> "DipaiGPT":
        return cls(ModelConfig(**d))


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative prob >= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Tokens that push cumulative over top_p (keep at least one).
    sorted_remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_remove] = -float("inf")
    # Scatter back to original order.
    return logits.scatter(1, sorted_indices, sorted_logits)


def build_model(cfg: ModelConfig) -> DipaiGPT:
    """Construct a DipaiGPT from a ModelConfig (validates dims)."""
    cfg.validate()
    return DipaiGPT(cfg)

"""Text generation for DIPAI with full sampling controls.

Controls:
  * temperature  - softmax temperature; <=0 means greedy argmax
  * top_k        - keep the k most likely tokens (0 disables)
  * top_p        - nucleus sampling threshold (1.0 disables)
  * max_new_tokens- hard token cap
  * repetition_penalty - penalize tokens already in context (1.0 disables)

The loop lives here (not in model.py) because sampling policy belongs to the
application layer; model.generate() remains the minimal primitive used by
tests. Generation streams through a callback so chat/webui can print tokens
as they arrive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F

from dipai.model import DipaiGPT


class GenerationError(Exception):
    """Raised for invalid generation settings or runtime failures."""


@dataclass
class GenerationSettings:
    """Sampling configuration for one generation call."""

    temperature: float = 0.8
    top_k: int = 40                # 0 = disabled
    top_p: float = 0.95            # 1.0 = disabled
    max_new_tokens: int = 200
    repetition_penalty: float = 1.0  # 1.0 = disabled; typical 1.05..1.2

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise GenerationError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise GenerationError(f"temperature must be >= 0, got {self.temperature}")
        if not 0.0 <= self.top_p <= 1.0:
            raise GenerationError(f"top_p must be in [0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise GenerationError(f"top_k must be >= 0, got {self.top_k}")
        if self.repetition_penalty <= 0:
            raise GenerationError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )

    def describe(self) -> str:
        return (f"temp={self.temperature} top_k={self.top_k} top_p={self.top_p} "
                f"max_new={self.max_new_tokens} rep_pen={self.repetition_penalty}")


def apply_repetition_penalty(
    logits: torch.Tensor,
    context_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalize logits of tokens already present in context_ids (last row).

    Standard CTRL-style penalty: logits > 0 are divided by penalty,
    logits < 0 are multiplied by penalty.
    """
    if penalty == 1.0:
        return logits
    seen = torch.unique(context_ids)
    selected = logits[0, seen]
    penalized = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits = logits.clone()
    logits[0, seen] = penalized
    return logits


def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Filter a (1, V) logits row by top-k then nucleus (top-p). In place-safe."""
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits[logits < kth] = -float("inf")
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Remove tokens that cross the threshold (keep the boundary token).
        remove = cumulative - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = logits.scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(
    model: DipaiGPT,
    prompt_ids: Sequence[int] | torch.Tensor,
    settings: GenerationSettings,
    device: str = "cpu",
    eos_token_id: Optional[int] = None,
    stream_fn: Optional[Callable[[int], None]] = None,
    stop_event: Optional[Callable[[], bool]] = None,
) -> List[int]:
    """Generate tokens for a prompt. Returns ONLY the newly generated ids.

    stream_fn : called with each new token id as it is produced (may be None)
    stop_event: callable returning True to abort generation early
    """
    settings.validate()
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = torch.tensor(list(prompt_ids), dtype=torch.long)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.numel() == 0:
        raise GenerationError("prompt is empty - provide at least one token")

    model.eval()
    idx = prompt_ids.to(device)
    block = model.cfg.block_size
    generated: List[int] = []
    t0 = time.perf_counter()

    for _ in range(settings.max_new_tokens):
        if stop_event is not None and stop_event():
            break
        idx_cond = idx if idx.size(1) <= block else idx[:, -block:]
        logits = model(idx_cond).logits[:, -1, :]

        # Repetition penalty against the full visible context.
        if settings.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(
                logits, idx_cond.squeeze(0), settings.repetition_penalty
            )

        if settings.temperature == 0.0:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / settings.temperature
            logits = top_k_top_p_filter(logits, settings.top_k, settings.top_p)
            probs = F.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(probs, num_samples=1)

        token = int(nxt.item())
        generated.append(token)
        idx = torch.cat([idx, nxt], dim=1)
        if stream_fn is not None:
            stream_fn(token)
        if eos_token_id is not None and token == eos_token_id:
            break

    _ = t0  # elapsed available to callers via their own timing if needed
    return generated


def generate_text(
    model: DipaiGPT,
    tokenizer,
    prompt: str,
    settings: GenerationSettings,
    device: str = "cpu",
    stream_fn: Optional[Callable[[str], None]] = None,
    stop_event: Optional[Callable[[], bool]] = None,
) -> str:
    """Encode a text prompt, generate, decode only the continuation.

    stream_fn receives the cumulative continuation text after each token -
    cumulative decoding keeps detokenization correct for SentencePiece.
    """
    from dipai.tokenizer import BOS_ID, EOS_ID

    if not prompt or not prompt.strip():
        raise GenerationError("prompt is empty - provide some text to continue")

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    seen: List[int] = []

    def _stream(tid: int) -> None:
        seen.append(tid)
        if stream_fn is not None:
            stream_fn(tokenizer.decode(seen, skip_special=True))

    new_ids = generate(
        model, prompt_ids, settings, device=device,
        eos_token_id=EOS_ID, stream_fn=_stream if stream_fn else None,
        stop_event=stop_event,
    )
    return tokenizer.decode([t for t in new_ids if t != EOS_ID], skip_special=True)

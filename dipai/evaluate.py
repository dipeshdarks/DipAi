"""Model evaluation: perplexity + generation smoke samples.

Perplexity is the canonical intrinsic LM metric: exp(mean cross-entropy loss)
over a held-out token set. It is comparable across checkpoints evaluated on the
SAME tokenized validation data and tokenizer, and is meaningless to compare
across different tokenizers/datasets (we warn loudly if the caller tries).

Two backends are supported:
  * scratch - a DipaiGPT checkpoint (dipai.model.DipaiGPT), loss from forward().
  * hf      - a Hugging Face causal LM (AutoModelForCausalLM), loss from the
              model's own forward(). Optional: apply the tokenizer's chat
              template before scoring so SFT models are scored as they are used.

Reports are written as JSON (+ optional Markdown) with: per-example losses,
mean loss, perplexity, token counts, generation samples, and (when a base model
is supplied) a delta table comparing candidate vs base.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_jsonl_texts(path: str | Path, text_key: str = "text") -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                t = obj.get(text_key)
                if isinstance(t, str) and t.strip():
                    texts.append(t)
            elif isinstance(obj, str) and obj.strip():
                texts.append(obj)
    return texts


@dataclass
class EvalResult:
    """Perplexity + loss over one model on one dataset."""
    label: str
    num_examples: int
    num_tokens: int
    mean_loss: float
    perplexity: float
    per_example_loss: List[float] = field(default_factory=list)
    backend: str = "scratch"
    dataset: str = ""
    tokenizer_vocab: int = 0
    generation_samples: List[Dict[str, str]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "backend": self.backend,
            "dataset": self.dataset,
            "num_examples": self.num_examples,
            "num_tokens": self.num_tokens,
            "mean_loss": self.mean_loss,
            "perplexity": self.perplexity,
            "tokenizer_vocab": self.tokenizer_vocab,
            "per_example_loss": self.per_example_loss,
            "generation_samples": self.generation_samples,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation: {self.label}",
            "",
            f"- backend: `{self.backend}`",
            f"- dataset: `{self.dataset}`",
            f"- examples: {self.num_examples}",
            f"- tokens: {self.num_tokens}",
            f"- mean loss: {self.mean_loss:.4f}",
            f"- **perplexity: {self.perplexity:.2f}**",
        ]
        if self.notes:
            lines.append(f"- notes: {self.notes}")
        if self.generation_samples:
            lines += ["", "## Generation samples", ""]
            for s in self.generation_samples:
                lines += [
                    f"**Prompt:** {s.get('prompt', '')}",
                    "",
                    f"**Completion:** {s.get('completion', '')}",
                    "",
                    "---",
                    "",
                ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perplexity backends
# ---------------------------------------------------------------------------

@torch.no_grad()
def _perplexity_scratch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
) -> EvalResult:
    """Perplexity for a DipaiGPT (scratch) model."""
    model.eval()
    losses: List[float] = []
    token_count = 0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        encs = [
            tokenizer.encode(t, add_bos=True, add_eos=True)[:max_length]
            for t in chunk
        ]
        encs = [e for e in encs if len(e) >= 2]
        if not encs:
            continue
        max_len = max(len(e) for e in encs)
        pad_id = tokenizer.pad_id
        input_ids = torch.full((len(encs), max_len), pad_id, dtype=torch.long)
        targets = torch.full((len(encs), max_len), -100, dtype=torch.long)
        for j, e in enumerate(encs):
            input_ids[j, : len(e)] = torch.tensor(e, dtype=torch.long)
            # next-token targets: shift by one; last position ignored
            targets[j, : len(e) - 1] = torch.tensor(e[1:], dtype=torch.long)
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        out = model(input_ids, targets=targets)
        loss = float(out.loss)
        valid = int((targets != -100).sum())
        losses.append(loss * valid)
        token_count += valid
    total_nll = sum(losses)
    mean_loss = total_nll / max(1, token_count)
    return EvalResult(
        label="scratch",
        backend="scratch",
        num_examples=len(texts),
        num_tokens=token_count,
        mean_loss=mean_loss,
        perplexity=math.exp(min(20.0, mean_loss)),
        tokenizer_vocab=getattr(tokenizer, "vocab_size", 0),
    )


@torch.no_grad()
def _perplexity_hf(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_length: int,
    batch_size: int,
    apply_chat_template: bool = False,
) -> EvalResult:
    """Perplexity for a Hugging Face causal LM.

    When apply_chat_template is True, each text is wrapped as a single
    assistant turn so SFT models are scored on the distribution they were
    trained to produce.
    """
    model.eval()
    losses: List[float] = []
    token_count = 0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        if apply_chat_template and getattr(tokenizer, "chat_template", None):
            chunk = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for t in chunk
            ]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = float(out.loss)
        valid = int((labels != -100).sum())
        losses.append(loss * valid)
        token_count += valid
    total_nll = sum(losses)
    mean_loss = total_nll / max(1, token_count)
    return EvalResult(
        label="hf",
        backend="hf",
        num_examples=len(texts),
        num_tokens=token_count,
        mean_loss=mean_loss,
        perplexity=math.exp(min(20.0, mean_loss)),
        tokenizer_vocab=len(tokenizer) if hasattr(tokenizer, "__len__") else 0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    dataset_path: str | Path,
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 1,
    limit: int = 0,
    generate_samples: int = 0,
    generate_prompt: str = "The meaning of life is",
    generate_max_tokens: int = 40,
) -> EvalResult:
    """Evaluate a DIPAI scratch checkpoint on a JSONL text dataset."""
    from dipai.hardware import get_device
    from dipai.inference import DIPAIEngine

    eng = DIPAIEngine.from_checkpoint(checkpoint_dir, device=device)
    dev = get_device(device)
    texts = _read_jsonl_texts(dataset_path)
    if limit and limit > 0:
        texts = texts[:limit]
    if not texts:
        raise ValueError(f"no usable texts in dataset: {dataset_path}")

    result = _perplexity_scratch(
        eng.model, eng.tokenizer, texts, dev, max_length, batch_size
    )
    result.label = Path(checkpoint_dir).name
    result.dataset = str(dataset_path)

    if generate_samples > 0:
        from dipai.inference import GenerationSettings

        settings = GenerationSettings(max_new_tokens=generate_max_tokens)
        for _ in range(generate_samples):
            completion = eng.generate(generate_prompt, settings=settings)
            result.generation_samples.append(
                {"prompt": generate_prompt, "completion": completion.strip()}
            )
    return result


def evaluate_hf_model(
    model_name_or_path: str,
    dataset_path: str | Path,
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 1,
    limit: int = 0,
    apply_chat_template: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
) -> EvalResult:
    """Evaluate a Hugging Face causal LM on a JSONL text dataset."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dipai.hardware import get_device

    dev = get_device(device)
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    quant = None
    if load_in_4bit and dev == "cuda":
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        quantization_config=quant,
        torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        device_map="auto" if dev == "cuda" else None,
    )
    texts = _read_jsonl_texts(dataset_path)
    if limit and limit > 0:
        texts = texts[:limit]
    if not texts:
        raise ValueError(f"no usable texts in dataset: {dataset_path}")

    result = _perplexity_hf(
        model, tok, texts, dev, max_length, batch_size, apply_chat_template
    )
    result.label = Path(model_name_or_path).name
    result.dataset = str(dataset_path)
    return result


def compare(
    candidate: EvalResult,
    base: EvalResult,
) -> Dict[str, Any]:
    """Delta table: candidate vs base. Negative loss/perp delta = improvement.

    Perplexity deltas are only meaningful when both sides were scored on the
    same dataset and the same tokenizer; mismatches are collected into a
    "warnings" list here and logged by the CLI.
    """
    warnings: List[str] = []
    if candidate.dataset != base.dataset:
        warnings.append(
            f"dataset mismatch: {candidate.dataset!r} vs {base.dataset!r} - "
            "perplexity deltas are not comparable across different eval sets"
        )
    if (
        candidate.tokenizer_vocab
        and base.tokenizer_vocab
        and candidate.tokenizer_vocab != base.tokenizer_vocab
    ):
        warnings.append(
            f"tokenizer vocab mismatch: {candidate.tokenizer_vocab} vs "
            f"{base.tokenizer_vocab} - perplexity is not comparable across "
            "different tokenizers"
        )
    return {
        "candidate": candidate.label,
        "base": base.label,
        "dataset_match": candidate.dataset == base.dataset,
        "candidate_loss": candidate.mean_loss,
        "base_loss": base.mean_loss,
        "loss_delta": candidate.mean_loss - base.mean_loss,
        "candidate_perplexity": candidate.perplexity,
        "base_perplexity": base.perplexity,
        "perplexity_delta": candidate.perplexity - base.perplexity,
        "perplexity_ratio": (
            candidate.perplexity / base.perplexity if base.perplexity else float("inf")
        ),
        "improved": candidate.mean_loss < base.mean_loss,
        "warnings": warnings,
    }


def write_report(
    result: EvalResult,
    out_json: str | Path,
    out_md: Optional[str | Path] = None,
    comparison: Optional[Dict[str, Any]] = None,
) -> Path:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    if comparison is not None:
        payload["comparison"] = comparison
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if out_md is not None:
        md = result.to_markdown()
        if comparison is not None:
            md += "\n\n## Comparison vs base\n\n"
            md += "| metric | base | candidate | delta |\n|---|---|---|---|\n"
            md += f"| loss | {comparison['base_loss']:.4f} | {comparison['candidate_loss']:.4f} | {comparison['loss_delta']:+.4f} |\n"
            md += f"| perplexity | {comparison['base_perplexity']:.2f} | {comparison['candidate_perplexity']:.2f} | {comparison['perplexity_delta']:+.2f} |\n"
            if comparison.get("warnings"):
                md += "\n**Caveats:**\n"
                for w in comparison["warnings"]:
                    md += f"- {w}\n"
        Path(out_md).write_text(md, encoding="utf-8")
    return out_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dipai.evaluate",
        description="Evaluate a DIPAI or HF model: perplexity + generation samples.",
    )
    p.add_argument("--checkpoint", default=None,
                   help="DIPAI scratch checkpoint dir (backend=scratch)")
    p.add_argument("--hf-model", default=None,
                   help="HF model name/path (backend=hf)")
    p.add_argument("--dataset", required=True,
                   help="JSONL eval dataset with a 'text' field")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="0 = all examples")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="wrap each text in the HF chat template before scoring")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--generate-samples", type=int, default=0)
    p.add_argument("--generate-prompt", default="The meaning of life is")
    p.add_argument("--generate-max-tokens", type=int, default=40)
    p.add_argument("--base-checkpoint", default=None,
                   help="optional base DIPAI ckpt to compare against")
    p.add_argument("--base-hf-model", default=None,
                   help="optional base HF model to compare against")
    p.add_argument("--out-json", default="reports/eval.json")
    p.add_argument("--out-md", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from dipai.utils import setup_logging

    log = setup_logging()
    args = build_arg_parser().parse_args(argv)

    if not args.checkpoint and not args.hf_model:
        log.error("provide --checkpoint or --hf-model")
        return 2

    try:
        if args.checkpoint:
            result = evaluate_checkpoint(
                args.checkpoint, args.dataset, args.device,
                args.max_length, args.batch_size, args.limit,
                args.generate_samples, args.generate_prompt,
                args.generate_max_tokens,
            )
        else:
            result = evaluate_hf_model(
                args.hf_model, args.dataset, args.device,
                args.max_length, args.batch_size, args.limit,
                args.apply_chat_template, args.load_in_4bit,
            )
    except Exception as exc:  # noqa: BLE001 - surface as clean CLI error
        log.error("evaluation failed: %s", exc)
        return 1

    comparison = None
    if args.base_checkpoint or args.base_hf_model:
        try:
            if args.base_checkpoint:
                base = evaluate_checkpoint(
                    args.base_checkpoint, args.dataset, args.device,
                    args.max_length, args.batch_size, args.limit,
                )
            else:
                base = evaluate_hf_model(
                    args.base_hf_model, args.dataset, args.device,
                    args.max_length, args.batch_size, args.limit,
                    args.apply_chat_template, args.load_in_4bit,
                )
            comparison = compare(result, base)
            for w in comparison.get("warnings", []):
                log.warning("%s", w)
        except Exception as exc:  # noqa: BLE001
            log.warning("base evaluation failed (%s); skipping comparison", exc)

    write_report(result, args.out_json, args.out_md, comparison)
    log.info(
        "eval %s: loss=%.4f ppl=%.2f tokens=%d -> %s",
        result.label, result.mean_loss, result.perplexity, result.num_tokens,
        args.out_json,
    )
    if comparison:
        verdict = "IMPROVED" if comparison["improved"] else "regressed"
        log.info("vs base %s: loss %+.4f, ppl %+.2f (%s)",
                 comparison["base"], comparison["loss_delta"],
                 comparison["perplexity_delta"], verdict)
    print(json.dumps(result.to_dict() | ({"comparison": comparison} if comparison else {}),
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

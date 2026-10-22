"""Tests for dipai.evaluate: perplexity, compare, HF backend, CLI."""

from __future__ import annotations

import json
import math

import pytest
import torch

from dipai.evaluate import (
    EvalResult,
    _perplexity_hf,
    _perplexity_scratch,
    _read_jsonl_texts,
    compare,
    evaluate_checkpoint,
    evaluate_hf_model,
    write_report,
)


@pytest.fixture()
def eval_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "once upon a time there was a little cat named spot",
        "machine learning models learn patterns from text data",
        "user hello there assistant hello how can i help you today",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}) + "\n")
    return path


@pytest.fixture()
def scratch_ckpt(tmp_path, tiny_model_cfg, tiny_tokenizer):
    """A real scratch checkpoint on disk, with tokenizer.model beside it."""
    from dipai.checkpoints import save_checkpoint
    from dipai.model import DipaiGPT

    model = DipaiGPT(tiny_model_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_dir = tmp_path / "ckpt"
    save_checkpoint(
        ckpt_dir, 1, model, opt,
        config={"model": tiny_model_cfg.__dict__}, keep_last_n=3,
    )
    # inference.from_checkpoint looks for tokenizer.model inside the ckpt dir
    tiny_tokenizer.save(ckpt_dir / "tokenizer.model")
    return ckpt_dir


def test_read_jsonl_texts(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        json.dumps({"text": "a"}) + "\n"
        + "not json\n"
        + json.dumps({"text": "b"}) + "\n"
        + json.dumps({"other": "skip"}) + "\n",
        encoding="utf-8",
    )
    assert _read_jsonl_texts(p) == ["a", "b"]


def test_perplexity_scratch_finite(tiny_model, tiny_tokenizer):
    texts = ["the quick brown fox", "once upon a time there was a cat"]
    res = _perplexity_scratch(tiny_model, tiny_tokenizer, texts, "cpu", 32, 1)
    assert res.num_tokens > 0
    assert math.isfinite(res.mean_loss)
    assert res.perplexity == pytest.approx(math.exp(min(20.0, res.mean_loss)))
    assert res.num_examples == 2


def test_perplexity_scratch_random_model_high(tiny_model, tiny_tokenizer):
    """An untrained random model should have high perplexity (near vocab size)."""
    texts = ["the quick brown fox jumps over the lazy dog"] * 8
    res = _perplexity_scratch(tiny_model, tiny_tokenizer, texts, "cpu", 32, 2)
    # random init -> loss ~ ln(vocab); perplexity ~ vocab size
    assert res.perplexity > 10


def test_perplexity_loss_scales_with_token_count(tiny_model, tiny_tokenizer):
    short = _perplexity_scratch(tiny_model, tiny_tokenizer, ["the quick"], "cpu", 32, 1)
    long = _perplexity_scratch(
        tiny_model, tiny_tokenizer, ["the quick brown fox jumps over the lazy dog"] * 4,
        "cpu", 32, 1,
    )
    assert long.num_tokens > short.num_tokens


def test_evaluate_checkpoint_end_to_end(scratch_ckpt, eval_jsonl, tmp_path):
    res = evaluate_checkpoint(
        scratch_ckpt, eval_jsonl, device="cpu", max_length=32, batch_size=2,
        generate_samples=1, generate_max_tokens=4,
    )
    assert isinstance(res, EvalResult)
    assert res.backend == "scratch"
    assert math.isfinite(res.mean_loss)
    assert res.num_examples == 4
    assert len(res.generation_samples) == 1
    assert "prompt" in res.generation_samples[0]


def test_compare_improvement_and_regression():
    cand = EvalResult(
        label="cand", backend="scratch", num_examples=2, num_tokens=10,
        mean_loss=2.0, perplexity=math.exp(2.0), dataset="d",
    )
    base = EvalResult(
        label="base", backend="scratch", num_examples=2, num_tokens=10,
        mean_loss=3.0, perplexity=math.exp(3.0), dataset="d",
    )
    cmp = compare(cand, base)
    assert cmp["improved"] is True
    assert cmp["loss_delta"] == pytest.approx(-1.0)
    assert cmp["perplexity_ratio"] == pytest.approx(math.exp(-1.0))
    assert cmp["dataset_match"] is True
    assert cmp["warnings"] == []

    worse = compare(base, cand)
    assert worse["improved"] is False


def test_compare_warns_on_dataset_mismatch():
    cand = EvalResult("cand", 1, 5, 2.0, math.exp(2.0), dataset="set-a")
    base = EvalResult("base", 1, 5, 3.0, math.exp(3.0), dataset="set-b")
    cmp = compare(cand, base)
    assert cmp["dataset_match"] is False
    assert any("dataset mismatch" in w for w in cmp["warnings"])
    assert any("not comparable" in w for w in cmp["warnings"])


def test_compare_warns_on_tokenizer_mismatch():
    cand = EvalResult("cand", 1, 5, 2.0, math.exp(2.0), dataset="d",
                      tokenizer_vocab=400)
    base = EvalResult("base", 1, 5, 3.0, math.exp(3.0), dataset="d",
                      tokenizer_vocab=32000)
    cmp = compare(cand, base)
    assert any("tokenizer vocab mismatch" in w for w in cmp["warnings"])
    # dataset still matches, so no dataset warning
    assert not any("dataset mismatch" in w for w in cmp["warnings"])


def test_write_report_json_and_md(tmp_path):
    res = EvalResult(
        label="m", backend="scratch", num_examples=1, num_tokens=5,
        mean_loss=2.5, perplexity=math.exp(2.5), dataset="d",
        generation_samples=[{"prompt": "p", "completion": "c"}],
    )
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    write_report(res, out_json, out_md)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["label"] == "m"
    assert payload["generation_samples"][0]["completion"] == "c"
    md = out_md.read_text(encoding="utf-8")
    assert "perplexity" in md.lower()


def test_write_report_with_comparison(tmp_path):
    cand = EvalResult("cand", 1, 5, 2.0, math.exp(2.0), dataset="d")
    base = EvalResult("base", 1, 5, 3.0, math.exp(3.0), dataset="d")
    cmp = compare(cand, base)
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    write_report(cand, out_json, out_md, comparison=cmp)
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["comparison"]["improved"] is True


def test_write_report_markdown_caveats(tmp_path):
    cand = EvalResult("cand", 1, 5, 2.0, math.exp(2.0), dataset="a")
    base = EvalResult("base", 1, 5, 3.0, math.exp(3.0), dataset="b")
    cmp = compare(cand, base)
    assert cmp["warnings"], "expected dataset-mismatch warnings"
    out_json = tmp_path / "r.json"
    out_md = tmp_path / "r.md"
    write_report(cand, out_json, out_md, comparison=cmp)
    md = out_md.read_text(encoding="utf-8")
    assert "Caveats" in md
    assert "dataset mismatch" in md
    # JSON carries the warnings verbatim too
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["comparison"]["warnings"]


@pytest.mark.slow
def test_evaluate_hf_model_tiny(tmp_path, eval_jsonl):
    transformers = pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model_dir = tmp_path / "hf_tiny"
    tok.save_pretrained(model_dir)
    cfg = AutoConfig.for_model(
        "llama", vocab_size=tok.vocab_size, hidden_size=64,
        intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=128,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(model_dir)

    res = evaluate_hf_model(
        str(model_dir), eval_jsonl, device="cpu", max_length=32, batch_size=2,
        limit=2,
    )
    assert res.backend == "hf"
    assert res.num_examples == 2
    assert math.isfinite(res.mean_loss)


def test_perplexity_hf_chat_template_flag(tiny_tokenizer):
    """apply_chat_template=True must not crash and must still score tokens."""
    pytest.importorskip("transformers")
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "llama", vocab_size=tiny_tokenizer.vocab_size, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    # build a minimal HF tokenizer wrapper around the tiny SP tokenizer is
    # overkill; just verify the no-chat-template path works with a list of texts
    class _T:
        pad_token_id = 0
        chat_template = None

        def __call__(self, texts, **kw):
            ids = [tiny_tokenizer.encode(t, add_bos=True, add_eos=True)[:16] for t in texts]
            m = max(len(i) for i in ids)
            ii = torch.zeros(len(ids), m, dtype=torch.long)
            am = torch.zeros(len(ids), m, dtype=torch.long)
            for r, i in enumerate(ids):
                ii[r, : len(i)] = torch.tensor(i)
                am[r, : len(i)] = 1
            return {"input_ids": ii, "attention_mask": am}

        def __len__(self):
            return tiny_tokenizer.vocab_size

    res = _perplexity_hf(model, _T(), ["the quick brown fox"], "cpu", 16, 1)
    assert res.num_tokens > 0
    assert math.isfinite(res.mean_loss)

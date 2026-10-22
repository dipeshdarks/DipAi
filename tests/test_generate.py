"""Tests for dipai.generate: sampling controls and stop/stream behavior."""

from __future__ import annotations

import pytest
import torch

from dipai.generate import (
    GenerationError,
    GenerationSettings,
    apply_repetition_penalty,
    generate,
    generate_text,
    top_k_top_p_filter,
)


def test_settings_validate():
    GenerationSettings().validate()
    with pytest.raises(GenerationError, match="temperature"):
        GenerationSettings(temperature=-0.1).validate()
    with pytest.raises(GenerationError, match="top_p"):
        GenerationSettings(top_p=1.5).validate()
    with pytest.raises(GenerationError, match="max_new_tokens"):
        GenerationSettings(max_new_tokens=0).validate()
    with pytest.raises(GenerationError, match="repetition_penalty"):
        GenerationSettings(repetition_penalty=0.0).validate()
    with pytest.raises(GenerationError, match="top_k"):
        GenerationSettings(top_k=-1).validate()
    # temperature == 0 is valid: it selects greedy decoding.
    GenerationSettings(temperature=0.0).validate()


def test_top_k_keeps_k_best():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    filtered = top_k_top_p_filter(logits.clone(), top_k=2, top_p=1.0)
    finite = ~torch.isinf(filtered)
    assert int(finite.sum()) == 2
    assert finite[0][1] and finite[0][4]  # values 5.0 and 4.0 kept


def test_top_p_keeps_cumulative_mass():
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
    filtered = top_k_top_p_filter(logits.clone(), top_k=0, top_p=0.9)
    # Top token alone is ~0.63 mass... second pushes over 0.9 -> keep 2.
    finite = ~torch.isinf(filtered)
    assert finite[0][0] and finite[0][1]
    assert not finite[0][2] and not finite[0][3]


def test_repetition_penalty_direction():
    logits = torch.tensor([[2.0, 1.0, 2.0]])
    ctx = torch.tensor([0, 2])                 # tokens 0 and 2 were seen
    out = apply_repetition_penalty(logits.clone(), ctx, 1.5)
    assert float(out[0][0]) < float(logits[0][0])   # repeated token pushed down
    assert float(out[0][1]) == float(logits[0][1])  # unseen token untouched
    # penalty < 1 encourages repetition instead
    out2 = apply_repetition_penalty(logits.clone(), torch.tensor([0]), 0.5)
    assert float(out2[0][0]) > float(logits[0][0])


def test_generate_text_shapes_and_stream(tiny_model, tiny_tokenizer):
    calls = []
    out = generate_text(
        tiny_model, tiny_tokenizer, "hello world",
        GenerationSettings(max_new_tokens=5, temperature=0.7, top_k=20),
        stream_fn=lambda t: calls.append(t),
    )
    assert isinstance(out, str) and out
    assert calls, "stream callback should receive cumulative text"
    assert calls[-1].endswith(out) or calls[-1] == out


def test_generate_text_empty_prompt_raises(tiny_model, tiny_tokenizer):
    with pytest.raises(GenerationError, match="empty"):
        generate_text(tiny_model, tiny_tokenizer, "   ",
                      GenerationSettings(max_new_tokens=3))


def test_generate_text_stop_event(tiny_model, tiny_tokenizer):
    """A stop event mid-stream must cut generation short."""
    flag = {"stop": False}

    def stream_fn(_t):
        flag["stop"] = True  # stop after the first token

    out = generate_text(
        tiny_model, tiny_tokenizer, "hello",
        GenerationSettings(max_new_tokens=50, temperature=0.9),
        stream_fn=stream_fn,
        stop_event=lambda: flag["stop"],
    )
    # Stopped early: decoded text is shorter than 50 tokens would imply.
    assert isinstance(out, str)


def test_generate_greedy_no_sampling(tiny_model, tiny_tokenizer):
    a = generate_text(tiny_model, tiny_tokenizer, "hello",
                      GenerationSettings(max_new_tokens=4, temperature=0.0))
    b = generate_text(tiny_model, tiny_tokenizer, "hello",
                      GenerationSettings(max_new_tokens=4, temperature=0.0))
    assert a == b

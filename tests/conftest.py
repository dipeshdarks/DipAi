"""Shared pytest fixtures for DIPAI tests (CPU-only, no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    """SentencePiece tokenizer trained once per session (~2s).

    Session-scoped because SP training is the slow part of the suite.
    """
    from dipai.tokenizer import DipaiTokenizer

    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "once upon a time there was a little cat named spot",
        "user hello there assistant hello how can i help you today",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "machine learning models learn patterns from text data",
    ] * 6
    return DipaiTokenizer.train(
        corpus,
        model_prefix=str(tmp_path_factory.mktemp("tok") / "tok"),
        vocab_size=400,
        sample_chars=200_000,
    )


@pytest.fixture()
def tiny_model_cfg(tiny_tokenizer):
    """Small fast config whose vocab MATCHES the tokenizer - the same
    invariant train.py enforces (model vocab always follows the tokenizer)."""
    from dipai.config import ModelConfig

    return ModelConfig(
        vocab_size=tiny_tokenizer.vocab_size, block_size=32, n_layer=2,
        n_head=2, n_embd=32, dropout=0.0, bias=False, tie_weights=True,
    )


@pytest.fixture()
def tiny_model(tiny_model_cfg):
    from dipai.model import DipaiGPT

    model = DipaiGPT(tiny_model_cfg)
    model.eval()
    return model

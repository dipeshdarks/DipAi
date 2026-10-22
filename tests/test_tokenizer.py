"""Tests for dipai.tokenizer: training, roundtrip, specials, persistence."""

from __future__ import annotations

import pytest

from dipai.tokenizer import DipaiTokenizer, TokenizerError


def test_special_ids_are_standard(tiny_tokenizer):
    t = tiny_tokenizer
    assert t.pad_id == 0
    assert t.unk_id == 1
    assert t.bos_id == 2
    assert t.eos_id == 3
    assert t.vocab_size >= 64


def test_encode_decode_roundtrip(tiny_tokenizer):
    text = "the quick brown fox jumps over the lazy dog"
    ids = tiny_tokenizer.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == tiny_tokenizer.bos_id
    assert ids[-1] == tiny_tokenizer.eos_id
    # SentencePiece + byte_fallback is lossless for arbitrary text.
    bare = tiny_tokenizer.encode(text, add_bos=False, add_eos=False)
    assert tiny_tokenizer.decode(bare) == text


def test_unk_never_for_ascii(tiny_tokenizer):
    ids = tiny_tokenizer.encode("abcdefghijklmnopqrstuvwxyz ABC123")
    assert tiny_tokenizer.unk_id not in ids


def test_save_and_load_roundtrip(tiny_tokenizer, tmp_path):
    path = tmp_path / "tok.model"
    tiny_tokenizer.save(str(path))
    assert path.exists()
    loaded = DipaiTokenizer.load(str(path))
    assert loaded.vocab_size == tiny_tokenizer.vocab_size
    text = "once upon a time there was a cat"
    assert loaded.encode(text) == tiny_tokenizer.encode(text)


def test_load_missing_raises(tmp_path):
    with pytest.raises(TokenizerError, match="not found|failed"):
        DipaiTokenizer.load(tmp_path / "missing.model")


def test_encode_empty_string(tiny_tokenizer):
    assert tiny_tokenizer.encode("", add_bos=False, add_eos=False) == []

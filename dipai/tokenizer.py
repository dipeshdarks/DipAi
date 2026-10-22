"""Tokenizer for DIPAI scratch mode.

Wraps SentencePiece BPE behind a single interface used by the rest of the
pipeline. Finetune mode uses the base model's native tokenizer instead (see
dipai/finetune.py), so only scratch training depends on this module.

Design notes:
  * Special IDs are fixed and reserved: <pad>=0, <unk>=1, <bos>=2, <eos>=3.
    Real vocab ids start at 4 so the first 4 ids are always safe to embed.
  * Training streams a bounded sample of the corpus so machines with little
    RAM never materialize the whole dataset for tokenizer training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

log = logging.getLogger("dipai.tokenizer")

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# SentencePiece control tokens; user text can never occupy these ids.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# Cap on characters handed to SentencePiece at once (keeps RSS bounded).
DEFAULT_TRAIN_SAMPLE_CHARS = 20_000_000  # ~20 MB of text


class TokenizerError(Exception):
    """Raised for tokenizer training/loading failures."""


class DipaiTokenizer:
    """SentencePiece BPE tokenizer with DIPAI's fixed special-token contract."""

    def __init__(self, sp_model):
        self.sp = sp_model
        self.vocab_size: int = self.sp.vocab_size()

        # Verify the special-token contract on load; fail loudly on mismatch
        # rather than silently producing garbage ids later.
        for name, expected in (("<pad>", PAD_ID), ("<unk>", UNK_ID),
                               ("<bos>", BOS_ID), ("<eos>", EOS_ID)):
            got = self.sp.piece_to_id(name)
            if got != expected:
                raise TokenizerError(
                    f"special token {name} has id {got}, expected {expected}. "
                    "The tokenizer file does not match DIPAI's contract."
                )

    # Special-token ids (fixed by DIPAI's contract; see module constants).
    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        if not isinstance(text, str):
            raise TokenizerError(f"encode expects str, got {type(text).__name__}")
        ids: List[int] = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(ids)

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    # ------------------------------------------------------------------
    # Training / persistence
    # ------------------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        model_prefix: str | Path,
        vocab_size: int = 8192,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        sample_chars: int = DEFAULT_TRAIN_SAMPLE_CHARS,
        max_lines: Optional[int] = None,
    ) -> "DipaiTokenizer":
        """Train a SentencePiece model from an iterable of text chunks.

        Reads at most `sample_chars` characters (and at most `max_lines` items)
        so training on machines with limited RAM stays bounded.
        """
        import sentencepiece as spm

        model_prefix = Path(model_prefix)
        # SentencePiece appends ".model" to the prefix itself. If the caller
        # passed a path already ending in ".model" (the natural convention),
        # strip it so we get exactly the requested filename.
        if model_prefix.suffix == ".model":
            model_prefix = model_prefix.with_suffix("")
        model_prefix.parent.mkdir(parents=True, exist_ok=True)

        # SentencePiece needs a file. We stream into a bounded temp corpus
        # file instead of holding text in memory.
        corpus_path = model_prefix.parent / (model_prefix.name + ".corpus.tmp")
        written = 0
        lines = 0
        try:
            with open(corpus_path, "w", encoding="utf-8") as out:
                for text in texts:
                    if not text:
                        continue
                    text = text.strip()
                    if not text:
                        continue
                    out.write(text.replace("\n", " ") + "\n")
                    written += len(text) + 1
                    lines += 1
                    if max_lines is not None and lines >= max_lines:
                        break
                    if written >= sample_chars:
                        log.info(
                            "tokenizer training sample limit reached "
                            "(%d chars, %d lines)", written, lines
                        )
                        break
            if lines == 0:
                raise TokenizerError(
                    "no usable text provided for tokenizer training. "
                    "Run scripts/prepare_data.py on a non-empty dataset first."
                )

            # +8 reserved ids (4 special x user ids can't reach these; SP adds
            # its own control pieces). Headroom avoids UNK-ification.
            sp_vocab = max(vocab_size, 256) + 8
            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                vocab_size=sp_vocab,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=PAD_ID,
                unk_id=UNK_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                pad_piece=PAD_TOKEN,
                unk_piece=UNK_TOKEN,
                bos_piece=BOS_TOKEN,
                eos_piece=EOS_TOKEN,
                user_defined_symbols=[],
                byte_fallback=True,
                train_extremely_large_corpus=False,
                hard_vocab_limit=False,
                shuffle_input_sentence=True,
                minloglevel=2,           # silence per-piece INFO spam (errors still show)
            )
        finally:
            if corpus_path.exists():
                try:
                    corpus_path.unlink()
                except OSError:
                    pass

        tok = cls.load(model_prefix)
        log.info("tokenizer trained: vocab_size=%d path=%s",
                 tok.vocab_size, model_prefix)
        return tok

    @classmethod
    def load(cls, path: str | Path) -> "DipaiTokenizer":
        import sentencepiece as spm

        path = Path(path)
        model_file = path
        if path.is_dir():
            model_file = path / "tokenizer.model"
        elif not path.exists() and Path(str(path) + ".model").exists():
            # Caller passed a SentencePiece prefix (e.g. "tokenizer" or
            # "tokenizer.model" already stripped by train()).
            model_file = Path(str(path) + ".model")
        if not model_file.exists():
            raise TokenizerError(
                f"tokenizer file not found: {model_file}. "
                "Run scripts/prepare_data.py --train-tokenizer first."
            )
        try:
            sp = spm.SentencePieceProcessor(model_file=str(model_file))
        except Exception as exc:
            raise TokenizerError(f"failed to load tokenizer {model_file}: {exc}") from exc
        # sentencepiece 0.2.x dropped is_loaded(); a construction that didn't
        # raise reports a sane vocab_size. Probe it to be certain.
        try:
            if sp.get_piece_size() <= 0:
                raise TokenizerError(f"tokenizer at {model_file} loaded with empty vocabulary")
        except Exception as exc:
            if isinstance(exc, TokenizerError):
                raise
            raise TokenizerError(f"tokenizer at {model_file} failed to load: {exc}") from exc
        tok = cls(sp)
        # Sidecar metadata (informational).
        meta_file = model_file.with_suffix(".model.meta.json")
        if meta_file.exists():
            try:
                tok.meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                tok.meta = {}
        else:
            tok.meta = {}
        return tok

    def save_meta(self, path: str | Path, extra: Optional[dict] = None) -> None:
        """Write a small JSON sidecar next to the .model file."""
        meta = {"vocab_size": self.vocab_size, "special_ids": {
            "pad": PAD_ID, "unk": UNK_ID, "bos": BOS_ID, "eos": EOS_ID,
        }}
        if extra:
            meta.update(extra)
        p = Path(path)
        p = p.with_suffix(".model.meta.json") if p.suffix == ".model" else p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def save(self, path: str | Path) -> Path:
        """Serialize the model to `path` (a `.model` file). Returns the path."""
        p = Path(path)
        if p.suffix != ".model":
            p = Path(str(p) + ".model")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.sp.serialized_model_proto())
        return p


def load_tokenizer(path: str | Path) -> DipaiTokenizer:
    """Convenience loader used by training/generation code."""
    return DipaiTokenizer.load(path)

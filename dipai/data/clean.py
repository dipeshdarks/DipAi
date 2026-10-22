"""Deterministic, resumable cleaning and quality pipeline for web text.

Every stage is independent, pure(ish), and testable. Stages run in a fixed
order and each records how many documents it dropped, so an audit trail falls
out naturally. Nothing here uses a neural model: language ID is stopword
based, quality scoring is heuristic, near-dup is banded MinHash. That keeps
the pipeline dependency-free, deterministic, and honest about what it can and
cannot do.

Stages (in order):
    1  normalize        - NFKC unicode + whitespace collapse + control-char strip
    2  drop_empty       - remove blank / whitespace-only docs
    3  encoding_filter  - drop docs with too many replacement/control chars
    4  length_filter    - drop < min_chars or > max_chars
    5  boilerplate      - drop navigation-heavy / spam / listicle garbage
    6  language         - keep only requested languages (stopword LID)
    7  dedup_exact      - sha1 exact-duplicate removal (streaming, bounded)
    8  dedup_near       - optional banded MinHash near-duplicate removal
    9  quality          - score + threshold; low scorers dropped (configurable)
    10 pii              - detect + optionally redact emails/phones/cards/ips
    11 report           - per-stage counters + token estimates -> JSON

CLI:
    python -m dipai.data.clean --input data/raw/web/*.jsonl \
        --output data/cleaned/web.jsonl --report data/cleaned/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

log = logging.getLogger("dipai.data.clean")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CleanConfig:
    input_paths: List[str] = field(default_factory=list)
    output_path: str = "data/cleaned/web.jsonl"
    report_path: str = "data/cleaned/clean_report.json"
    text_key: str = "text"
    # stage toggles
    normalize: bool = True
    drop_empty: bool = True
    encoding_filter: bool = True
    length_filter: bool = True
    boilerplate: bool = True
    language: bool = True
    dedup_exact: bool = True
    dedup_near: bool = False        # off by default: costs RAM
    quality: bool = True
    pii: bool = True
    pii_redact: bool = True         # False = only count, do not modify
    # parameters
    min_chars: int = 100
    max_chars: int = 100_000
    languages: List[str] = field(default_factory=lambda: ["en"])
    max_replacement_ratio: float = 0.02
    quality_threshold: float = 0.35   # 0..1; docs below are dropped
    dedup_max_entries: int = 500_000
    near_dup_num_perm: int = 64
    near_dup_bands: int = 16          # rows per band = num_perm/bands
    near_dup_threshold: float = 0.8   # Jaccard >= this => duplicate
    max_docs: int = 0                 # 0 = no cap
    seed: int = 1337

    def validate(self) -> None:
        if self.min_chars < 0 or self.max_chars <= 0 or self.min_chars > self.max_chars:
            raise ValueError("bad min_chars/max_chars")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be in [0,1]")
        if self.near_dup_num_perm % self.near_dup_bands != 0:
            raise ValueError("near_dup_num_perm must be divisible by near_dup_bands")
        if not 0.0 < self.near_dup_threshold <= 1.0:
            raise ValueError("near_dup_threshold must be in (0,1]")


def load_clean_config(path: Optional[str] = None,
                      overrides: Optional[Dict[str, Any]] = None) -> CleanConfig:
    data: Dict[str, Any] = {}
    if path:
        from dipai.config import load_yaml
        data = load_yaml(path)
    known = set(CleanConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
    cfg = CleanConfig(**data)
    for k, v in (overrides or {}).items():
        if v is not None and k in known:
            setattr(cfg, k, v)
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Stage counters / audit trail
# ---------------------------------------------------------------------------

@dataclass
class StageStats:
    name: str
    kept: int = 0
    dropped: int = 0
    modified: int = 0          # docs changed in place (e.g. redaction)
    detail: Dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.detail[key] = self.detail.get(key, 0) + n


@dataclass
class CleanReport:
    schema_version: int = 1
    inputs: List[str] = field(default_factory=list)
    output: str = ""
    total_in: int = 0
    total_out: int = 0
    tokens_estimated: int = 0
    stages: List[StageStats] = field(default_factory=list)
    quality_hist: Dict[str, int] = field(default_factory=dict)  # 10 buckets
    pii_detected: Dict[str, int] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": self.inputs,
            "output": self.output,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "kept_ratio": (self.total_out / self.total_in) if self.total_in else 0.0,
            "tokens_estimated": self.tokens_estimated,
            "stages": [asdict(s) for s in self.stages],
            "quality_hist": self.quality_hist,
            "pii_detected": self.pii_detected,
            "config": self.config,
        }

    def summary(self) -> str:
        lines = [f"clean: {self.total_in:,} in -> {self.total_out:,} out "
                 f"({100.0 * self.total_out / max(self.total_in, 1):.1f}% kept), "
                 f"~{self.tokens_estimated:,} tokens"]
        for s in self.stages:
            lines.append(f"  {s.name:16} kept={s.kept:<7} dropped={s.dropped:<7} "
                         f"modified={s.modified}" +
                         (f"  {s.detail}" if s.detail else ""))
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    def to_markdown(self) -> str:
        """Human-readable report (used by evaluate + docs)."""
        out = ["# Data cleaning report", ""]
        out.append(f"- documents in: **{self.total_in:,}**")
        out.append(f"- documents out: **{self.total_out:,}** "
                   f"({100.0 * self.total_out / max(self.total_in, 1):.1f}% kept)")
        out.append(f"- estimated tokens: **{self.tokens_estimated:,}**")
        out.append("")
        out.append("| stage | kept | dropped | modified | detail |")
        out.append("|---|---:|---:|---:|---|")
        for s in self.stages:
            d = ", ".join(f"{k}={v}" for k, v in s.detail.items()) or "-"
            out.append(f"| {s.name} | {s.kept:,} | {s.dropped:,} | {s.modified:,} | {d} |")
        if self.pii_detected:
            out.append("")
            out.append("## PII detected")
            for k, v in sorted(self.pii_detected.items()):
                out.append(f"- {k}: {v}")
        if self.quality_hist:
            out.append("")
            out.append("## Quality score histogram (deciles)")
            for bucket in sorted(self.quality_hist):
                out.append(f"- {bucket}: {self.quality_hist[bucket]}")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Stage 1-4: text hygiene
# ---------------------------------------------------------------------------

_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """NFKC-normalise, strip control chars (keeping \\n\\t), collapse whitespace."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _CTRL_RE.sub("", t)
    t = _WS_RE.sub(" ", t)
    t = _MULTI_NL_RE.sub("\n\n", t)
    return t.strip()


def encoding_ok(text: str, max_replacement_ratio: float) -> Tuple[bool, str]:
    """Reject docs with too many U+FFFD / C1 control chars (mojibake)."""
    n = len(text)
    if n == 0:
        return False, "empty"
    bad = text.count("�") + sum(
        1 for ch in text if unicodedata.category(ch) == "Cc" and ch not in "\n\t")
    if bad / n > max_replacement_ratio:
        return False, "replacement_chars"
    return True, ""


# ---------------------------------------------------------------------------
# Stage 6: stopword language ID
# ---------------------------------------------------------------------------

# Small deterministic stopword tables. Not a real LID model: good enough to
# filter obvious out-of-language docs, honest about being a heuristic.
_STOPWORDS: Dict[str, set] = {
    "en": {"the", "and", "for", "that", "with", "this", "from", "are", "was",
           "were", "have", "has", "had", "not", "you", "but", "they", "which",
           "their", "there", "would", "can", "will", "one", "all", "been",
           "when", "who", "them", "than", "its", "into", "some", "could",
           "other", "more", "very", "what", "about", "out", "these", "his",
           "her", "she", "each", "how", "their", "our", "us", "such"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "sich",
           "auch", "auf", "für", "mit", "den", "dem", "zu", "im", "dem",
           "sie", "wir", "werden", "bei", "einer", "um", "am", "sind", "noch",
           "wie", "einem", "über", "einen", "so", "zum", "war", "haben"},
    "fr": {"le", "la", "les", "des", "une", "est", "que", "dans", "pour",
           "qui", "sur", "pas", "au", "avec", "ce", "cette", "sont", "par",
           "plus", "vous", "nous", "être", "tout", "comme", "mais", "où",
           "leur", "fait", "elle", "entre", "après", "aussi"},
    "es": {"el", "la", "los", "las", "que", "de", "en", "un", "una", "por",
           "con", "no", "se", "para", "es", "al", "lo", "como", "más", "pero",
           "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre"},
    "it": {"il", "che", "di", "e", "la", "in", "un", "per", "è", "su", "con",
           "non", "sono", "mi", "anche", "io", "questo", "come", "quello",
           "più", "se", "tutto", "ma", "dei", "della", "nelle", "nel"},
    "pt": {"o", "a", "os", "as", "de", "do", "da", "em", "um", "uma", "que",
           "não", "com", "por", "mais", "para", "se", "foi", "ao", "ele",
           "das", "tem", "à", "ser", "quando", "muito", "há", "nos", "já"},
    "nl": {"de", "het", "een", "van", "en", "in", "is", "dat", "op", "te",
           "voor", "met", "zijn", "niet", "aan", "er", "maar", "om", "ook",
           "wij", "wat", "werd", "door", "over", "uit", "kan", "wordt"},
    "ru": {"и", "в", "на", "не", "что", "он", "с", "как", "а", "по", "это",
           "к", "но", "they", "из", "какой", "мы", "за", "бы", "его", "все",
           "так", "его", "для", "ты", "же", "все", "этот", "только", "ее"},
}


def detect_language(text: str) -> str:
    """Stopword-ratio language guess. Returns ISO code or 'unknown'.

    Emails and URLs are stripped first so domains like example.com do not
    pollute the token set.
    """
    probe = _URL_RE.sub(" ", text.lower())
    probe = _PII_PATTERNS["email"].sub(" ", probe)
    words = re.findall(r"[^\W\d_]{2,}", probe)
    if len(words) < 5:
        return "unknown"
    counts: Dict[str, int] = {}
    for lang, stops in _STOPWORDS.items():
        counts[lang] = sum(1 for w in words if w in stops)
    best, best_n = "unknown", 0
    for lang, n in counts.items():
        if n > best_n:
            best, best_n = lang, n
    # require a minimal signal: >=2% of words are stopwords of that language
    return best if best_n >= max(3, int(0.02 * len(words))) else "unknown"


# ---------------------------------------------------------------------------
# Stage 7: exact dedup (streaming, bounded memory)
# ---------------------------------------------------------------------------

class ExactDeduper:
    """sha1 exact-duplicate filter with a bounded seen-set.

    When the set is full it degrades to a rotating window (same behaviour as
    the existing dataset.py dedup) so memory stays bounded.
    """

    def __init__(self, max_entries: int = 500_000):
        self.max_entries = max_entries
        self._seen: set = set()
        self._overflow = False

    def seen(self, text: str) -> bool:
        h = hashlib.sha1(text.encode("utf-8", "replace")).digest()
        if h in self._seen:
            return True
        if len(self._seen) >= self.max_entries:
            self._overflow = True
            # rotating window: drop ~1/8 of the set
            to_drop = [x for i, x in enumerate(self._seen) if i % 8 == 0]
            for x in to_drop:
                self._seen.discard(x)
        self._seen.add(h)
        return False

    @property
    def overflowed(self) -> bool:
        return self._overflow


# ---------------------------------------------------------------------------
# Stage 8: near-duplicate MinHash (optional)
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set:
    """Word-level k-shingle set (lowercased, whitespace-normalised)."""
    toks = text.lower().split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


class MinHasher:
    """Banded MinHash LSH for near-duplicate detection.

    Pure-python + sha1 as the universal hash. Deterministic given `seed`.
    """

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 1337):
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        # fixed coefficient table derived from the seed (deterministic)
        import random
        rng = random.Random(seed)
        self._a = [rng.randint(1, 2**61 - 1) for _ in range(num_perm)]
        self._b = [rng.randint(0, 2**61 - 1) for _ in range(num_perm)]
        self._p = 2**61 - 1

    def signature(self, shingle_set: set) -> List[int]:
        if not shingle_set:
            return [0] * self.num_perm
        sig = []
        for i in range(self.num_perm):
            hmin = None
            for s in shingle_set:
                v = (self._a[i] * int(hashlib.sha1(s.encode()).hexdigest()[:12], 16)
                     + self._b[i]) % self._p
                if hmin is None or v < hmin:
                    hmin = v
            sig.append(hmin)
        return sig

    def buckets(self, sig: List[int]) -> List[Tuple[int, int]]:
        out = []
        for b in range(self.bands):
            chunk = tuple(sig[b * self.rows:(b + 1) * self.rows])
            out.append((b, hash(chunk) & 0xFFFFFFFF))
        return out


class NearDeduper:
    """LSH-backed near-duplicate filter over a bounded candidate table."""

    def __init__(self, num_perm: int = 64, bands: int = 16,
                 threshold: float = 0.8, max_entries: int = 200_000):
        self.mh = MinHasher(num_perm, bands)
        self.threshold = threshold
        self.max_entries = max_entries
        self._buckets: Dict[Tuple[int, int], List[int]] = {}
        self._sigs: List[List[int]] = []

    def is_duplicate(self, text: str) -> bool:
        sh = _shingles(text)
        sig = self.mh.signature(sh)
        cand: set = set()
        for key in self.mh.buckets(sig):
            for idx in self._buckets.get(key, []):
                cand.add(idx)
        for idx in cand:
            if _jaccard(sig, self._sigs[idx]) >= self.threshold:
                return True
        if len(self._sigs) >= self.max_entries:
            # bounded: reset tables rather than grow unbounded
            self._sigs.clear()
            self._buckets.clear()
        my_id = len(self._sigs)
        self._sigs.append(sig)
        for key in self.mh.buckets(sig):
            self._buckets.setdefault(key, []).append(my_id)
        return False


def _jaccard(sig_a: List[int], sig_b: List[int]) -> float:
    same = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return same / len(sig_a)


# ---------------------------------------------------------------------------
# Stage 9: quality scoring
# ---------------------------------------------------------------------------

_LINE_END_PUNCT = re.compile(r"[.!?]$")
_ALPHA_RE = re.compile(r"[^\W\d_]")
_URL_RE = re.compile(r"https?://|www\.")
_NUM_RE = re.compile(r"\d")


def quality_score(text: str) -> float:
    """Heuristic 0..1 quality score. NOT a classifier: a cheap sanity gate.

    Rewards: reasonable length, alphabetic ratio, sentence structure,
    moderate punctuation. Penalises: link/number density, repeated lines,
    ALL-CAPS walls, single-line walls, extreme char diversity collapse.
    """
    if not text:
        return 0.0
    n = len(text)
    score = 0.0

    # length term: peak between 500 and 8000 chars
    if n < 200:
        score += 0.10 * (n / 200)
    elif n <= 8000:
        score += 0.25
    else:
        score += max(0.05, 0.25 - (n - 8000) / 100_000)

    # alphabetic ratio
    alpha = len(_ALPHA_RE.findall(text))
    ar = alpha / n
    score += 0.25 if 0.6 <= ar <= 0.98 else (0.1 if 0.4 <= ar < 0.6 else 0.0)

    # sentence structure
    lines = [l for l in text.split("\n") if l.strip()]
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
    if sents:
        avg_sent = sum(len(s) for s in sents) / len(sents)
        if 30 <= avg_sent <= 300:
            score += 0.20
        elif 15 <= avg_sent < 30 or 300 < avg_sent <= 500:
            score += 0.10
    # lines ending in punctuation (prose vs list/nav)
    if lines:
        ended = sum(1 for l in lines if _LINE_END_PUNCT.search(l.strip()))
        score += 0.10 * min(1.0, ended / max(1, len(lines) * 0.5))

    # penalties
    urls = len(_URL_RE.findall(text))
    score -= min(0.20, 0.03 * urls)
    nums = len(_NUM_RE.findall(text)) / n
    if nums > 0.25:
        score -= 0.10
    # repeated lines
    if lines:
        uniq = len(set(l.strip() for l in lines))
        if uniq / len(lines) < 0.5:
            score -= 0.15
    # ALL CAPS wall
    letters = sum(1 for c in text if c.isalpha())
    if letters > 100:
        caps = sum(1 for c in text if c.isupper()) / letters
        if caps > 0.6:
            score -= 0.15

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Stage 10: PII detection / redaction
# ---------------------------------------------------------------------------

_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}(?!\d)"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)"),
    "ssn_like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
}
_REDACTIONS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "ipv4": "[IP]",
    "credit_card": "[CARD]",
    "ssn_like": "[SSN]",
}


def find_pii(text: str) -> Dict[str, int]:
    """Count PII-like hits per category (heuristic regexes)."""
    out: Dict[str, int] = {}
    for name, pat in _PII_PATTERNS.items():
        hits = pat.findall(text)
        if name == "phone":
            # too many false positives: only count when a +country or () hint
            hits = [h for h in hits if (h.strip().startswith("+") or
                                        "(" in h or h.count("-") >= 2 or
                                        h.count(" ") >= 2)]
        if hits:
            out[name] = len(hits)
    return out


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Replace detected PII with category tags. Returns (text, counts)."""
    counts: Dict[str, int] = {}
    for name, pat in _PII_PATTERNS.items():
        if name == "phone":
            def _phone_sub(m):
                s = m.group(0)
                if not (s.strip().startswith("+") or "(" in s or
                        s.count("-") >= 2 or s.count(" ") >= 2):
                    return s
                counts["phone"] = counts.get("phone", 0) + 1
                return _REDACTIONS["phone"]
            text = pat.sub(_phone_sub, text)
            continue
        n = len(pat.findall(text))
        if n:
            counts[name] = n
            text = pat.sub(_REDACTIONS[name], text)
    return text, counts


# ---------------------------------------------------------------------------
# Contamination check (train vs eval n-gram overlap)
# ---------------------------------------------------------------------------

def contamination_report(train_texts: Iterable[str], eval_texts: Iterable[str],
                         ngram: int = 8, sample: int = 20_000) -> Dict[str, Any]:
    """Measure n-gram overlap between train and eval (leakage detector).

    Builds a bounded set of eval n-grams (hashed), then samples train docs and
    reports what fraction share at least one n-gram. High overlap = leakage.
    """
    eval_ngrams: set = set()
    eval_docs = 0
    for t in eval_texts:
        eval_docs += 1
        toks = t.lower().split()
        for i in range(max(0, len(toks) - ngram + 1)):
            g = " ".join(toks[i:i + ngram])
            eval_ngrams.add(hashlib.sha1(g.encode()).digest()[:8])
        if eval_docs >= sample:
            break
    hit_docs = 0
    train_docs = 0
    total_ngrams = 0
    hit_ngrams = 0
    for t in train_texts:
        train_docs += 1
        toks = t.lower().split()
        doc_hit = False
        for i in range(max(0, len(toks) - ngram + 1)):
            g = " ".join(toks[i:i + ngram])
            total_ngrams += 1
            if hashlib.sha1(g.encode()).digest()[:8] in eval_ngrams:
                hit_ngrams += 1
                doc_hit = True
        if doc_hit:
            hit_docs += 1
        if train_docs >= sample:
            break
    return {
        "ngram": ngram,
        "eval_docs": eval_docs,
        "train_docs_sampled": train_docs,
        "train_docs_with_overlap": hit_docs,
        "doc_overlap_ratio": hit_docs / train_docs if train_docs else 0.0,
        "ngram_overlap_ratio": hit_ngrams / total_ngrams if total_ngrams else 0.0,
        "verdict": ("LEAKAGE" if train_docs and hit_docs / train_docs > 0.05
                    else "clean"),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def iter_jsonl(paths: List[str], text_key: str = "text") -> Iterator[Dict[str, Any]]:
    for pat in paths:
        p = Path(pat)
        files = [p] if p.is_file() else sorted(Path().glob(pat))
        if p.is_file() and not files:
            files = [p]
        for f in files:
            if not f.exists():
                log.warning("input missing: %s", f)
                continue
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        yield {"_bad_json": True, "_file": str(f), "_line": lineno}
                        continue
                    yield rec


def clean_records(records: Iterable[Dict[str, Any]], cfg: CleanConfig
                  ) -> Tuple[List[Dict[str, Any]], CleanReport]:
    """Run the full stage pipeline over an iterable of records."""
    cfg.validate()
    report = CleanReport(inputs=list(cfg.input_paths), output=cfg.output_path,
                         config=json.loads(json.dumps(asdict(cfg), default=str)))
    st = {name: StageStats(name=name) for name in (
        "normalize", "drop_empty", "encoding_filter", "length_filter",
        "boilerplate", "language", "dedup_exact", "dedup_near",
        "quality", "pii")}
    dedup = ExactDeduper(cfg.dedup_max_entries) if cfg.dedup_exact else None
    near = (NearDeduper(cfg.near_dup_num_perm, cfg.near_dup_bands,
                        cfg.near_dup_threshold)
            if cfg.dedup_near else None)

    kept: List[Dict[str, Any]] = []
    for rec in records:
        if cfg.max_docs and len(kept) >= cfg.max_docs:
            break
        report.total_in += 1
        if rec.get("_bad_json"):
            st["normalize"].dropped += 1
            st["normalize"].bump("bad_json")
            continue
        text = rec.get(cfg.text_key)
        if not isinstance(text, str):
            st["drop_empty"].dropped += 1
            st["drop_empty"].bump("missing_text")
            continue
        source = rec.get("source", "")

        # 1 normalize
        if cfg.normalize:
            new = normalize_text(text)
            if new != text:
                st["normalize"].modified += 1
            text = new
        st["normalize"].kept += 1

        # 2 drop_empty
        if cfg.drop_empty and not text.strip():
            st["drop_empty"].dropped += 1
            continue
        st["drop_empty"].kept += 1

        # 3 encoding
        if cfg.encoding_filter:
            ok, why = encoding_ok(text, cfg.max_replacement_ratio)
            if not ok:
                st["encoding_filter"].dropped += 1
                st["encoding_filter"].bump(why or "bad")
                continue
        st["encoding_filter"].kept += 1

        # 4 length
        if cfg.length_filter:
            if len(text) < cfg.min_chars:
                st["length_filter"].dropped += 1
                st["length_filter"].bump("short")
                continue
            if len(text) > cfg.max_chars:
                st["length_filter"].dropped += 1
                st["length_filter"].bump("long")
                continue
        st["length_filter"].kept += 1

        # 5 boilerplate / spam heuristics
        if cfg.boilerplate:
            bad, why = _looks_like_boilerplate(text)
            if bad:
                st["boilerplate"].dropped += 1
                st["boilerplate"].bump(why)
                continue
        st["boilerplate"].kept += 1

        # 6 language
        if cfg.language and cfg.languages:
            lang = detect_language(text)
            if lang not in {l.lower() for l in cfg.languages} and lang != "unknown":
                st["language"].dropped += 1
                st["language"].bump(lang)
                continue
            # 'unknown' is kept: heuristic LID is not authoritative
        st["language"].kept += 1

        # 7 exact dedup
        if dedup is not None:
            if dedup.seen(text):
                st["dedup_exact"].dropped += 1
                st["dedup_exact"].bump("duplicate")
                continue
        st["dedup_exact"].kept += 1

        # 8 near dedup
        if near is not None:
            if near.is_duplicate(text):
                st["dedup_near"].dropped += 1
                st["dedup_near"].bump("near_duplicate")
                continue
        st["dedup_near"].kept += 1

        # 9 quality
        q = quality_score(text)
        bucket = f"{int(q * 10) * 10}-{int(q * 10) * 10 + 9}"
        report.quality_hist[bucket] = report.quality_hist.get(bucket, 0) + 1
        if cfg.quality and q < cfg.quality_threshold:
            st["quality"].dropped += 1
            st["quality"].bump(f"below_{cfg.quality_threshold:.2f}")
            continue
        st["quality"].kept += 1

        # 10 pii
        if cfg.pii:
            hits = find_pii(text)
            for k, v in hits.items():
                report.pii_detected[k] = report.pii_detected.get(k, 0) + v
            if hits:
                if cfg.pii_redact:
                    text, counts = redact_pii(text)
                    st["pii"].modified += 1
                else:
                    counts = hits
            else:
                counts = {}
        else:
            counts = {}

        out_rec = {k: v for k, v in rec.items() if k != cfg.text_key}
        out_rec[cfg.text_key] = text
        out_rec["quality_score"] = round(q, 4)
        if cfg.language and cfg.languages:
            out_rec["lang"] = detect_language(text)
        if counts:
            out_rec["pii_redacted"] = counts
        kept.append(out_rec)

    report.total_out = len(kept)
    report.stages = list(st.values())
    # ~4 chars/token rough estimate (documented as an estimate)
    report.tokens_estimated = sum(len(r.get(cfg.text_key, "")) for r in kept) // 4
    if dedup is not None and dedup.overflowed:
        report.stages[6].detail["dedup_overflow"] = 1
    return kept, report


def _looks_like_boilerplate(text: str) -> Tuple[bool, str]:
    """Cheap navigation/spam/listicle detectors. Returns (drop?, reason)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return True, "no_lines"
    n = len(text)
    # menu wall: many short lines, few sentence enders
    if len(lines) > 20:
        short = sum(1 for l in lines if len(l) < 25)
        if short / len(lines) > 0.8:
            return True, "menu_wall"
    # cookie / consent boilerplate
    lowered = text.lower()
    for marker in ("accept all cookies", "we use cookies", "privacy policy",
                   "terms of service", "sign up for our newsletter",
                   "subscribe to our newsletter", "enable javascript",
                   "cookie policy"):
        if marker in lowered and n < 1500:
            return True, "cookie_or_consent"
    # bullet/emoji listicle garbage
    if n > 0:
        bullets = sum(1 for l in lines if l[:1] in "•·▪●-–—*")
        if len(lines) > 10 and bullets / len(lines) > 0.7:
            return True, "bullet_list"
    # single mega-line wall
    if len(lines) == 1 and n > 1500:
        return True, "one_line_wall"
    return False, ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dipai.data.clean",
        description="Deterministic web-text cleaning: dedup, LID, quality, PII.",
    )
    p.add_argument("--input", nargs="+", required=True,
                   help="input JSONL files or glob patterns")
    p.add_argument("--output", default=None)
    p.add_argument("--report", default=None)
    p.add_argument("--text-key", default=None)
    p.add_argument("--min-chars", type=int, default=None)
    p.add_argument("--max-chars", type=int, default=None)
    p.add_argument("--languages", nargs="*", default=None)
    p.add_argument("--quality-threshold", type=float, default=None)
    p.add_argument("--dedup-near", action="store_true", default=None)
    p.add_argument("--no-pii-redact", dest="pii_redact", action="store_false")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--markdown", action="store_true",
                   help="also write <report>.md")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    from dipai.utils import setup_logging
    setup_logging()
    args = build_arg_parser().parse_args(argv)
    overrides = {
        "input_paths": args.input,
        "output_path": args.output,
        "report_path": args.report,
        "text_key": args.text_key,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "languages": args.languages,
        "quality_threshold": args.quality_threshold,
        "dedup_near": args.dedup_near,
        "pii_redact": args.pii_redact if args.pii_redact is not None else None,
        "max_docs": args.max_docs,
    }
    try:
        cfg = load_clean_config(overrides=overrides)
    except ValueError as exc:
        log.error("bad config: %s", exc)
        return 2

    records = iter_jsonl(cfg.input_paths, cfg.text_key)
    kept, report = clean_records(records, cfg)

    out = Path(cfg.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report.save(cfg.report_path)
    if args.markdown:
        Path(cfg.report_path).with_suffix(".md").write_text(
            report.to_markdown(), encoding="utf-8")
    print(report.summary())
    print(f"wrote {len(kept):,} docs -> {out}")
    print(f"report -> {cfg.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

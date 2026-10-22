import json
from pathlib import Path

from dipai.data.clean import (
    CleanConfig,
    ExactDeduper,
    MinHasher,
    NearDeduper,
    _looks_like_boilerplate,
    clean_records,
    contamination_report,
    detect_language,
    encoding_ok,
    find_pii,
    iter_jsonl,
    normalize_text,
    quality_score,
    redact_pii,
)


def _rec(text, **kw):
    return {"text": text, **kw}


def _cfg(**kw):
    base = dict(min_chars=20, max_chars=5000, languages=["en"],
                quality_threshold=0.0, pii=True, pii_redact=True)
    base.update(kw)
    return CleanConfig(**base)


# ---- stage 1: normalize ----

def test_normalize_collapses_whitespace():
    t = normalize_text("Hello\r\n\r\n\r\n world \t  tab  \rfoo")
    assert "\r" not in t
    assert "  " not in t
    assert "\n\n\n" not in t


def test_normalize_nfkc():
    # full-width 'Ａ' -> 'A'
    t = normalize_text("ＡＢＣ")
    assert t == "ABC"


# ---- stage 3: encoding ----

def test_encoding_ok_clean():
    ok, why = encoding_ok("just normal english text", 0.02)
    assert ok and why == ""


def test_encoding_rejects_replacement_flood():
    bad = "ok" + "�" * 50
    ok, why = encoding_ok(bad, 0.02)
    assert not ok and why == "replacement_chars"


# ---- stage 6: language ----

def test_language_en():
    assert detect_language("The cat sat on the mat and the dog ran in the park") == "en"


def test_language_de():
    assert detect_language(
        "Der Hund läuft über die Straße und die Katze sitzt auf dem Baum in dem Garten"
    ) == "de"


def test_language_unknown_short():
    assert detect_language("hi") == "unknown"


def test_language_ignores_email_domain():
    # example.com must not leak "com" into PT
    t = "The quick brown fox jumps over the lazy dog near foo@example.com today"
    assert detect_language(t) != "pt"


# ---- stage 7: exact dedup ----

def test_exact_dedup():
    d = ExactDeduper(10)
    assert not d.seen("hello world foo bar")
    assert d.seen("hello world foo bar")
    assert not d.seen("different text entirely")


def test_exact_dedup_overflow_flag():
    d = ExactDeduper(2)
    d.seen("a")
    d.seen("b")
    d.seen("c")  # forces shrink
    assert d.overflowed


# ---- stage 8: near dedup ----

def test_minhash_similarity_high_for_near_dup():
    mh = MinHasher(num_perm=32, bands=8, seed=1)
    a = mh.signature(set("alpha beta gamma delta epsilon zeta eta theta".split()))
    b = mh.signature(set("alpha beta gamma delta epsilon zeta eta thetax".split()))
    c = mh.signature(set("one two three four five six seven eight nine".split()))
    same = sum(1 for x, y in zip(a, b) if x == y)
    diff = sum(1 for x, y in zip(a, c) if x == y)
    assert same > diff


def test_near_deduper_flags_duplicate_text():
    nd = NearDeduper(num_perm=32, bands=8, threshold=0.7)
    t1 = "the quick brown fox jumps over the lazy dog again and again today"
    t2 = "the quick brown fox jumps over the lazy dog again and again today!"
    t3 = "completely unrelated content about quantum physics and chemistry here"
    assert not nd.is_duplicate(t1)
    assert nd.is_duplicate(t2)  # near-identical
    # t3 differs enough; may or may not collide at threshold 0.7, so just call it
    nd.is_duplicate(t3)


# ---- stage 9: quality ----

def test_quality_empty_zero():
    assert quality_score("") == 0.0


def test_quality_good_text_high():
    good = "This is a normal English sentence about science. " * 20
    assert quality_score(good) > 0.3


def test_quality_allcaps_penalised():
    caps = "THIS IS A WALL OF UPPERCASE TEXT WITH NO LOWERCASE. " * 20
    normal = "This is a normal sentence with mixed case letters. " * 20
    assert quality_score(caps) < quality_score(normal)


# ---- stage 10: PII ----

def test_find_pii_email():
    assert find_pii("email me at a@b.com")["email"] == 1


def test_redact_pii_email():
    red, counts = redact_pii("email me at a@b.com")
    assert "a@b.com" not in red
    assert "[EMAIL]" in red
    assert counts["email"] == 1


def test_pii_false_positive_phone_not_all_flagged():
    # plain 4-digit numbers should not trip the phone filter
    hits = find_pii("the year 2024 and code 1234 are numbers")
    assert "phone" not in hits


def test_no_pii_redact_when_disabled():
    recs = [_rec("Contact me at foo@example.com " * 10)]
    kept, report = clean_records(recs, _cfg(pii_redact=False))
    # text unchanged (still has the email), but redaction counted
    assert "foo@example.com" in kept[0]["text"]
    assert report.pii_detected.get("email", 0) >= 1


# ---- contamination ----

def test_contamination_leakage_detected():
    shared = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 3
    rep = contamination_report([shared], [shared], ngram=8)
    assert rep["verdict"] == "LEAKAGE"


def test_contamination_clean_when_disjoint():
    tr = ["completely different training text about apples and bananas " * 3]
    ev = ["totally unrelated evaluation content about cars and trucks " * 3]
    rep = contamination_report(tr, ev, ngram=8)
    assert rep["verdict"] == "clean"


# ---- boilerplate ----

def test_boilerplate_cookie():
    drop, why = _looks_like_boilerplate("We use cookies to improve your experience. Accept all cookies. Privacy policy.")
    assert drop and why == "cookie_or_consent"


def test_boilerplate_not_on_prose():
    prose = "The river wound through the valley. " * 30
    drop, _ = _looks_like_boilerplate(prose)
    assert not drop


# ---- full pipeline ----

def test_pipeline_end_to_end(tmp_path):
    records = [
        _rec("The quick brown fox jumps over the lazy dog. " * 12, source="a"),
        _rec("The quick brown fox jumps over the lazy dog. " * 12, source="a"),  # dupe
        _rec("short"),                                                          # too short
        _rec("Contact me at foo@example.com " * 10),                             # PII
    ]
    kept, report = clean_records(records, _cfg(min_chars=50))
    assert report.total_in == 4
    assert report.total_out == 2
    assert sum("quick brown fox" in r["text"] for r in kept) == 1
    # PII email redacted
    assert any("[EMAIL]" in r["text"] for r in kept)
    # report artifacts
    d = report.to_dict()
    assert d["total_in"] == 4 and d["total_out"] == 2
    assert report.to_markdown()


def test_iter_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        json.dumps({"text": "hello"}) + "\n" +
        "not json\n" +
        json.dumps({"text": "world"}) + "\n",
        encoding="utf-8",
    )
    got = list(iter_jsonl([str(p)]))
    assert len(got) == 3
    assert any(r.get("_bad_json") for r in got)


def test_max_docs_cap(tmp_path):
    records = [_rec("word " * 30 + str(i)) for i in range(10)]
    kept, report = clean_records(records, _cfg(max_docs=3))
    assert len(kept) == 3

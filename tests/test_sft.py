"""SFT conversation pipeline: serialization, loss masking, tree-level split.

The central contract tested here: ConversationDataset trains on EXACTLY the
token stream of dipai.conversations.format_dialog (the layout chat inference
uses), with loss masked on everything the assistant should not imitate.
"""

from __future__ import annotations

import json

import pytest
import torch

from dipai.conversations import (
    ConversationError,
    assign_tree_split,
    build_conversations,
    format_dialog,
    write_splits,
)
from dipai.finetune import ConversationDataset, _collate_conversations


def _msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# dipai.conversations
# ---------------------------------------------------------------------------

def test_format_dialog_layout():
    text = format_dialog(_msgs(("user", "hello"), ("assistant", "hi there!")))
    assert text == "User: hello\n\nAssistant: hi there!"


def test_assign_tree_split_is_deterministic_and_disjoint():
    a = assign_tree_split("tree-1", 0.5, seed=7)
    b = assign_tree_split("tree-1", 0.5, seed=7)
    assert a == b
    assert assign_tree_split("tree-1", 0.5, seed=8) in ("train", "val")
    sides = {assign_tree_split(f"t{i}", 0.5, seed=7) for i in range(200)}
    assert sides == {"train", "val"}  # both sides populated at 50/50


def test_build_conversations_alternation_merge_and_tree_split():
    msgs = []
    for i, (mid, parent, role) in enumerate([
        ("m1", None, "prompter"),
        ("m2", "m1", "assistant"),
        ("m3", "m2", "prompter"),   # extra prompter message -> merged
        ("m4", "m3", "prompter"),   # merged with m3
        ("m5", "m4", "assistant"),
    ]):
        msgs.append({
            "message_id": mid, "parent_id": parent, "message_tree_id": "T1",
            "role": role, "lang": "en", "deleted": False,
            "review_count": 1, "review_result": True, "text": f"content {i}",
        })
    convs, stats = build_conversations(msgs, val_fraction=0.5, seed=7)
    assert stats.conversations == 1
    c = convs[0]
    roles = [m["role"] for m in c["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]  # merged pair
    assert "\n" in c["messages"][2]["content"]  # merged with newline
    assert stats.dropped_unreviewed == 0

    # Unreviewed messages are dropped (documented default).
    msgs2 = msgs + [{
        "message_id": "mX", "parent_id": "m5", "message_tree_id": "T1",
        "role": "assistant", "lang": "en", "deleted": False,
        "review_count": 0, "review_result": False, "text": "unreviewed",
    }]
    convs2, stats2 = build_conversations(msgs2, val_fraction=0.5, seed=7)
    assert stats2.dropped_unreviewed == 1
    assert stats2.conversations == 1


def test_tree_level_split_keeps_all_paths_together():
    # Two branching trees with distinct text (identical serializations would
    # legitimately be deduplicated); every path of a tree must share one side.
    msgs = []
    for tree in ("A", "B"):
        msgs.append({"message_id": f"{tree}0", "parent_id": None,
                     "message_tree_id": tree, "role": "prompter", "lang": "en",
                     "review_count": 1, "review_result": True, "text": f"{tree} q"})
        msgs.append({"message_id": f"{tree}1", "parent_id": f"{tree}0",
                     "message_tree_id": tree, "role": "assistant", "lang": "en",
                     "review_count": 1, "review_result": True, "text": f"{tree} a"})
        msgs.append({"message_id": f"{tree}2", "parent_id": f"{tree}1",
                     "message_tree_id": tree, "role": "prompter", "lang": "en",
                     "review_count": 1, "review_result": True, "text": f"{tree} q2"})
        for leaf in ("3", "4"):
            msgs.append({"message_id": f"{tree}{leaf}", "parent_id": f"{tree}2",
                         "message_tree_id": tree, "role": "assistant", "lang": "en",
                         "review_count": 1, "review_result": True, "text": f"{tree} a{leaf}"})
    convs, stats = build_conversations(msgs, val_fraction=0.5, seed=7)
    assert stats.conversations == 4
    by_tree = {}
    for c in convs:
        by_tree.setdefault(c["tree_id"], set()).add(c["split"])
    for tree, sides in by_tree.items():
        assert len(sides) == 1, f"tree {tree} leaked across splits: {sides}"


def test_write_splits_rejects_zero_train(tmp_path):
    recs = [{"messages": _msgs(("user", "q"), ("assistant", "a")),
             "tree_id": "t", "split": "val", "assistant_turns": 1}]
    with pytest.raises(ConversationError, match="0 training"):
        write_splits(recs, tmp_path / "tr.jsonl", tmp_path / "va.jsonl")


def test_write_splits_counts(tmp_path):
    recs = [
        {"messages": _msgs(("user", "q1"), ("assistant", "a1")),
         "tree_id": "t1", "split": "train", "assistant_turns": 1},
        {"messages": _msgs(("user", "q2"), ("assistant", "a2")),
         "tree_id": "t2", "split": "val", "assistant_turns": 1},
    ]
    n_tr, n_va = write_splits(recs, tmp_path / "tr.jsonl", tmp_path / "va.jsonl")
    assert (n_tr, n_va) == (1, 1)
    tr = [json.loads(l) for l in (tmp_path / "tr.jsonl").read_text("utf-8").splitlines()]
    assert tr[0]["messages"][0] == {"role": "user", "content": "q1"}


# ---------------------------------------------------------------------------
# ConversationDataset
# ---------------------------------------------------------------------------

@pytest.fixture()
def sft_ds(tmp_path, tiny_tokenizer):
    path = _write_jsonl(tmp_path / "sft.jsonl", [
        {"messages": _msgs(("user", "hello there"), ("assistant", "hi, how can I help you today?"))},
        {"messages": _msgs(("user", "one"), ("assistant", "two"), ("user", "three"), ("assistant", "four"))},
        {"messages": _msgs(("user", "question about machine learning"), ("assistant", "a long answer about machine learning patterns"))},
    ])
    return ConversationDataset(path, tiny_tokenizer, max_seq_length=64), tiny_tokenizer, path


def test_dataset_roundtrips_format_dialog_token_stream(sft_ds):
    ds, tok, _ = sft_ds
    for i in range(len(ds)):
        messages = ds._read(i)
        ids, _labels = ds._serialize(messages)
        fd = format_dialog(messages)
        expected = [tok.bos_id] + tok.encode(fd, add_bos=False, add_eos=False)
        if messages[-1]["role"] == "assistant":
            expected.append(tok.eos_id)
        assert ids == expected, f"record {i}: serialization != encode(format_dialog)"


def test_dataset_xy_alignment_and_masking(sft_ds):
    ds, tok, _ = sft_ds
    for i in range(len(ds)):
        item = ds[i]
        x, y = item["x"], item["y"]
        assert len(x) == len(y)
        assert x[0] == tok.bos_id
        # Supervised targets are exactly the next input token.
        for j in range(len(y) - 1):
            if y[j] != -100:
                assert y[j] == x[j + 1]
        # At least one supervised target (never an all-masked sample).
        assert any(v != -100 for v in y)
        # The conversation ends on an assistant -> trailing eos is supervised.
        if ds._read(i)[-1]["role"] == "assistant":
            assert y[-1] == tok.eos_id


def test_user_segments_masked_assistant_not(sft_ds):
    ds, tok, _ = sft_ds
    messages = ds._read(0)  # user "hello there" then assistant reply
    ids, labels = ds._serialize(messages)
    # Find the first assistant-side token ("Assistant" label piece) and check
    # everything before it (user turn + separator) is masked.
    first_asst = next(i for i, l in enumerate(labels) if l != -100)
    assert all(l == -100 for l in labels[:first_asst])
    assert all(l != -100 for l in labels[first_asst:])


def test_overlong_drops_oldest_then_scales(tmp_path, tiny_tokenizer):
    long_msgs = _msgs(
        ("user", "tell me more " * 20),
        ("assistant", "sure thing " * 20),
        ("user", "and then " * 20),
        ("assistant", "finally " * 20),
        ("user", "one last " * 20),
        ("assistant", "the end " * 20),
    )
    path = _write_jsonl(tmp_path / "long.jsonl", [{"messages": long_msgs}])
    ds = ConversationDataset(path, tiny_tokenizer, max_seq_length=48)
    item = ds[0]
    assert len(item["x"]) <= 47  # ids <= 48, x = ids[:-1]
    assert any(v != -100 for v in item["y"])


def test_user_only_overflow_still_supervised(tmp_path, tiny_tokenizer):
    """A user message alone longer than the window must not produce a
    fully-masked sample (that would be a NaN loss)."""
    path = _write_jsonl(tmp_path / "huge.jsonl", [
        {"messages": _msgs(("user", "long question " * 100), ("assistant", "short answer"))},
    ])
    ds = ConversationDataset(path, tiny_tokenizer, max_seq_length=32)
    item = ds[0]
    assert len(item["y"]) > 0
    assert any(v != -100 for v in item["y"])


def test_collate_pads_and_keeps_masks(tmp_path, tiny_tokenizer):
    path = _write_jsonl(tmp_path / "sft.jsonl", [
        {"messages": _msgs(("user", "short"), ("assistant", "a"))},
        {"messages": _msgs(("user", "a somewhat longer question here"), ("assistant", "and a longer answer as well ok"))},
    ])
    ds = ConversationDataset(path, tiny_tokenizer, max_seq_length=64)
    x, y = _collate_conversations([ds[0], ds[1]], pad_id=tiny_tokenizer.pad_id)
    assert x.shape == y.shape and x.dtype == torch.long
    assert x.shape[0] == 2
    # Shorter row: pad targets must be -100; pad inputs the pad id.
    short = 0 if len(ds[0]["x"]) < len(ds[1]["x"]) else 1
    n_real = len(ds[[0, 1][short]]["x"])
    assert (y[short, n_real:] == -100).all()
    assert (x[short, n_real:] == tiny_tokenizer.pad_id).all()


def test_dataset_rejects_bad_records(tmp_path, tiny_tokenizer):
    p = _write_jsonl(tmp_path / "bad.jsonl", [{"text": "no messages key"}])
    ds = ConversationDataset(p, tiny_tokenizer, max_seq_length=32)
    with pytest.raises(ValueError, match="messages"):
        ds[0]

    p2 = _write_jsonl(tmp_path / "bad2.jsonl", [
        {"messages": [{"role": "system", "content": "nope"}]},
    ])
    ds2 = ConversationDataset(p2, tiny_tokenizer, max_seq_length=32)
    with pytest.raises(ValueError, match="role"):
        ds2[0]


def test_dataset_missing_file_raises(tmp_path, tiny_tokenizer):
    with pytest.raises(FileNotFoundError, match="prepare_finetune_data"):
        ConversationDataset(tmp_path / "nope.jsonl", tiny_tokenizer, max_seq_length=32)


def test_ram_need_bound_is_resume_aware():
    """Fresh runs only touch model pages -> smaller bound; resume -> full file."""
    from types import SimpleNamespace
    from dipai.finetune import _ram_need_gb

    fresh = SimpleNamespace(resume_from_checkpoint=None)
    resume = SimpleNamespace(resume_from_checkpoint="auto")
    file_gb = 0.554 * 1024**3
    need_fresh = _ram_need_gb(fresh, file_gb)
    need_resume = _ram_need_gb(resume, file_gb)
    assert abs(need_resume - (0.554 + 0.35)) < 1e-6
    assert abs(need_fresh - (0.554 * 0.45 + 0.35)) < 1e-6
    assert need_fresh < need_resume
    assert need_fresh >= 0.35  # fixed safety margin always present

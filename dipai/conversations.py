"""Reconstruct OASST1-style message trees into chat conversations.

Why this exists: individual role-tagged messages (scripts/download_oasst.py
default output) are a TEXT corpus, not training conversations. Supervised
fine-tuning needs full user/assistant exchange paths, and the train/val split
must happen at TREE level - fragments of the same tree in both splits would
leak nearly identical content across them.

Input: JSONL with the metadata fields kept by
    python scripts/download_oasst.py --with-metadata
    (message_id, parent_id, message_tree_id, role, lang, deleted,
     review_count, review_result, tree_state, text)

Output records:
    {"messages": [{"role": "user"|"assistant", "content": str}, ...],
     "tree_id": str}

Serialization (format_dialog) is the single source of truth for the
prompt layout used by BOTH SFT training and chat inference:
    "User: <text>\\n\\nAssistant: <text>\\n\\nUser: ..."
so the model always sees at training time the format it sees at inference.

Split: deterministic hash of (seed, tree_id) -> train|val. Every path of a
tree lands in exactly one split (leak-free), independent of ordering/runs.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

log_name = "dipai.conversations"

ROLE_MAP = {"prompter": "user", "assistant": "assistant"}


class ConversationError(Exception):
    """Raised when conversation reconstruction input is missing/invalid."""


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class ConversationStats:
    schema_version: int = 1
    input: str = ""
    messages_read: int = 0
    trees_seen: int = 0
    dropped_lang: int = 0
    dropped_deleted: int = 0
    dropped_unreviewed: int = 0
    dropped_bad_role: int = 0        # unknown role / non-prompter root
    dropped_duplicate_id: int = 0
    paths_seen: int = 0
    dropped_too_short: int = 0       # fewer than min_messages after merging
    dropped_too_long: int = 0        # exceeds max_messages or max_chars
    dropped_duplicate_text: int = 0  # identical serialized conversation
    conversations: int = 0
    train_conversations: int = 0
    val_conversations: int = 0
    train_assistant_turns: int = 0
    val_assistant_turns: int = 0
    val_fraction: float = 0.0
    seed: int = 0
    limits: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "Conversation statistics (OASST threads)",
            "-" * 46,
            f"  messages read          : {self.messages_read}",
            f"  trees                  : {self.trees_seen}",
            f"  dropped lang/deleted   : {self.dropped_lang} / {self.dropped_deleted}",
            f"  dropped unreviewed     : {self.dropped_unreviewed}",
            f"  dropped bad role/id    : {self.dropped_bad_role} / {self.dropped_duplicate_id}",
            f"  paths (tree -> leaf)   : {self.paths_seen}",
            f"  dropped short/long     : {self.dropped_too_short} / {self.dropped_too_long}",
            f"  dropped duplicate text : {self.dropped_duplicate_text}",
            f"  conversations          : {self.conversations}",
            f"  train/val (tree split) : {self.train_conversations} / "
            f"{self.val_conversations}",
            f"  train/val asst. turns  : {self.train_assistant_turns} / "
            f"{self.val_assistant_turns}",
            f"  split (seed)           : {self.val_fraction} val ({self.seed})",
        ]
        if self.limits:
            lines.append(f"  limits                 : {self.limits}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reading + splitting
# ---------------------------------------------------------------------------

def iter_oasst_messages(path: str | Path) -> Iterator[dict]:
    """Yield metadata message dicts from a --with-metadata JSONL file."""
    path = Path(path)
    if not path.exists():
        raise ConversationError(f"OASST metadata file not found: {path}")
    if path.stat().st_size == 0:
        raise ConversationError(f"OASST metadata file is empty: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversationError(
                    f"{path}:{line_no} is not valid JSON ({exc})"
                ) from exc
            if not isinstance(obj, dict):
                raise ConversationError(f"{path}:{line_no} is not a JSON object")
            yield obj


def assign_tree_split(tree_id: str, val_fraction: float, seed: int) -> str:
    """Deterministic leak-free split: every path of a tree -> same side."""
    h = hashlib.sha1(f"{seed}\x00tree\x00{tree_id}".encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "val" if v < val_fraction else "train"


def format_dialog(messages: Iterable[dict]) -> str:
    """Role-labelled serialization: the ONE prompt format for train + chat."""
    return "\n\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: "
        f"{str(m['content']).strip()}"
        for m in messages
    )


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def build_conversations(
    messages: Iterable[dict],
    *,
    val_fraction: float = 0.1,
    seed: int = 1337,
    lang: str = "en",
    min_review_count: int = 1,
    allow_unreviewed: bool = False,
    min_messages: int = 2,
    max_messages: int = 10,
    max_chars: int = 4000,
    limit: int = 0,
    input_name: str = "",
) -> Tuple[List[dict], ConversationStats]:
    """Rebuild user/assistant conversations from a message tree.

    Filters (defaults documented, never silent):
      * lang (if set), deleted == True dropped always
      * review_count >= min_review_count unless allow_unreviewed
        (with the OASST1 English export this drops only messages that
        never passed review; review_result=True always accepted)
      * roots must be prompters; roles mapped prompter->user
      * consecutive same-role messages are MERGED with '\\n' (keeps strict
        alternation for the serialized format)
      * every root->leaf path becomes one conversation (paths of the same
        tree share tree_id -> all land in one split)
      * min/max messages, max_chars, exact-duplicate serialization

    Returns (conversations, stats). Conversations are ordered by tree, and
    the train/val assignment is by hash of tree_id (leak-free, stable).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ConversationError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if min_messages < 2:
        raise ConversationError("min_messages must be >= 2 (need a user+assistant pair)")

    stats = ConversationStats(
        input=input_name, val_fraction=val_fraction, seed=seed,
        limits={
            "min_messages": min_messages, "max_messages": max_messages,
            "max_chars": max_chars, "min_review_count": min_review_count,
            **({"limit": float(limit)} if limit else {}),
        },
    )

    by_id: Dict[str, dict] = {}
    children: Dict[Optional[str], List[str]] = defaultdict(list)
    roots: List[str] = []

    for m in messages:
        stats.messages_read += 1
        if lang and m.get("lang") != lang:
            stats.dropped_lang += 1
            continue
        if m.get("deleted") is True:
            stats.dropped_deleted += 1
            continue
        if not allow_unreviewed:
            reviewed = (m.get("review_count") or 0) >= min_review_count \
                or m.get("review_result") is True
            if not reviewed:
                stats.dropped_unreviewed += 1
                continue
        msg_id = m.get("message_id")
        role = m.get("role")
        text = (m.get("text") or "").replace("\x00", "").strip()
        if not msg_id or role not in ROLE_MAP or not text:
            stats.dropped_bad_role += 1
            continue
        if msg_id in by_id:
            stats.dropped_duplicate_id += 1
            continue
        by_id[msg_id] = {"message_id": msg_id, "role": role,
                         "parent_id": m.get("parent_id"),
                         "tree_id": m.get("message_tree_id") or msg_id,
                         "text": text}
        children[m.get("parent_id")].append(msg_id)
        if m.get("parent_id") is None:
            roots.append(msg_id)

    stats.trees_seen = len({by_id[r]["tree_id"] for r in roots})

    conversations: List[dict] = []
    seen_serialized: set = set()

    def _emit(path: List[str]) -> None:
        nonlocal conversations
        stats.paths_seen += 1
        raw = [by_id[i] for i in path]
        if raw[0]["role"] != "prompter":
            stats.dropped_bad_role += 1
            return
        turns: List[dict] = []
        for m in raw:
            role = ROLE_MAP[m["role"]]
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n" + m["text"]
            else:
                turns.append({"role": role, "content": m["text"]})
        if len(turns) < min_messages:
            stats.dropped_too_short += 1
            return
        if len(turns) > max_messages:
            stats.dropped_too_long += 1
            return
        total_chars = sum(len(t["content"]) for t in turns)
        if total_chars > max_chars:
            stats.dropped_too_long += 1
            return
        blob = format_dialog(turns)
        key = hashlib.sha1(blob.encode("utf-8")).digest()
        if key in seen_serialized:
            stats.dropped_duplicate_text += 1
            return
        seen_serialized.add(key)

        tree_id = raw[0]["tree_id"]
        split = assign_tree_split(tree_id, val_fraction, seed)
        conversations.append({
            "messages": turns,
            "tree_id": tree_id,
            "split": split,
            "assistant_turns": sum(1 for t in turns if t["role"] == "assistant"),
        })

    def _walk(path: List[str]) -> None:
        kids = children.get(path[-1], ())
        if not kids:
            _emit(path)
            return
        for k in kids:
            _walk(path + [k])

    for root in roots:
        _walk([root])

    conversations.sort(key=lambda c: (c["tree_id"], format_dialog(c["messages"])))
    if limit and len(conversations) > limit:
        # Deterministic cap: keep the first N after stable sort.
        conversations = conversations[:limit]

    stats.conversations = len(conversations)
    stats.train_conversations = sum(1 for c in conversations if c["split"] == "train")
    stats.val_conversations = sum(1 for c in conversations if c["split"] == "val")
    stats.train_assistant_turns = sum(
        c["assistant_turns"] for c in conversations if c["split"] == "train"
    )
    stats.val_assistant_turns = sum(
        c["assistant_turns"] for c in conversations if c["split"] == "val"
    )
    return conversations, stats


def write_splits(
    conversations: List[dict],
    train_path: str | Path,
    val_path: str | Path,
) -> Tuple[int, int]:
    """Write train/val JSONL files ({"messages": [...], "tree_id": ...})."""
    train_path, val_path = Path(train_path), Path(val_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    n_train = n_val = 0
    with open(train_path, "w", encoding="utf-8", newline="\n") as f_tr, \
            open(val_path, "w", encoding="utf-8", newline="\n") as f_va:
        for c in conversations:
            rec = {"messages": c["messages"], "tree_id": c["tree_id"]}
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            if c["split"] == "val":
                f_va.write(line)
                n_val += 1
            else:
                f_tr.write(line)
                n_train += 1
    if n_train == 0:
        raise ConversationError(
            "0 training conversations after filtering - relax the filters "
            "(--min-review-count / --max-chars / --limit) or check the input."
        )
    return n_train, n_val

"""Tests for dipai.inference session handling and engine error paths.

These are deliberately CPU-light: no checkpoint is loaded except the
error-path tests (which never touch model weights).
"""

from __future__ import annotations

import pytest

from dipai.inference import (
    ConversationSession,
    DIPAIEngine,
    EngineError,
)


def test_session_add_and_flatten():
    s = ConversationSession()
    assert len(s) == 0
    s.add("user", "hello there")
    s.add("assistant", "hi!")
    assert len(s) == 2
    # Ends on the assistant turn -> plain role-labelled dialog, no header.
    assert s.as_prompt() == "User: hello there\n\nAssistant: hi!"


def test_session_prompt_hands_control_to_assistant():
    s = ConversationSession()
    s.add("user", "what is 2+2?")
    # Ends on the user turn -> trailing "Assistant: " hands control over,
    # matching the SFT training format (format_dialog + header).
    assert s.as_prompt() == "User: what is 2+2?\n\nAssistant: "


def test_session_trims_oldest():
    s = ConversationSession(max_turns=4)
    for i in range(10):
        s.add("user", f"msg {i}")
    assert len(s) == 4
    assert "msg 9" in s.turns[-1]["content"]
    assert "msg 0" not in s.as_prompt()


def test_session_clear():
    s = ConversationSession()
    s.add("user", "x")
    s.clear()
    assert len(s) == 0 and s.as_prompt() == ""


def test_engine_not_loaded_raises():
    eng = DIPAIEngine()
    assert eng.ready is False
    with pytest.raises(EngineError, match="not loaded"):
        eng.generate("hello")


def test_engine_missing_checkpoint_dir(tmp_path):
    with pytest.raises(EngineError, match="not found"):
        DIPAIEngine.from_checkpoint(tmp_path / "does-not-exist")


def test_engine_empty_checkpoint_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(EngineError, match="no checkpoint"):
        DIPAIEngine.from_checkpoint(tmp_path / "empty")


def test_status_line_defaults():
    eng = DIPAIEngine()
    line = eng.status().line()
    assert isinstance(line, str) and line

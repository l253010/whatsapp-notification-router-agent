"""Tests for agent.py — prompt building and safe fallback behavior.

These tests do NOT call the real Gemini API. The "no API key" tests exercise
the real fallback path. The "mocked response" test fakes the google.genai
module in sys.modules so get_agent_decision's parsing logic can be tested
without network access or a real key.
"""

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent import build_user_prompt, get_agent_decision
from context import MessageContext


@dataclass
class FakeMessage:
    message_id: str = "m1"
    user_id: str = "user_1"
    conversation_type: str = "personal"
    group_id: str | None = None
    business_id: str | None = None
    sender_user_id: str = "sender_9"
    created_at: str = "2026-07-31T09:00:00"
    message_text: str = "are we still meeting at 5"
    media_type: str | None = None
    media_id: str | None = None
    forwarded_count: int = 0
    is_valid: bool = True


def _empty_context() -> MessageContext:
    return MessageContext(
        user=None, group=None, membership=None, business=None, biz_history=None,
        sender_history=[], repetition_count=0, is_likely_spam_pattern=False,
        evidence_message_ids=[], media_info=None,
    )


# ---------- build_user_prompt ----------

def test_build_user_prompt_contains_message_text():
    prompt = build_user_prompt(FakeMessage(), _empty_context(), missing_field_flags=[])
    assert "are we still meeting at 5" in prompt
    assert "message:" in prompt
    assert "context:" in prompt


def test_build_user_prompt_includes_missing_field_flags_when_present():
    prompt = build_user_prompt(FakeMessage(), _empty_context(), missing_field_flags=["timestamp_unresolved"])
    assert "missing_field_flags" in prompt
    assert "timestamp_unresolved" in prompt


def test_build_user_prompt_uses_media_transcript_for_voice_messages():
    msg = FakeMessage(media_type="voice", media_id="voice_1", message_text=None)
    prompt = build_user_prompt(msg, _empty_context(), missing_field_flags=[], media_transcript="hey call me back")
    assert "hey call me back" in prompt


# ---------- get_agent_decision: fallback when no API key ----------

def test_get_agent_decision_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    decision = get_agent_decision(FakeMessage(), _empty_context(), missing_field_flags=[], api_key=None)
    assert decision.action == "mute"
    assert decision.reason == "agent call failed, fallback applied"
    assert decision.confidence == 0.3


# ---------- get_agent_decision: mocked Gemini response ----------

def test_get_agent_decision_parses_mocked_response(monkeypatch):
    """Fake out google.genai so we can test the happy path without a real API call."""

    class FakeResponse:
        text = (
            '{"action": "notify", "message_type": "personal", '
            '"reason": "genuine question needing a reply", '
            '"confidence": 0.87, "evidence_message_ids": []}'
        )

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient

    fake_types = types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = lambda **kwargs: kwargs

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    decision = get_agent_decision(
        FakeMessage(), _empty_context(), missing_field_flags=[], api_key="fake-key-for-testing"
    )
    assert decision.action == "notify"
    assert decision.message_type == "personal"
    assert decision.confidence == 0.87


def test_get_agent_decision_reads_api_key_from_dotenv_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    class FakeResponse:
        text = (
            '{"action": "digest", "message_type": "promotion", '
            '"reason": "low priority but relevant", '
            '"confidence": 0.72, "evidence_message_ids": []}'
        )

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient

    fake_types = types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = lambda **kwargs: kwargs

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    decision = get_agent_decision(
        FakeMessage(),
        _empty_context(),
        missing_field_flags=[],
        api_key=None,
        env_path=str(env_file),
    )
    assert decision.action == "digest"
    assert decision.message_type == "promotion"
    assert decision.confidence == 0.72


def test_get_agent_decision_falls_back_on_malformed_json(monkeypatch):
    class FakeResponse:
        text = '{"action": "notify", "message_type": "personal"'  # malformed JSON

    class FakeModels:
        def generate_content(self, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            self.models = FakeModels()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient

    fake_types = types.ModuleType("google.genai.types")
    fake_types.GenerateContentConfig = lambda **kwargs: kwargs

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    decision = get_agent_decision(
        FakeMessage(), _empty_context(), missing_field_flags=[], api_key="fake-key-for-testing"
    )
    assert decision.action == "mute"
    assert decision.reason == "agent call failed, fallback applied"
    assert decision.confidence == 0.3

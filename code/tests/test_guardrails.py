"""Tests for guardrails.py — pre-agent checks, post-agent normalization, coverage, CSV output."""

import csv
import os
import sys
import tempfile
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from guardrails import (
    pre_check,
    post_check,
    ensure_full_coverage,
    decision_to_csv_row,
    write_output_csv,
    AgentDecision,
)


@dataclass
class FakeMessage:
    user_id: str | None = None
    message_text: str | None = None
    media_type: str | None = None
    conversation_type: str | None = None
    group_id: str | None = None
    business_id: str | None = None
    sender_user_id: str | None = None
    created_at: object = None
    media_processing_unavailable: bool = False


# ---------- pre_check ----------

def test_pre_check_skips_agent_when_user_id_missing():
    msg = FakeMessage(user_id=None, message_text="hello")
    result = pre_check(msg)
    assert result.skip_agent is True
    assert result.action == "mute"
    assert result.reason == "missing user_id, message cannot be routed"


def test_pre_check_skips_agent_when_no_content():
    msg = FakeMessage(user_id="user_1", message_text="   ", media_type=None)
    result = pre_check(msg)
    assert result.skip_agent is True
    assert result.reason == "no text or media content present"


def test_pre_check_flags_unresolved_group_context():
    msg = FakeMessage(
        user_id="user_1",
        message_text="hi",
        conversation_type="group",
        group_id=None,
        created_at="2026-07-31T09:00:00",
    )
    result = pre_check(msg)
    assert result.skip_agent is False
    assert "group_context_unresolved" in result.missing_field_flags


def test_pre_check_flags_missing_timestamp():
    msg = FakeMessage(
        user_id="user_1",
        message_text="hi",
        conversation_type="personal",
        sender_user_id="sender_9",
        created_at=None,
    )
    result = pre_check(msg)
    assert result.skip_agent is False
    assert "timestamp_unresolved" in result.missing_field_flags


def test_pre_check_clean_message_has_no_flags():
    msg = FakeMessage(
        user_id="user_1",
        message_text="hi",
        conversation_type="personal",
        sender_user_id="sender_9",
        created_at="2026-07-31T09:00:00",
    )
    result = pre_check(msg)
    assert result.skip_agent is False
    assert result.missing_field_flags == []


def test_pre_check_routes_media_processing_failure_to_digest():
    msg = FakeMessage(
        user_id="user_1",
        message_text="hello",
        media_type="image",
        media_processing_unavailable=True,
    )
    result = pre_check(msg)
    assert result.skip_agent is True
    assert result.action == "digest"
    assert result.message_type == "unknown"
    assert result.reason == "media processing unavailable, manual review recommended"
    assert result.confidence == 0.3
    assert result.evidence_message_ids == []


# ---------- post_check ----------

def test_post_check_fixes_invalid_action():
    decision = AgentDecision(
        message_id="m1",
        action="delete",  # not in ALLOWED_ACTIONS
        message_type="personal",
        reason="",
        confidence=5,
        evidence_message_ids=None,
    )
    fixed = post_check(decision)
    assert fixed.action == "mute"
    assert fixed.confidence == 1.0
    assert fixed.evidence_message_ids == []
    assert fixed.reason == "no reason provided"


def test_post_check_clamps_confidence_low():
    decision = AgentDecision(
        message_id="m1", action="notify", message_type="personal",
        reason="ok", confidence=-3, evidence_message_ids=[],
    )
    fixed = post_check(decision)
    assert fixed.confidence == 0.0


def test_post_check_fixes_bad_message_type():
    decision = AgentDecision(
        message_id="m1", action="notify", message_type="not_a_type",
        reason="ok", confidence=0.5, evidence_message_ids=[],
    )
    fixed = post_check(decision)
    assert fixed.message_type == "unknown"


# ---------- ensure_full_coverage ----------

def test_ensure_full_coverage_adds_missing_ids():
    all_ids = ["m1", "m2", "m3"]
    existing = [
        AgentDecision(message_id="m1", action="notify", message_type="personal",
                      reason="ok", confidence=0.9, evidence_message_ids=[])
    ]
    completed = ensure_full_coverage(all_ids, existing)
    ids_present = {d.message_id for d in completed}
    assert ids_present == {"m1", "m2", "m3"}
    fallback = next(d for d in completed if d.message_id == "m2")
    assert fallback.action == "mute"
    assert fallback.reason == "no decision generated, fallback applied"


def test_ensure_full_coverage_no_duplicates_when_all_present():
    all_ids = ["m1"]
    existing = [
        AgentDecision(message_id="m1", action="digest", message_type="event",
                      reason="ok", confidence=0.6, evidence_message_ids=[])
    ]
    completed = ensure_full_coverage(all_ids, existing)
    assert len(completed) == 1


# ---------- decision_to_csv_row / write_output_csv ----------

def test_decision_to_csv_row_formats_evidence_and_confidence():
    decision = AgentDecision(
        message_id="m1", action="notify", message_type="urgent",
        reason="time sensitive", confidence=0.856, evidence_message_ids=["a", "b"],
    )
    row = decision_to_csv_row(decision)
    assert row == ["m1", "notify", "urgent", "time sensitive", "0.86", "a;b"]


def test_decision_to_csv_row_none_evidence_becomes_literal_none():
    decision = AgentDecision(
        message_id="m1", action="mute", message_type="spam",
        reason="spam", confidence=0.9, evidence_message_ids=[],
    )
    row = decision_to_csv_row(decision)
    assert row[-1] == "none"


def test_write_output_csv_produces_expected_header_and_rows():
    decisions = [
        AgentDecision(message_id="m1", action="notify", message_type="urgent",
                      reason="ok", confidence=0.9, evidence_message_ids=[]),
        AgentDecision(message_id="m2", action="mute", message_type="spam",
                      reason="spammy", confidence=0.7, evidence_message_ids=["m0"]),
    ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "output.csv")
        write_output_csv(decisions, out_path)
        with open(out_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
        assert rows[1] == ["m1", "notify", "urgent", "ok", "0.90", "none"]
        assert rows[2] == ["m2", "mute", "spam", "spammy", "0.70", "m0"]

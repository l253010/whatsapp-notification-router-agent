"""Tests for parser.py — CSV parsing and row-level validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parser import parse_messages

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MESSAGES_CSV = os.path.join(DATASET_DIR, "messages.csv")


def test_parses_all_rows():
    messages = parse_messages(MESSAGES_CSV)
    assert len(messages) == 5


def test_valid_group_message_is_valid():
    messages = parse_messages(MESSAGES_CSV)
    msg = next(m for m in messages if m.message_id == "msg_0001")
    assert msg.is_valid is True
    assert msg.conversation_type == "group"
    assert msg.invalid_reason is None


def test_missing_sender_on_personal_message_is_invalid():
    messages = parse_messages(MESSAGES_CSV)
    msg = next(m for m in messages if m.message_id == "msg_0003")
    assert msg.is_valid is False
    assert msg.invalid_reason == "sender_user_id missing for personal message"


def test_missing_user_id_is_invalid():
    messages = parse_messages(MESSAGES_CSV)
    msg = next(m for m in messages if m.message_id == "msg_0004")
    assert msg.is_valid is False
    assert msg.invalid_reason == "user_id is missing"


def test_forwarded_count_defaults_to_zero():
    messages = parse_messages(MESSAGES_CSV)
    msg = next(m for m in messages if m.message_id == "msg_0001")
    assert msg.forwarded_count == 0


def test_no_content_message_is_invalid():
    """A row with no message_text, media_type, or media_id should fail validation."""
    import csv
    import tempfile

    rows = [
        {
            "message_id": "no_content_1",
            "user_id": "user_1",
            "conversation_type": "personal",
            "group_id": "",
            "business_id": "",
            "sender_user_id": "sender_9",
            "created_at": "2026-07-31T09:00:00",
            "message_text": "",
            "media_type": "",
            "media_id": "",
            "forwarded_count": "0",
        }
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        tmp_path = tmp.name

    try:
        messages = parse_messages(tmp_path)
        assert messages[0].is_valid is False
        assert messages[0].invalid_reason == "no content: text and media both missing"
    finally:
        os.remove(tmp_path)

"""Tests for the main message-processing pipeline."""

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import process_message


@dataclass
class FakeMessage:
    message_id: str = "msg_1"
    user_id: str = "user_1"
    conversation_type: str = "personal"
    group_id: str | None = None
    business_id: str | None = None
    sender_user_id: str = "sender_1"
    created_at: str = "2026-07-31T09:00:00"
    message_text: str = "hello"
    media_type: str | None = "image"
    media_id: str | None = "img_1"
    forwarded_count: int = 0
    is_valid: bool = True
    invalid_reason: str | None = None


def test_process_message_routes_to_digest_when_media_transcript_fails():
    message = FakeMessage()
    store = SimpleNamespace(
        images_index={"img_1": {"file_path": "dummy.jpg"}},
        voice_index={},
        users_index={},
        groups_index={},
        group_members_index={},
        business_index={},
        user_business_index={},
        history_by_sender={},
        events_by_message_id={},
        daily_summary_index={},
    )

    with patch("main.tools.get_media_transcript", return_value="[image could not be processed]"), patch(
        "main.context.get_context", return_value=SimpleNamespace()
    ) as get_context_mock, patch("main.agent.get_agent_decision", side_effect=AssertionError("should not call agent")):
        decision = process_message(message, store, api_key="test-key")

    assert decision.action == "digest"
    assert decision.message_type == "unknown"
    assert decision.reason == "media processing unavailable, manual review recommended"
    assert decision.confidence == 0.3
    assert decision.evidence_message_ids == []
    get_context_mock.assert_not_called()

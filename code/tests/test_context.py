"""Tests for context.py — index loading and per-message context lookup."""

import os
import sys
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from context import load_all_indices, get_context

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


@dataclass
class FakeMessage:
    """Stand-in for ParsedMessage so context tests don't depend on parser.py."""

    user_id: str | None = None
    group_id: str | None = None
    business_id: str | None = None
    sender_user_id: str | None = None
    message_text: str | None = None
    created_at: object = None
    media_type: str | None = None
    media_id: str | None = None


def test_load_all_indices_reads_every_file():
    store = load_all_indices(DATASET_DIR)
    assert "user_1" in store.users_index
    assert "group_1" in store.groups_index
    assert ("sender_9", "group_1") in store.group_members_index
    assert "biz_1" in store.business_index
    assert "sender_9" in store.history_by_sender
    assert len(store.history_by_sender["sender_9"]) == 3


def test_get_context_resolves_user_and_group():
    store = load_all_indices(DATASET_DIR)
    msg = FakeMessage(user_id="user_1", group_id="group_1", sender_user_id="sender_9")
    ctx = get_context(msg, store)
    assert ctx.user["name"] == "Ayesha"
    assert ctx.group["group_name"] == "Neighborhood Watch"
    assert ctx.membership is not None


def test_repetition_and_spam_pattern_detected():
    """Same sender, repeated near-identical promo text, same day -> spam pattern True."""
    store = load_all_indices(DATASET_DIR)
    msg = FakeMessage(
        user_id="user_2",
        sender_user_id="sender_9",
        message_text="Buy our new kurta collection now",
        created_at=datetime(2026, 7, 30, 13, 0, 0),
    )
    ctx = get_context(msg, store)
    assert ctx.repetition_count == 3
    assert ctx.is_likely_spam_pattern is True
    assert "hist_1" in ctx.evidence_message_ids


def test_no_history_gives_empty_context():
    store = load_all_indices(DATASET_DIR)
    msg = FakeMessage(user_id="user_1", sender_user_id="unknown_sender")
    ctx = get_context(msg, store)
    assert ctx.sender_history == []
    assert ctx.repetition_count == 0
    assert ctx.is_likely_spam_pattern is False


def test_media_info_resolved_for_image():
    store = load_all_indices(DATASET_DIR)
    msg = FakeMessage(user_id="user_1", media_type="image", media_id="img_1")
    ctx = get_context(msg, store)
    assert ctx.media_info["caption"] == "kurta set photo"

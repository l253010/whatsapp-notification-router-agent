"""Context indexing and lookup utilities for the WhatsApp routing pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _clean(value: Any) -> str | None:
	"""Convert empty CSV cells to None while preserving raw non-empty strings."""
	if value is None:
		return None
	text = str(value)
	return text if text != "" else None


def _parse_datetime(value: str | None) -> datetime | None:
	"""Best-effort parse of timestamp strings used in dataset history files."""
	if not value:
		return None
	try:
		return datetime.fromisoformat(value)
	except ValueError:
		return None


def _repetition_similarity(a: str | None, b: str | None) -> bool:
	"""Return True when two messages are close enough to count as repeated content."""
	left = (a or "").strip().lower()
	right = (b or "").strip().lower()
	if not left or not right:
		return False
	if left == right:
		return True
	if len(left) < 20 or len(right) < 20:
		return left == right
	return left[:80] == right[:80] or left in right or right in left


@dataclass
class ContextStore:
	"""All dataset-backed lookup tables needed to build per-message context."""

	users_index: dict[str, dict]
	groups_index: dict[str, dict]
	group_members_index: dict[tuple[str, str], dict]
	business_index: dict[str, dict]
	user_business_index: dict[tuple[str, str], dict]
	history_by_sender: dict[str, list[dict]]
	events_by_message_id: dict[str, list[dict]]
	images_index: dict[str, dict]
	voice_index: dict[str, dict]
	daily_summary_index: dict[str, dict]


@dataclass
class MessageContext:
	"""Derived context for one parsed message, computed entirely from cached indices."""

	user: dict | None
	group: dict | None
	membership: dict | None
	business: dict | None
	biz_history: dict | None
	sender_history: list[dict]
	repetition_count: int
	is_likely_spam_pattern: bool
	evidence_message_ids: list[str]
	media_info: dict | None


def _read_csv_rows(csv_path: Path) -> list[dict]:
	"""Read a CSV file once and return its raw DictReader rows."""
	with csv_path.open("r", encoding="utf-8", newline="") as handle:
		reader = csv.DictReader(handle)
		return [dict(row) for row in reader]


def load_all_indices(dataset_dir: str) -> ContextStore:
	"""Load all supporting CSVs into memory once and build raw lookup indices."""
	dataset_path = Path(dataset_dir)

	users_index: dict[str, dict] = {}
	for row in _read_csv_rows(dataset_path / "users.csv"):
		user_id = row.get("user_id")
		if user_id and user_id not in users_index:
			users_index[user_id] = row

	groups_index: dict[str, dict] = {}
	for row in _read_csv_rows(dataset_path / "groups.csv"):
		group_id = row.get("group_id")
		if group_id and group_id not in groups_index:
			groups_index[group_id] = row

	group_members_index: dict[tuple[str, str], dict] = {}
	for row in _read_csv_rows(dataset_path / "group_members.csv"):
		user_id = row.get("user_id")
		group_id = row.get("group_id")
		if user_id and group_id and (user_id, group_id) not in group_members_index:
			group_members_index[(user_id, group_id)] = row

	business_index: dict[str, dict] = {}
	for row in _read_csv_rows(dataset_path / "business_accounts.csv"):
		business_id = row.get("business_id")
		if business_id and business_id not in business_index:
			business_index[business_id] = row

	user_business_index: dict[tuple[str, str], dict] = {}
	for row in _read_csv_rows(dataset_path / "user_business_history.csv"):
		user_id = row.get("user_id")
		business_id = row.get("business_id")
		if user_id and business_id and (user_id, business_id) not in user_business_index:
			user_business_index[(user_id, business_id)] = row

	history_by_sender: dict[str, list[dict]] = {}
	for row in _read_csv_rows(dataset_path / "message_history.csv"):
		sender_user_id = row.get("sender_user_id")
		if sender_user_id:
			history_by_sender.setdefault(sender_user_id, []).append(row)

	events_by_message_id: dict[str, list[dict]] = {}
	for row in _read_csv_rows(dataset_path / "message_events.csv"):
		message_id = row.get("message_id")
		if message_id:
			events_by_message_id.setdefault(message_id, []).append(row)

	# images_index: dict[str, dict] = {}
	# for row in _read_csv_rows(dataset_path / "images.csv"):
	# 	image_id = row.get("image_id")
	# 	if image_id and image_id not in images_index:
	# 		images_index[image_id] = row

	# voice_index: dict[str, dict] = {}
	# for row in _read_csv_rows(dataset_path / "voice_notes.csv"):
	# 	voice_id = row.get("voice_note_id")
	# 	if voice_id and voice_id not in voice_index:
	# 		voice_index[voice_id] = row

	images_index: dict[str, dict] = {}
	for row in _read_csv_rows(dataset_path / "images.csv"):
		image_id = row.get("image_id")
		if image_id and image_id not in images_index:
			file_path = row.get("file_path")
			if file_path:
				row["file_path"] = str((dataset_path / file_path).resolve())
			images_index[image_id] = row

	voice_index: dict[str, dict] = {}
	for row in _read_csv_rows(dataset_path / "voice_notes.csv"):
		voice_id = row.get("voice_note_id")
		if voice_id and voice_id not in voice_index:
			file_path = row.get("file_path")
			if file_path:
				row["file_path"] = str((dataset_path / file_path).resolve())
			voice_index[voice_id] = row

	daily_summary_index: dict[str, dict] = {}
	for row in _read_csv_rows(dataset_path / "daily_notification_summary.csv"):
		user_id = row.get("user_id")
		if user_id and user_id not in daily_summary_index:
			daily_summary_index[user_id] = row

	return ContextStore(
		users_index=users_index,
		groups_index=groups_index,
		group_members_index=group_members_index,
		business_index=business_index,
		user_business_index=user_business_index,
		history_by_sender=history_by_sender,
		events_by_message_id=events_by_message_id,
		images_index=images_index,
		voice_index=voice_index,
		daily_summary_index=daily_summary_index,
	)


def get_context(message, store: ContextStore) -> MessageContext:
	"""Build per-message lookup context using only in-memory indices and pure dict access."""
	user = store.users_index.get(getattr(message, "user_id", None))
	group = store.groups_index.get(getattr(message, "group_id", None))
	membership = store.group_members_index.get((getattr(message, "sender_user_id", None), getattr(message, "group_id", None)))
	business = store.business_index.get(getattr(message, "business_id", None))
	biz_history = store.user_business_index.get((getattr(message, "user_id", None), getattr(message, "business_id", None)))

	sender_user_id = getattr(message, "sender_user_id", None)
	sender_history = list(store.history_by_sender.get(sender_user_id, [])) if sender_user_id else []

	repetition_count = 0
	if sender_user_id:
		repetition_count = sum(1 for row in sender_history if row.get("sender_user_id") == sender_user_id)

	message_text = getattr(message, "message_text", None)
	message_time = getattr(message, "created_at", None)
	message_day = message_time.date() if isinstance(message_time, datetime) else None
	matching_rows: list[dict] = []
	for row in sender_history:
		if row.get("sender_user_id") != sender_user_id:
			continue
		if _repetition_similarity(row.get("message_text"), message_text):
			row_time = _parse_datetime(row.get("created_at"))
			if message_day is None or row_time is None or row_time.date() == message_day:
				matching_rows.append(row)

	is_likely_spam_pattern = repetition_count >= 3 and len(matching_rows) >= 2

	evidence_message_ids: list[str] = []
	for row in reversed(sender_history):
		message_id = row.get("message_id")
		if message_id:
			evidence_message_ids.append(message_id)
		if len(evidence_message_ids) >= 5:
			break

	media_info = None
	media_type = getattr(message, "media_type", None)
	media_id = getattr(message, "media_id", None)
	if media_type == "image" and media_id:
		media_info = store.images_index.get(media_id)
	elif media_type == "voice" and media_id:
		media_info = store.voice_index.get(media_id)

	return MessageContext(
		user=user,
		group=group,
		membership=membership,
		business=business,
		biz_history=biz_history,
		sender_history=sender_history,
		repetition_count=repetition_count,
		is_likely_spam_pattern=is_likely_spam_pattern,
		evidence_message_ids=evidence_message_ids,
		media_info=media_info,
	)

"""CSV parsing and validation for WhatsApp urgency-filtering messages."""

from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass
class ParsedMessage:
	"""Represents one parsed CSV row plus validation outcome."""

	message_id: str | None
	user_id: str | None
	conversation_type: str | None
	group_id: str | None
	business_id: str | None
	sender_user_id: str | None
	created_at: str | None
	message_text: str | None
	media_type: str | None
	media_id: str | None
	forwarded_count: int
	is_valid: bool
	invalid_reason: str | None


def _clean(value: object) -> str | None:
	"""Normalize CSV field values by converting empty/whitespace values to None."""
	if value is None:
		return None
	text = str(value).strip()
	return text if text else None


def _validate(row: dict, conversation_type: str | None) -> tuple[bool, str | None]:
	"""Validate one CSV row and return (is_valid, invalid_reason)."""
	message_id = _clean(row.get("message_id"))
	user_id = _clean(row.get("user_id"))

	if not message_id and not user_id:
		return False, "message_id and user_id are missing"
	if not message_id:
		return False, "message_id is missing"
	if not user_id:
		return False, "user_id is missing"

	valid_types = {"group", "business", "personal"}
	if conversation_type not in valid_types:
		return False, "invalid conversation_type"

	if conversation_type == "group" and not _clean(row.get("group_id")):
		return False, "group_id missing for group message"
	if conversation_type == "business" and not _clean(row.get("business_id")):
		return False, "business_id missing for business message"
	if conversation_type == "personal" and not _clean(row.get("sender_user_id")):
		return False, "sender_user_id missing for personal message"

	message_text = _clean(row.get("message_text"))
	media_type = _clean(row.get("media_type"))
	media_id = _clean(row.get("media_id"))
	if not message_text and not media_type and not media_id:
		return False, "no content: text and media both missing"

	forwarded_raw = _clean(row.get("forwarded_count"))
	if forwarded_raw is not None:
		try:
			int(forwarded_raw)
		except ValueError:
			return False, "forwarded_count is not a valid integer"

	return True, None


def _row_to_message(row: dict) -> ParsedMessage:
	"""Convert one DictReader row into ParsedMessage, including validation flags."""
	conversation_type = _clean(row.get("conversation_type"))

	forwarded_raw = _clean(row.get("forwarded_count"))
	forwarded_count = 0
	if forwarded_raw is not None:
		try:
			forwarded_count = int(forwarded_raw)
		except ValueError:
			# Keep a safe default while surfacing the issue through validation.
			forwarded_count = 0

	is_valid, invalid_reason = _validate(row, conversation_type)

	return ParsedMessage(
		message_id=_clean(row.get("message_id")),
		user_id=_clean(row.get("user_id")),
		conversation_type=conversation_type,
		group_id=_clean(row.get("group_id")),
		business_id=_clean(row.get("business_id")),
		sender_user_id=_clean(row.get("sender_user_id")),
		created_at=_clean(row.get("created_at")),
		message_text=_clean(row.get("message_text")),
		media_type=_clean(row.get("media_type")),
		media_id=_clean(row.get("media_id")),
		forwarded_count=forwarded_count,
		is_valid=is_valid,
		invalid_reason=invalid_reason,
	)


def parse_messages(filepath: str) -> list[ParsedMessage]:
	"""Read a CSV file eagerly and return every row as a ParsedMessage."""
	messages: list[ParsedMessage] = []
	with open(filepath, "r", encoding="utf-8", newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		for row in reader:
			if row is None:
				continue
			messages.append(_row_to_message(row))
	return messages


if __name__ == "__main__":
	# Minimal usage example: python code/evaluation/parser.py dataset/messages.csv
	import sys

	if len(sys.argv) < 2:
		raise SystemExit("Usage: python parser.py <path-to-csv>")

	parsed = parse_messages(sys.argv[1])
	print(f"Parsed {len(parsed)} messages")

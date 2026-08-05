"""Minimal validation guardrails for the WhatsApp routing pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Any

try:
	from .parser import ParsedMessage
except ImportError:  # pragma: no cover - fallback for direct execution
	try:
		from parser import ParsedMessage  # type: ignore
	except ImportError:  # pragma: no cover - fallback for incomplete workspaces
		ParsedMessage = Any  # type: ignore

try:
	from .agent import AgentDecision
except ImportError:  # pragma: no cover - fallback for direct execution
	try:
		from agent import AgentDecision  # type: ignore
	except ImportError:  # pragma: no cover - fallback for incomplete workspaces

		@dataclass
		class AgentDecision:  # type: ignore[no-redef]
			"""Fallback decision model used when the agent module is not yet implemented."""

			message_id: str
			action: str
			message_type: str
			reason: str
			confidence: float
			evidence_message_ids: list[str]


ALLOWED_ACTIONS = {"notify", "digest", "mute"}
ALLOWED_MESSAGE_TYPES = {
	"personal",
	"urgent",
	"event",
	"payment",
	"business_update",
	"promotion",
	"greeting",
	"forward",
	"spam",
	"scam",
	"unknown",
}


@dataclass
class PreCheckResult:
	"""Outcome of the pre-agent guardrail checks for a parsed message."""

	skip_agent: bool
	action: str | None = None
	message_type: str | None = None
	reason: str | None = None
	confidence: float | None = None
	evidence_message_ids: list[str] | None = None
	missing_field_flags: list[str] | None = None


def pre_check(message: Any) -> PreCheckResult:
	"""Run lightweight pre-agent validation and return a routing-safe fallback when needed."""
	user_id = getattr(message, "user_id", None)
	if user_id is None or user_id == "":
		return PreCheckResult(
			skip_agent=True,
			action="mute",
			message_type="unknown",
			reason="missing user_id, message cannot be routed",
			confidence=0.95,
			evidence_message_ids=[],
		)

	message_text = getattr(message, "message_text", "")
	media_type = getattr(message, "media_type", None)
	if (message_text or "").strip() == "" and media_type is None:
		return PreCheckResult(
			skip_agent=True,
			action="mute",
			message_type="unknown",
			reason="no text or media content present",
			confidence=0.95,
			evidence_message_ids=[],
		)

	media_processing_unavailable = getattr(message, "media_processing_unavailable", False)
	if media_processing_unavailable:
		return PreCheckResult(
			skip_agent=True,
			action="digest",
			message_type="unknown",
			reason="media processing unavailable, manual review recommended",
			confidence=0.3,
			evidence_message_ids=[],
		)

	flags: list[str] = []
	conversation_type = getattr(message, "conversation_type", None)
	if conversation_type == "group" and getattr(message, "group_id", None) is None:
		flags.append("group_context_unresolved")
	if conversation_type == "business" and getattr(message, "business_id", None) is None:
		flags.append("business_context_unresolved")
	if conversation_type == "personal" and getattr(message, "sender_user_id", None) is None:
		flags.append("sender_identity_unresolved")
	if getattr(message, "created_at", None) is None:
		flags.append("timestamp_unresolved")

	return PreCheckResult(skip_agent=False, missing_field_flags=flags)


def post_check(agent_output: AgentDecision) -> AgentDecision:
	"""Normalize agent output fields without applying business-logic overrides."""
	if getattr(agent_output, "action", None) not in ALLOWED_ACTIONS:
		agent_output.action = "digest"

	if getattr(agent_output, "message_type", None) not in ALLOWED_MESSAGE_TYPES:
		agent_output.message_type = "unknown"

	confidence = getattr(agent_output, "confidence", 0.0)
	try:
		confidence = float(confidence)
	except (TypeError, ValueError):
		confidence = 0.0
	agent_output.confidence = max(0.0, min(1.0, confidence))

	if getattr(agent_output, "evidence_message_ids", None) is None:
		agent_output.evidence_message_ids = []

	reason = getattr(agent_output, "reason", None)
	if reason is None or reason == "":
		agent_output.reason = "no reason provided"

	return agent_output


def ensure_full_coverage(all_message_ids: list[str], output_rows: list[AgentDecision]) -> list[AgentDecision]:
	"""Append fallback decisions for any message IDs that were not assigned a decision."""
	seen_message_ids = {getattr(row, "message_id", None) for row in output_rows}
	completed_rows = list(output_rows)

	for message_id in all_message_ids:
		if message_id not in seen_message_ids:
			completed_rows.append(
				AgentDecision(
					message_id=message_id,
					action="digest",
					message_type="unknown",
					reason="no decision generated, fallback applied",
					confidence=0.5,
					evidence_message_ids=[],
				)
			)

	return completed_rows


def decision_to_csv_row(decision: AgentDecision) -> list[str]:
	"""Convert a validated decision into one CSV row in the required output column order."""
	evidence_message_ids = getattr(decision, "evidence_message_ids", []) or []
	evidence_value = "none" if not evidence_message_ids else ";".join(str(message_id) for message_id in evidence_message_ids)
	confidence_value = f"{float(getattr(decision, 'confidence', 0.0)):.2f}"
	return [
		str(getattr(decision, "message_id", "")),
		str(getattr(decision, "action", "")),
		str(getattr(decision, "message_type", "")),
		str(getattr(decision, "reason", "")),
		confidence_value,
		evidence_value,
	]


def write_output_csv(decisions: list[AgentDecision], output_path: str) -> None:
	"""Write the final output.csv file with the required header and one row per decision."""
	with open(output_path, "w", encoding="utf-8", newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([
			"message_id",
			"action",
			"message_type",
			"reason",
			"confidence",
			"evidence_message_ids",
		])
		for decision in decisions:
			writer.writerow(decision_to_csv_row(decision))

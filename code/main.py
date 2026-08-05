"""End-to-end message routing runner for the evaluation pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time
from typing import Any

try:
	from . import agent, context, guardrails, tools
	from .parser import parse_messages
except ImportError:  # pragma: no cover - fallback for direct execution
	import agent  # type: ignore
	import context  # type: ignore
	import guardrails  # type: ignore
	import tools  # type: ignore
	from parser import parse_messages  # type: ignore


def _build_decision_from_precheck(message_id: str, result: Any) -> Any:
	"""Construct an AgentDecision-shaped object from a pre-check result."""
	return guardrails.AgentDecision(
		message_id=message_id,
		action=result.action or "mute",
		message_type=result.message_type or "unknown",
		reason=result.reason or "no reason provided",
		confidence=result.confidence if result.confidence is not None else 0.3,
		evidence_message_ids=list(result.evidence_message_ids or []),
	)


def _looks_like_media_failure(transcript: str | None) -> bool:
	"""Return True when media processing produced a fallback/transcript failure marker."""
	if transcript is None:
		return True
	if not isinstance(transcript, str):
		return False
	text = transcript.strip()
	return text in {"[image could not be processed]", "[voice note could not be transcribed]"}


def process_message(message: Any, store: Any, api_key: str | None = None, env_path: str | None = None) -> Any:
	"""Process one message in a plain sequential loop and return a final decision."""
	message_id = getattr(message, "message_id", "") or "unknown"
	try:
		if getattr(message, "is_valid", True) is False:
			pre_result = guardrails.pre_check(message)
			if pre_result.skip_agent:
				return _build_decision_from_precheck(message_id, pre_result)
			missing_field_flags = pre_result.missing_field_flags or []
		else:
			missing_field_flags = []

		media_transcript: str | None = None
		media_type = getattr(message, "media_type", None)
		if media_type in {"image", "voice"}:
			media_transcript = tools.get_media_transcript(message, store.images_index, store.voice_index, api_key or "")
			if _looks_like_media_failure(media_transcript):
				setattr(message, "media_processing_unavailable", True)
				pre_result = guardrails.pre_check(message)
				if pre_result.skip_agent:
					return _build_decision_from_precheck(message_id, pre_result)
			else:
				setattr(message, "media_processing_unavailable", False)

		msg_context = context.get_context(message, store)
		raw_decision = agent.get_agent_decision(
			message,
			msg_context,
			missing_field_flags,
			media_transcript,
			api_key,
			env_path,
		)
		return guardrails.post_check(raw_decision)
	except Exception as exc:
		print(f"message_id={message_id} error={exc}", flush=True)
		return guardrails.AgentDecision(
			message_id=message_id,
			action="digest",
			message_type="unknown",
			reason="processing error, fallback applied",
			confidence=0.3,
			evidence_message_ids=[],
		)


def main() -> None:
	"""Parse the dataset, route each message, and write output.csv."""
	repo_root = Path(__file__).resolve().parents[2]
	dataset_dir = repo_root / "dataset"
	messages_path = dataset_dir / "messages.csv"
	output_path = dataset_dir / "output.csv"

	api_key = os.environ.get("GEMINI_API_KEY")
	store = context.load_all_indices(str(dataset_dir))
	messages = parse_messages(str(messages_path))

	decisions: list[Any] = []
	for message in messages:
		decision = process_message(message, store, api_key=api_key, env_path=str(repo_root / ".env"))
		decisions.append(decision)

		print(
			f"message_id={getattr(message, 'message_id', '')} action={getattr(decision, 'action', '')} message_type={getattr(decision, 'message_type', '')}",
			flush=True,
		)

		guardrails.write_output_csv(decisions, str(output_path))

	completed = guardrails.ensure_full_coverage([getattr(msg, "message_id", None) for msg in messages], decisions)
	guardrails.write_output_csv(completed, str(output_path))
	print(f"Wrote {len(completed)} decisions to {output_path}", flush=True)

if __name__ == "__main__":
	main()

"""Utilities for captioning images and transcribing voice notes via Gemini."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

try:
    from .parser import ParsedMessage
except ImportError:  # pragma: no cover - fallback for direct execution
    from parser import ParsedMessage  # type: ignore


_IMAGE_FALLBACK = "[image could not be processed]"
_VOICE_FALLBACK = "[voice note could not be transcribed]"


def _resolve_media_path(media_id: str, index: dict[str, dict], key_name: str) -> str | None:
    """Return the on-disk file path for a media entry using the index's actual field names."""
    entry = index.get(media_id, {})
    if not isinstance(entry, dict):
        return None
    file_path = entry.get("file_path")
    if isinstance(file_path, str) and file_path:
        return file_path
    if isinstance(entry.get(key_name), str) and entry.get(key_name):
        return entry.get(key_name)
    return None


def _read_media_bytes(file_path: str) -> bytes:
    """Read a media file from disk and raise if it cannot be read."""
    return Path(file_path).read_bytes()


def caption_image(media_id: str, images_index: dict, api_key: str) -> str:
    """Return a factual image description using Gemini, or a fallback string on failure."""
    file_path = _resolve_media_path(media_id, images_index, "image_path")
    if not file_path:
        return _IMAGE_FALLBACK

    try:
        abs_path = Path(file_path)
        if not abs_path.is_absolute():
            abs_path = (Path.cwd() / abs_path).resolve()
        if not abs_path.exists():
            return _IMAGE_FALLBACK

        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        return _IMAGE_FALLBACK

    for attempt in range(1, 3):
        print(f"Processing image media_id={media_id} with Gemini attempt={attempt}", flush=True)
        try:
            image_bytes = _read_media_bytes(str(abs_path))
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="text/plain",
            )
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Describe this image factually and neutrally, with enough detail for a downstream classifier to judge whether it is a promotion, event notice, scam/phishing, payment request, or personal photo."},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
                        ],
                    }
                ],
                config=config,
            )
            return getattr(response, "text", str(response)).strip() or _IMAGE_FALLBACK
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429 or getattr(getattr(exc, "response", None), "status_code", None) == 429 or "429" in str(exc).lower() or "rate limit" in str(exc).lower() or "resource exhausted" in str(exc).lower():
                if attempt < 2:
                    time.sleep(2)
                    continue
            return _IMAGE_FALLBACK

    return _IMAGE_FALLBACK


def transcribe_voice_note(media_id: str, voice_index: dict, api_key: str) -> str:
    """Return a plain-text transcript of a voice note, or a fallback string on failure."""
    file_path = _resolve_media_path(media_id, voice_index, "voice_path")
    if not file_path:
        return _VOICE_FALLBACK

    try:
        abs_path = Path(file_path)
        if not abs_path.is_absolute():
            abs_path = (Path.cwd() / abs_path).resolve()
        if not abs_path.exists():
            return _VOICE_FALLBACK

        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception:
        return _VOICE_FALLBACK

    for attempt in range(1, 3):
        print(f"Processing voice media_id={media_id} with Gemini attempt={attempt}", flush=True)
        try:
            audio_bytes = _read_media_bytes(str(abs_path))
            client = genai.Client(api_key=api_key)
            config = types.GenerateContentConfig(
                temperature=0,
                response_mime_type="text/plain",
            )
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": "Transcribe the spoken content accurately in whatever language is spoken. Return plain text only."},
                            {"inline_data": {"mime_type": "audio/mpeg", "data": audio_bytes}},
                        ],
                    }
                ],
                config=config,
            )
            return getattr(response, "text", str(response)).strip() or _VOICE_FALLBACK
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429 or getattr(getattr(exc, "response", None), "status_code", None) == 429 or "429" in str(exc).lower() or "rate limit" in str(exc).lower() or "resource exhausted" in str(exc).lower():
                if attempt < 2:
                    time.sleep(2)
                    continue
            return _VOICE_FALLBACK

    return _VOICE_FALLBACK


def get_media_transcript(message: ParsedMessage, images_index: dict, voice_index: dict, api_key: str) -> str | None:
    """Dispatch to the correct media helper based on the message's media type."""
    media_type = getattr(message, "media_type", None)
    media_id = getattr(message, "media_id", None)

    if media_type == "image" and isinstance(media_id, str):
        return caption_image(media_id, images_index, api_key)
    if media_type == "voice" and isinstance(media_id, str):
        return transcribe_voice_note(media_id, voice_index, api_key)
    return None

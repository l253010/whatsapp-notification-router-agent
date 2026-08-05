import os
import sys
import types
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parser import ParsedMessage
from tools import caption_image, get_media_transcript, transcribe_voice_note


@dataclass
class FakeMessage:
    media_type: str | None = None
    media_id: str | None = None
    message_text: str | None = None


def test_caption_image_returns_fallback_for_missing_file():
    images_index = {"img_001": {"image_id": "img_001", "file_path": "media/images/does_not_exist.jpg"}}
    assert caption_image("img_001", images_index, api_key="fake") == "[image could not be processed]"


def test_transcribe_voice_note_returns_fallback_for_missing_file():
    voice_index = {"vn_001": {"voice_note_id": "vn_001", "file_path": "media/audio/does_not_exist.mp3"}}
    assert transcribe_voice_note("vn_001", voice_index, api_key="fake") == "[voice note could not be transcribed]"


def test_get_media_transcript_dispatches_to_image_and_voice():
    image_message = FakeMessage(media_type="image", media_id="img_001")
    voice_message = FakeMessage(media_type="voice", media_id="vn_001")

    assert get_media_transcript(image_message, {"img_001": {"file_path": "missing.jpg"}}, {}, "fake") == "[image could not be processed]"
    assert get_media_transcript(voice_message, {}, {"vn_001": {"file_path": "missing.mp3"}}, "fake") == "[voice note could not be transcribed]"
    assert get_media_transcript(FakeMessage(media_type="text", media_id=None), {}, {}, "fake") is None

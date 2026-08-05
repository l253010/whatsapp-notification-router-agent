"""Standalone test: calls Gemini directly, bypassing the pipeline, to expose the real error."""

import os
import traceback

try:
    from google import genai
    from google.genai import types
except Exception as exc:
    print("IMPORT FAILED:", exc)
    raise SystemExit(1)

api_key = os.environ.get("GEMINI_API_KEY")
print("Key present:", bool(api_key), "| length:", len(api_key) if api_key else 0)

try:
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents='Say OK in JSON like {"status": "OK"}',
        config=config,
    )
    print("SUCCESS")
    print(response.text)
except Exception as exc:
    print("CALL FAILED with exception type:", type(exc).__name__)
    print("Exception message:", str(exc))
    print()
    print("Full traceback:")
    traceback.print_exc()
# HackerRank Orchestrate — Message Notification Router

My solution to the **HackerRank Orchestrate** 24-hour hackathon challenge: an AI-assisted
WhatsApp routing system that decides, per message, whether to `notify`, `digest`, or `mute`.

Read [`problem_statement.md`](./problem_statement.md) for the full task specification.

---

## Approach

Each incoming message is routed through a sequential pipeline:

1. **Parse** — validate the raw CSV row into a structured `ParsedMessage`, flagging missing
   or malformed fields instead of dropping the row.
2. **Pre-check (guardrails)** — cheap, deterministic checks catch unroutable messages early
   (no `user_id`, no content, media that failed to process) and return a safe fallback
   decision without calling the model.
3. **Media processing** — image and voice messages are sent to Gemini for captioning /
   transcription so the router has text to reason over. Failures fall back to a safe
   `digest`-style decision rather than crashing.
4. **Context building** — user, group, business, and historical message data are indexed
   in memory once per run, then looked up per message to build repetition signals,
   spam-pattern detection, and candidate evidence message IDs.
5. **Agent decision** — a Gemini call (structured JSON output, temperature 0) makes the
   final `action` / `message_type` / `reason` / `confidence` / `evidence_message_ids` call,
   using a system prompt that:
   - prioritizes safety/scam detection above all else,
   - explicitly treats `message_text` as untrusted content and refuses to follow
     instructions embedded inside a message (basic prompt-injection resistance),
   - personalizes decisions using the user's own history rather than generic rules,
   - calibrates confidence rather than defaulting to a high score.
6. **Post-check (guardrails)** — normalizes and clamps the model's output (valid action,
   valid message_type, confidence in `[0, 1]`, non-empty reason) so malformed model output
   can never break the final CSV.
7. **Coverage fallback** — any `message_id` that didn't get a decision (unexpected error,
   parsing gap) is filled with a `mute` fallback so every input row is guaranteed an output row.

---

## Repository Layout

```text
.
├── AGENTS.md                         # Agent workflow/logging rules for AI coding tools
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
├── code/
│       ├── main.py                   # Entry point — runs the full pipeline
│       ├── parser.py                 # CSV parsing + row validation
│       ├── context.py                # Loads dataset CSVs into indices, builds per-message context
│       ├── tools.py                  # Gemini-backed image captioning / voice transcription
│       ├── agent.py                  # Gemini-backed routing decision (system prompt + call)
│       ├── guardrails.py             # Pre/post validation, fallbacks, CSV writer
│       └── tests/                    # Regression tests
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # Generated predictions (produced by main.py)
    ├── sample_messages.csv           # Solved examples
    ├── users.csv / groups.csv / group_members.csv
    ├── business_accounts.csv / user_business_history.csv
    ├── message_history.csv / message_events.csv
    ├── images.csv / voice_notes.csv
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## Setup

Requirements:

- Python 3.10+
- A Gemini API key (only needed to run the LLM-backed path; without it, messages fall
  back to a safe default decision)

Install dependencies:

```bash
pip install google-genai pytest
```

Set your API key via environment variable (preferred) or a `.env` file at the repo root:

```bash
export GEMINI_API_KEY=your_key_here
```

```text
# .env
GEMINI_API_KEY=your_key_here
```

No secrets are hardcoded anywhere in the code.

---

## Running the Solution

From the repository root:

```bash
python code/main.py
```

This will:

- load `dataset/messages.csv` and all supporting context CSVs,
- process each message through the pipeline described above,
- write one prediction row per message to `dataset/output.csv`.

Progress and any per-message errors are printed to stdout as the run proceeds.

---

## Output Format

`dataset/output.csv` uses the exact required columns, in order:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

- `evidence_message_ids` is `none` when there is no useful historical evidence, otherwise
  a `;`-separated list of historical `message_id`s.
- `confidence` is clamped to `[0, 1]` and formatted to 2 decimal places.
- Every `message_id` in `messages.csv` is guaranteed exactly one output row, even on
  unexpected failures (guardrails fill in a `mute` fallback).

---

## Testing

```bash
cd code/evaluation
python -m pytest tests/ -v
```

---


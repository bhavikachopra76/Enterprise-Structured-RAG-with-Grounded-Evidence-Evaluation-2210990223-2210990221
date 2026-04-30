"""
gemini_client.py
Centralised Gemini API wrapper for HierRAG — using Vertex AI.

Provides a thin abstraction over the Vertex AI Gemini SDK so every other
module in the project can call the LLM without worrying about auth,
retry logic, or rate limiting.

AUTHENTICATION:
    Uses Google Cloud Vertex AI instead of consumer API keys.
    Requires two environment variables in .env:

        GCP_PROJECT_ID=your-gcp-project-id
        GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/your/service-account.json

    The service account must have the "Vertex AI User" role (or equivalent).
    Vertex AI has enterprise-grade quotas managed at the GCP project level,
    so there's no API key rotation needed.

PUBLIC API:
    call_gemini(prompt, temperature)  → str
    call_gemini_json(prompt, temperature) → dict
    sleep_between_calls(seconds)

    All downstream modules (hierrag_pipeline, hierarchical_index, etc.)
    import these three functions. The rest is internal.

SECURITY:
    Never hardcode credentials in source. Always load from .env.
"""

import os
import time
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── configuration ──────────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION   = os.getenv("GCP_LOCATION", "global")
MODEL_NAME     = "gemini-3-flash-preview"


# ── custom exceptions (kept for backwards-compat with run_experiment.py) ──────
class KeysExhaustedError(Exception):
    """Legacy exception from the old multi-key rotation setup.

    With Vertex AI this should never fire under normal operation, but
    run_experiment.py still catches it, so we keep it around.
    """
    pass


# ── Vertex AI client ──────────────────────────────────────────────────────────
def _init_client() -> genai.Client:
    """Create a Vertex AI Gemini client using service-account credentials.

    The service-account JSON path is picked up automatically from the
    GOOGLE_APPLICATION_CREDENTIALS env var (standard GCP convention).
    This function runs once at module import time.
    """
    if not GCP_PROJECT_ID:
        raise EnvironmentError(
            "GCP_PROJECT_ID not set.  Add to .env:\n"
            "  GCP_PROJECT_ID=your-gcp-project-id\n"
            "  GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/cred.json"
        )

    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )
    print(f"  [gemini] Vertex AI client initialised  "
          f"(project={GCP_PROJECT_ID}, location={GCP_LOCATION}, "
          f"model={MODEL_NAME})")
    return client


# initialise once at import time — every subsequent call reuses this client
_client = _init_client()
_total_calls = 0          # simple counter for observability / debugging


# ── core call with retry ────────────────────────────────────────────────────────
def call_gemini(prompt: str, temperature: float = 0.0) -> str:
    """Send a prompt to Gemini and return the raw response text.

    Uses exponential backoff (2s, 4s, 8s) to ride out transient 429 / 503
    errors from the Vertex AI endpoint.

    Args:
        prompt:      full prompt string
        temperature: 0.0 for deterministic output (default)

    Returns:
        Response text as a plain string.

    Raises:
        RuntimeError: if all 3 retries are exhausted.
    """
    global _total_calls
    config = types.GenerateContentConfig(temperature=temperature)

    last_error = None
    for attempt in range(3):
        try:
            response = _client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            _total_calls += 1
            return response.text.strip()

        except Exception as e:
            last_error = e
            wait = 2 ** attempt   # 1s, 2s, 4s
            print(f"  [gemini] attempt {attempt + 1} failed: {e} "
                  f"— retrying in {wait}s")
            time.sleep(wait)

    raise RuntimeError(
        f"Gemini call failed after 3 retries: {last_error}"
    )


# ── json-specific call ──────────────────────────────────────────────────────────
def call_gemini_json(prompt: str, temperature: float = 0.0) -> dict:
    """Call Gemini and parse the response as JSON.

    Gemini sometimes wraps JSON output in markdown code fences (```json ... ```)
    even when told not to. This function strips those fences before parsing.

    Args:
        prompt:      prompt that instructs Gemini to reply in JSON only
        temperature: default 0.0

    Returns:
        Parsed dict from the JSON response.

    Raises:
        ValueError:  response cannot be parsed as valid JSON
        RuntimeError: all retries failed (bubbles up from call_gemini)
    """
    raw = call_gemini(prompt, temperature=temperature)

    # strip markdown code fences if Gemini wrapped the output in them
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        # drop the first (```json) and last (```) fence lines
        clean = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON.\nRaw response:\n{raw}\nError: {e}"
        )


# ── rate-limit helper ───────────────────────────────────────────────────────────
def sleep_between_calls(seconds: float = 10.0):
    """Sleep between back-to-back Gemini requests to stay under rate limits."""
    time.sleep(seconds)


# ── status helpers (simplified — no more multi-key tracking) ─────────────────
def key_status() -> dict:
    """Return current usage stats as a dict. Useful for logging / debugging."""
    return {
        "backend"     : "vertex_ai",
        "project"     : GCP_PROJECT_ID,
        "location"    : GCP_LOCATION,
        "model"       : MODEL_NAME,
        "total_calls" : _total_calls,
    }


def reset_key_counts():
    """Reset the call counter back to zero. Handy between experiment runs."""
    global _total_calls
    _total_calls = 0
    print("  [gemini] call counter reset to 0")


def print_key_status():
    """Print a one-liner with current backend info and total call count."""
    s = key_status()
    print(f"  [gemini] backend={s['backend']}  "
          f"project={s['project']}  model={s['model']}  "
          f"total_calls={s['total_calls']}")


# ── TEST ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("gemini_client.py — running tests (Vertex AI)")
    print("=" * 55)

    # test 1 — client init
    print("\nTest 1: Vertex AI client")
    status = key_status()
    print(f"  Backend : {status['backend']}")
    print(f"  Project : {status['project']}")
    print(f"  Model   : {status['model']}")
    assert status["backend"] == "vertex_ai"
    print("  PASS\n")

    # test 2 — basic call
    print("Test 2: basic text call")
    response = call_gemini("Reply with exactly the words: HierRAG online",
                           temperature=0.0)
    assert "HierRAG" in response, f"Unexpected response: {response}"
    print(f"  PASS — got: {response}")
    print_key_status()
    print()

    # test 3 — json call
    print("Test 3: JSON call")
    result = call_gemini_json(
        'Reply with JSON only, no markdown. '
        'Keys: "status" = "ok", "system" = "HierRAG"'
    )
    assert result.get("status") == "ok",       f"Bad JSON: {result}"
    assert result.get("system") == "HierRAG",  f"Bad JSON: {result}"
    print(f"  PASS — got: {result}")
    print_key_status()
    print()

    # test 4 — call counting
    print("Test 4: call counter increments")
    before = _total_calls
    call_gemini("Say: count test", temperature=0.0)
    after = _total_calls
    assert after == before + 1, f"Counter didn't increment: {before} → {after}"
    print(f"  PASS -- counter: {before} -> {after}\n")

    # test 5 — rate limit helper
    print("Test 5: sleep helper")
    sleep_between_calls(2.0)
    print("  PASS\n")

    print("=" * 55)
    print("Final status:")
    print_key_status()
    print("=" * 55)
    print("All tests passed.")
    print("=" * 55)
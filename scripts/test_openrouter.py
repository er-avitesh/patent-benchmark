"""
test_openrouter.py
------------------
Quick connection and vision test for OpenRouter Qwen2.5-VL-72B.
Tests both text-only and a real base64 image call using the first
PNG found in data/processed/ so we know it works before touching
the main pipeline.

Usage:
    python scripts/test_openrouter.py

Expects .env to have:
    OPENROUTER_API_KEY=sk-or-...
"""

import os
import sys
import base64
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT          = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL         = "google/gemma-4-31b-it"
BASE_URL      = "https://openrouter.ai/api/v1"

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def get_client():
    from openai import OpenAI
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        print("  [FAIL] OPENROUTER_API_KEY not set in .env")
        sys.exit(1)
    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/patent-benchmark",
            "X-Title": "Patent Benchmark",
        },
    )

# ── Test 1: text ping ─────────────────────────────────────────────────────────

def test_text_ping(client):
    section("Test 1 / 2  Text ping")
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: connected"}],
        )
        reply = resp.choices[0].message.content.strip().lower()
        model_used = resp.model
        print(f"  [PASS] reply={reply!r}  model={model_used}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

# ── Test 2: real base64 image call ────────────────────────────────────────────

def test_vision(client):
    section("Test 2 / 2  Vision call with real base64 PNG")

    # Find any PNG in data/processed/
    pngs = list(PROCESSED_DIR.glob("*.png"))
    if not pngs:
        print(f"  [SKIP] No PNGs found in {PROCESSED_DIR}")
        print(f"         Run 04_normalize.py first, or place any PNG there.")
        return False

    png_path = pngs[0]
    print(f"  Using: {png_path.name}")

    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  [PASS] Model responded to image input.")
        print(f"  Reply (first 150 chars): {reply[:150]!r}")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nOpenRouter — Qwen2.5-VL-72B Vision Test")
    print(f"Model : {MODEL}")
    print(f"URL   : {BASE_URL}")

    client = get_client()

    r1 = test_text_ping(client)
    r2 = test_vision(client)

    section("SUMMARY")
    print(f"  {'✓' if r1 else '✗'}  Text ping")
    print(f"  {'✓' if r2 else '✗'}  Vision (base64 PNG)")

    if r1 and r2:
        print("\n  All tests passed. Ready to update 07_run_models.py.\n")
        sys.exit(0)
    else:
        print("\n  Some tests failed. Check errors above before proceeding.\n")
        sys.exit(1)
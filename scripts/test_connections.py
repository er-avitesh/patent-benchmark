"""
test_connections.py
-------------------
Verifies API connectivity for all 4 models used in the patent benchmark.
Run this before any pilot or full benchmark run.

Usage:
    python test_connections.py

Expects .env in the same directory (or project root) with:
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AIza...
    TOGETHER_API_KEY=...
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

RESULTS = {}

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(label, note=""):
    msg = f"{label}" + (f" ({note})" if note else "")
    print(f"  [PASS] {msg}")
    RESULTS[label] = "PASS"

def fail(label, err):
    print(f"  [FAIL] {label}: {err}")
    RESULTS[label] = f"FAIL: {err}"

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

# ── 1. Anthropic (Claude Sonnet 4.6) ─────────────────────────────────────────

def test_anthropic():
    section("1 / 4  Anthropic — Claude Sonnet 4.6")
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        fail("claude-sonnet-4-6 reachable", "ANTHROPIC_API_KEY not set in .env")
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: connected"}],
        )
        reply = msg.content[0].text.strip().lower()
        ok("claude-sonnet-4-6 reachable", f"reply: {reply!r}")
    except Exception as e:
        fail("claude-sonnet-4-6 reachable", e)

# ── 2. OpenAI (GPT-5.5) ──────────────────────────────────────────────────────

def test_openai():
    section("2 / 4  OpenAI — GPT-5.5")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        fail("gpt-5.5 reachable", "OPENAI_API_KEY not set in .env")
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        # gpt-5.5 requires max_completion_tokens, not max_tokens
        resp = client.chat.completions.create(
            model="gpt-5.5",
            max_completion_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: connected"}],
        )
        reply = resp.choices[0].message.content.strip().lower()
        model_used = resp.model
        ok("gpt-5.5 reachable", f"model: {model_used}, reply: {reply!r}")
    except Exception as e:
        err_str = str(e)
        if "max_tokens" in err_str:
            fail("gpt-5.5 reachable", "Parameter error — should not occur with updated script")
        elif "not found" in err_str.lower() or "does not exist" in err_str.lower():
            fail("gpt-5.5 reachable", "Model not accessible on your tier — contact OpenAI or use gpt-4.1")
        else:
            fail("gpt-5.5 reachable", e)

# ── 3. Google (Gemini 2.5 Pro) ───────────────────────────────────────────────

def test_google():
    section("3 / 4  Google — Gemini 2.5 Pro")
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        fail("gemini-2.5-pro reachable", "GOOGLE_API_KEY not set in .env")
        return
    try:
        # Preferred: new google.genai SDK
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="gemini-2.5-pro",
            contents="Reply with the single word: connected",
        )
        reply = resp.text.strip().lower()
        ok("gemini-2.5-pro reachable", f"reply: {reply!r}")
    except ImportError:
        # Fallback: deprecated google.generativeai (suppress FutureWarning)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=key)
            model = genai_legacy.GenerativeModel("gemini-2.5-pro")
            resp = model.generate_content("Reply with the single word: connected")
            reply = resp.text.strip().lower()
            ok("gemini-2.5-pro reachable",
               f"reply: {reply!r} — upgrade: pip install google-genai")
        except Exception as e2:
            fail("gemini-2.5-pro reachable", e2)
    except Exception as e:
        fail("gemini-2.5-pro reachable", e)

# ── 4. Together AI (Qwen2.5-VL-72B) ─────────────────────────────────────────

def test_together():
    section("4 / 4  Together AI — Qwen3.5-9B")
    key = os.getenv("TOGETHER_API_KEY", "")
    if not key:
        fail("Qwen2.5-VL-72B reachable", "TOGETHER_API_KEY not set in .env")
        return
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=key,
            base_url="https://api.together.xyz/v1",
        )
        resp = client.chat.completions.create(
            model="Qwen/Qwen3.5-9B",
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: connected"}],
        )
        reply = resp.choices[0].message.content.strip().lower()
        model_used = resp.model
        ok("Qwen2.5-VL-72B reachable", f"model: {model_used}, reply: {reply!r}")
    except Exception as e:
        err_str = str(e)
        if "402" in err_str or "credit" in err_str.lower():
            fail("Qwen2.5-VL-72B reachable",
                 "Credit limit — payment may still be processing (wait 5 min and retry)")
        else:
            fail("Qwen2.5-VL-72B reachable", e)

# ── summary ───────────────────────────────────────────────────────────────────

def summary():
    section("SUMMARY")
    passed = sum(1 for v in RESULTS.values() if v == "PASS")
    total  = len(RESULTS)
    for label, status in RESULTS.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon}  {label}: {status}")
    print(f"\n  {passed}/{total} checks passed")
    if passed < total:
        print("  Fix failing connections before running the pilot.\n")
        sys.exit(1)
    else:
        print("  All connections OK. Ready to run module 7 pilot.\n")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nPatent Benchmark — API Connection Test")
    print(f"Python {sys.version.split()[0]}\n")

    #test_anthropic()
    time.sleep(0.5)
   # test_openai()
    time.sleep(0.5)
   # test_google()
    time.sleep(0.5)
    test_together()

    summary()
"""
07_run_models.py
----------------
Runs multimodal LLM inference for the patent infringement benchmark.

Modes:
    python scripts/07_run_models.py --pilot
        20 queries x 4 models x 1 strategy x 1 rep = 80 calls (binary only)

    python scripts/07_run_models.py --full
        192 queries x 4 models x 3 strategies x 3 reps = 6,912 calls

    python scripts/07_run_models.py --pilot --model claude
        Run pilot for a single model only

    python scripts/07_run_models.py --full --model gemini
        Run full study for a single model only

Resume behaviour:
    Every completed job is immediately saved to results/raw_responses/<job_id>.json
    On any restart the script skips jobs that already have a file there.
    You can kill the process at any time and rerun — no work is lost.

Job ID format:
    {query_id}__{model}__{strategy}__rep{rep}
    e.g.  q0042__claude__zero_shot__rep1

Model notes:
    - GPT-5.5  requires max_completion_tokens (not max_tokens)
    - Gemini    uses deprecated google.generativeai (warning suppressed)
    - Qwen3.5-9B thinking mode disabled via extra_body to avoid token waste
"""

import os
import sys
import json
import time
import base64
import argparse
import random
import logging
import warnings
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

# ── configuration ─────────────────────────────────────────────────────────────

load_dotenv()

ROOT          = Path(__file__).resolve().parent.parent
CANDIDATES    = ROOT / "candidates" / "candidates.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR       = ROOT / "results" / "raw_responses"
RAW_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = ROOT / "results" / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42

# Model identifiers — internal key → API model string
MODELS = {
    "claude": "claude-sonnet-4-6",
    "openai": "gpt-5.5",
    "gemini": "gemini-2.5-pro",
    "gemma":  "google/gemma-4-31b-it",           # Gemma 4 31B (Google DeepMind, Apache 2.0) via OpenRouter; confirmed base64 vision
}

STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought"]

PILOT_QUERIES = 20    # 10 positive + 10 negative
PILOT_REPS    = 1
PILOT_STRATS  = ["zero_shot"]

FULL_REPS     = 3
CANDIDATES_PER_QUERY = 5

# Retry settings
MAX_RETRIES  = 4
RETRY_BACKOFF = [5, 15, 45, 120]    # seconds between attempts

# Courtesy delay between calls (per model) to avoid rate-limit bursts
CALL_DELAY = {
    "claude": 0.5,
    "openai": 0.5,
    "gemini": 1.0,
    "gemma":  1.0,
}

# ── prompts ───────────────────────────────────────────────────────────────────

PROMPT_ZERO_SHOT = """\
You are evaluating potential design patent infringement under the ordinary \
observer test (Egyptian Goddess v. Swisa, 2008).

An ordinary observer is a typical purchaser of the product, not a patent expert. \
The question is: would an ordinary purchaser be deceived into thinking \
a candidate design is the same as the query design?

The query design is shown first. Then 5 candidate designs follow.

For each candidate, give a binary verdict:
  INFRINGE     -- an ordinary observer would likely be deceived
  NO_INFRINGE  -- an ordinary observer would not be deceived

Respond ONLY with a JSON object in this exact format, no explanation:
{
  "candidate_1": "INFRINGE" or "NO_INFRINGE",
  "candidate_2": "INFRINGE" or "NO_INFRINGE",
  "candidate_3": "INFRINGE" or "NO_INFRINGE",
  "candidate_4": "INFRINGE" or "NO_INFRINGE",
  "candidate_5": "INFRINGE" or "NO_INFRINGE"
}"""

PROMPT_FEW_SHOT = """\
You are evaluating potential design patent infringement under the ordinary \
observer test (Egyptian Goddess v. Swisa, 2008).

An ordinary observer is a typical purchaser of the product, not a patent expert. \
The question is: would an ordinary purchaser be deceived into thinking \
a candidate design is the same as the query design?

EXAMPLE (illustrative only — not real patents):
  - A chair with identical leg curvature, seat profile, and back shape → INFRINGE
  - A chair with different leg style and rounded back where query has angular → NO_INFRINGE

The query design is shown first. Then 5 candidate designs follow.

For each candidate, give a binary verdict:
  INFRINGE     -- an ordinary observer would likely be deceived
  NO_INFRINGE  -- an ordinary observer would not be deceived

Respond ONLY with a JSON object in this exact format, no explanation:
{
  "candidate_1": "INFRINGE" or "NO_INFRINGE",
  "candidate_2": "INFRINGE" or "NO_INFRINGE",
  "candidate_3": "INFRINGE" or "NO_INFRINGE",
  "candidate_4": "INFRINGE" or "NO_INFRINGE",
  "candidate_5": "INFRINGE" or "NO_INFRINGE"
}"""

PROMPT_CHAIN_OF_THOUGHT = """\
You are evaluating potential design patent infringement under the ordinary \
observer test (Egyptian Goddess v. Swisa, 2008).

An ordinary observer is a typical purchaser of the product, not a patent expert. \
The question is: would an ordinary purchaser be deceived into thinking \
a candidate design is the same as the query design?

The query design is shown first. Then 5 candidate designs follow.

For each candidate:
  1. Briefly note the dominant visual features of the query (1 sentence).
  2. Briefly note how the candidate differs or matches (1 sentence).
  3. Give your verdict: INFRINGE or NO_INFRINGE.

Then end with a JSON summary containing ONLY the verdicts:
{
  "candidate_1": "INFRINGE" or "NO_INFRINGE",
  "candidate_2": "INFRINGE" or "NO_INFRINGE",
  "candidate_3": "INFRINGE" or "NO_INFRINGE",
  "candidate_4": "INFRINGE" or "NO_INFRINGE",
  "candidate_5": "INFRINGE" or "NO_INFRINGE"
}"""

PROMPTS = {
    "zero_shot":        PROMPT_ZERO_SHOT,
    "few_shot":         PROMPT_FEW_SHOT,
    "chain_of_thought": PROMPT_CHAIN_OF_THOUGHT,
}

# ── image utilities ───────────────────────────────────────────────────────────

def load_image_b64(png_path: Path) -> str:
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def resolve_png(relative_path: str) -> Path:
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return PROCESSED_DIR / p.name

# ── job management ────────────────────────────────────────────────────────────

def job_id(query_id: str, model_key: str, strategy: str, rep: int) -> str:
    return f"{query_id}__{model_key}__{strategy}__rep{rep}"

def job_done(jid: str) -> bool:
    return (RAW_DIR / f"{jid}.json").exists()

def save_job(jid: str, payload: dict):
    """Atomic write: temp file then rename."""
    path = RAW_DIR / f"{jid}.json"
    tmp  = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.rename(path)

def build_jobs(df: pd.DataFrame, model_keys: list,
               strategies: list, reps: int) -> list:
    """Group 5 candidates per query and enumerate all (query, model, strategy, rep) combos."""
    jobs = []
    for qid, group in df.groupby("query_id"):
        group = group.sort_values("candidate_rank")
        if len(group) != CANDIDATES_PER_QUERY:
            log.warning(f"Query {qid} has {len(group)} candidates, expected "
                        f"{CANDIDATES_PER_QUERY} — skipping")
            continue
        for model_key in model_keys:
            for strategy in strategies:
                for rep in range(1, reps + 1):
                    jid = job_id(qid, model_key, strategy, rep)
                    jobs.append({
                        "jid":         jid,
                        "query_id":    qid,
                        "model_key":   model_key,
                        "strategy":    strategy,
                        "rep":         rep,
                        "query_png":   resolve_png(group.iloc[0]["query_png"]),
                        "candidates":  [
                            {
                                "rank":            int(row["candidate_rank"]),
                                "candidate_id":    row["candidate_id"],
                                "candidate_label": row["candidate_label"],
                                "candidate_png":   resolve_png(row["candidate_png"]),
                            }
                            for _, row in group.iterrows()
                        ],
                        "query_label": group.iloc[0]["query_label"],
                        "query_uspc":  group.iloc[0]["query_uspc"],
                    })
    return jobs

def select_pilot_queries(df: pd.DataFrame,
                         n: int = PILOT_QUERIES) -> pd.DataFrame:
    """Stratified sample: n/2 positive + n/2 negative, spread across USPC classes."""
    rng = random.Random(RANDOM_SEED)
    pos = df[df["query_label"] == "positive"]["query_id"].unique().tolist()
    neg = df[df["query_label"] == "negative"]["query_id"].unique().tolist()
    rng.shuffle(pos)
    rng.shuffle(neg)
    selected = pos[: n // 2] + neg[: n // 2]
    return df[df["query_id"].isin(selected)]

# ── model callers ─────────────────────────────────────────────────────────────

def call_claude(prompt: str, query_b64: str, candidate_b64s: list) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    content = [{"type": "text", "text": "Query design:"}]
    content.append({"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png",
                                "data": query_b64}})
    for i, b64 in enumerate(candidate_b64s, 1):
        content.append({"type": "text", "text": f"Candidate {i}:"})
        content.append({"type": "image",
                         "source": {"type": "base64",
                                    "media_type": "image/png",
                                    "data": b64}})
    content.append({"type": "text", "text": prompt})

    resp = client.messages.create(
        model=MODELS["claude"],
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


def call_openai(prompt: str, query_b64: str, candidate_b64s: list) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    content = [{"type": "text", "text": "Query design:"},
               {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{query_b64}"}}]
    for i, b64 in enumerate(candidate_b64s, 1):
        content.append({"type": "text", "text": f"Candidate {i}:"})
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model=MODELS["openai"],
        max_completion_tokens=2048,         # gpt-5.5 requires this, not max_tokens; 512 too small for image-heavy prompts
        messages=[{"role": "user", "content": content}],
    )
    choice = resp.choices[0]
    finish = choice.finish_reason
    text   = choice.message.content or ""
    if not text.strip():
        raise RuntimeError(
            f"GPT-5.5 returned empty response (finish_reason={finish!r}). "
            f"Possible content filter or model refusal on this image set."
        )
    return text


def call_gemini(prompt: str, query_b64: str, candidate_b64s: list) -> str:
    """
    Gemini 2.5 Pro via new google.genai SDK.
    Thinking mode budget set to 0 — prevents hidden reasoning tokens from
    consuming output budget before the JSON verdict is emitted.
    Falls back to deprecated google.generativeai if google.genai not installed.
    """
    try:
        from google import genai as genai_new
        from google.genai import types as genai_types

        client = genai_new.Client(api_key=os.environ["GOOGLE_API_KEY"])

        def b64_to_part_new(b64: str):
            return genai_types.Part.from_bytes(
                data=base64.b64decode(b64),
                mime_type="image/png",
            )

        parts = [
            genai_types.Part.from_text(text="Query design:"),
            b64_to_part_new(query_b64),
        ]
        for i, b64 in enumerate(candidate_b64s, 1):
            parts.append(genai_types.Part.from_text(text=f"Candidate {i}:"))
            parts.append(b64_to_part_new(b64))
        parts.append(genai_types.Part.from_text(text=prompt))

        resp = client.models.generate_content(
            model=MODELS["gemini"],
            contents=parts,
            config=genai_types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=3000,  # thinking tokens + response tokens
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=512,  # minimum allowed; model requires thinking mode
                ),
            ),
        )
        # Check for truncation
        if resp.candidates and resp.candidates[0].finish_reason.name == "MAX_TOKENS":
            raise RuntimeError("Gemini finish_reason=MAX_TOKENS — response truncated")
        return resp.text

    except ImportError:
        # Fallback: deprecated google.generativeai
        import io
        import PIL.Image
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import google.generativeai as genai_legacy

        genai_legacy.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai_legacy.GenerativeModel(MODELS["gemini"])

        def b64_to_pil(b64: str) -> PIL.Image.Image:
            return PIL.Image.open(io.BytesIO(base64.b64decode(b64)))

        parts = ["Query design:", b64_to_pil(query_b64)]
        for i, b64 in enumerate(candidate_b64s, 1):
            parts.append(f"Candidate {i}:")
            parts.append(b64_to_pil(b64))
        parts.append(prompt)

        resp = model.generate_content(
            parts,
            generation_config={"temperature": 0, "max_output_tokens": 2048},
        )
        if resp.candidates[0].finish_reason == 2:
            raise RuntimeError("Gemini finish_reason=MAX_TOKENS — response truncated")
        return resp.text


def call_gemma_openrouter(prompt: str, query_b64: str, candidate_b64s: list) -> str:
    """
    Gemma 4 31B (Google DeepMind, Apache 2.0) via OpenRouter.
    Confirmed working with base64 PNG data URIs (tested May 2026).
    Used as open-weight baseline after Qwen/Llama removed from serverless tiers.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/patent-benchmark",
            "X-Title": "Patent Benchmark",
        },
    )

    content = [
        {"type": "text", "text": "Query design:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{query_b64}"}},
    ]
    for i, b64 in enumerate(candidate_b64s, 1):
        content.append({"type": "text", "text": f"Candidate {i}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model=MODELS["gemma"],
        max_tokens=1024,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


MODEL_CALLERS = {
    "claude": call_claude,
    "openai": call_openai,
    "gemini": call_gemini,
    "gemma":  call_gemma_openrouter,
}

# ── response parsing ──────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """
    Extract the JSON verdict block from model response.
    Handles:
      - Pure JSON responses (zero_shot)
      - JSON wrapped in markdown fences (```json ... ```) — GPT-5.5 style
      - JSON embedded in CoT reasoning text (chain_of_thought)
      - Thinking-mode residue like <think>…</think> before JSON — Qwen style
    """
    import re

    if not text:
        return {}

    # Strip any <think>…</think> block (Qwen thinking mode residue)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # Try to find the JSON verdict object anywhere in the text
    match = re.search(
        r'\{[^{}]*"candidate_[12345]"[^{}]*\}', text, re.DOTALL
    )
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try the whole cleaned text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}

# ── core runner ───────────────────────────────────────────────────────────────

def run_job(job: dict) -> dict:
    """Execute one API call with retries. Returns a result dict."""
    jid       = job["jid"]
    model_key = job["model_key"]
    strategy  = job["strategy"]
    prompt    = PROMPTS[strategy]
    caller    = MODEL_CALLERS[model_key]

    query_b64      = load_image_b64(job["query_png"])
    candidate_b64s = [load_image_b64(c["candidate_png"])
                      for c in job["candidates"]]

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0       = time.time()
            raw_text = caller(prompt, query_b64, candidate_b64s)
            elapsed  = round(time.time() - t0, 2)
            verdicts = extract_json(raw_text)

            return {
                "job_id":         jid,
                "query_id":       job["query_id"],
                "model_key":      model_key,
                "model_string":   MODELS[model_key],
                "strategy":       strategy,
                "rep":            job["rep"],
                "query_label":    job["query_label"],
                "query_uspc":     job["query_uspc"],
                "candidates": [
                    {
                        "rank":            c["rank"],
                        "candidate_id":    c["candidate_id"],
                        "candidate_label": c["candidate_label"],
                    }
                    for c in job["candidates"]
                ],
                "raw_response":   raw_text,
                "verdicts":       verdicts,
                "verdict_count":  len(verdicts),
                "parse_ok":       len(verdicts) == CANDIDATES_PER_QUERY,
                "elapsed_sec":    elapsed,
                "attempt":        attempt,
                "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
                "error":          None,
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt - 1]
                log.warning(f"  [{jid}] attempt {attempt} failed: {e} "
                            f"— retrying in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"  [{jid}] all {MAX_RETRIES} attempts failed: {e}")

    # All retries exhausted — save failure record so resume skips it cleanly.
    # Delete the file manually to force a retry on a specific job.
    return {
        "job_id":        jid,
        "query_id":      job["query_id"],
        "model_key":     model_key,
        "model_string":  MODELS[model_key],
        "strategy":      strategy,
        "rep":           job["rep"],
        "query_label":   job["query_label"],
        "query_uspc":    job["query_uspc"],
        "raw_response":  None,
        "verdicts":      {},
        "verdict_count": 0,
        "parse_ok":      False,
        "elapsed_sec":   None,
        "attempt":       MAX_RETRIES,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "error":         last_error,
    }


def run_jobs(jobs: list):
    """Main loop: skip done jobs, run pending, save each result immediately."""
    total   = len(jobs)
    pending = [j for j in jobs if not job_done(j["jid"])]
    skipped = total - len(pending)

    log.info(f"Total jobs           : {total}")
    log.info(f"Already done (skip)  : {skipped}")
    log.info(f"Pending              : {len(pending)}")

    if not pending:
        log.info("Nothing to do — all jobs complete.")
        return

    errors  = 0
    success = 0

    for i, job in enumerate(pending, 1):
        jid = job["jid"]
        log.info(f"[{i}/{len(pending)}] {jid}")

        result = run_job(job)
        save_job(jid, result)

        if result["error"]:
            errors += 1
            log.error(f"  FAILED — {result['error']}")
        else:
            success += 1
            log.info(f"  OK  elapsed={result['elapsed_sec']}s  "
                     f"parse_ok={result['parse_ok']}  "
                     f"verdicts={result['verdict_count']}")

        time.sleep(CALL_DELAY.get(job["model_key"], 0.5))

    log.info(f"\nRun complete.  success={success}  errors={errors}  "
             f"total_run={len(pending)}")
    if errors:
        log.warning(f"{errors} jobs failed — inspect error fields in "
                    f"results/raw_responses/ and delete those files to retry.")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Patent benchmark model runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/07_run_models.py --pilot
  python scripts/07_run_models.py --pilot --model claude
  python scripts/07_run_models.py --pilot --dry-run
  python scripts/07_run_models.py --full
  python scripts/07_run_models.py --full --model gemini
        """,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true",
                      help="20-query pilot, zero_shot only, 1 rep (80 calls)")
    mode.add_argument("--full",  action="store_true",
                      help="Full study: 192 queries, 3 strategies, 3 reps (6,912 calls)")

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Restrict run to one model (default: all 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print job list without making any API calls",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("=" * 60)
    log.info(f"Patent benchmark runner  —  {'PILOT' if args.pilot else 'FULL'}")
    log.info(f"Model filter : {args.model or 'all'}")
    log.info(f"Dry run      : {args.dry_run}")
    log.info("=" * 60)

    if not CANDIDATES.exists():
        log.error(f"candidates.csv not found at {CANDIDATES}")
        sys.exit(1)

    df = pd.read_csv(CANDIDATES)
    log.info(f"Loaded {len(df)} candidate rows, "
             f"{df['query_id'].nunique()} unique queries")

    if args.pilot:
        df = select_pilot_queries(df)
        n_pos = df[df["query_label"] == "positive"]["query_id"].nunique()
        n_neg = df[df["query_label"] == "negative"]["query_id"].nunique()
        log.info(f"Pilot: {df['query_id'].nunique()} queries "
                 f"({n_pos} positive, {n_neg} negative)")

    model_keys = [args.model] if args.model else list(MODELS.keys())
    strategies = PILOT_STRATS if args.pilot else STRATEGIES
    reps       = PILOT_REPS   if args.pilot else FULL_REPS

    log.info(f"Models     : {model_keys}")
    log.info(f"Strategies : {strategies}")
    log.info(f"Reps       : {reps}")

    jobs = build_jobs(df, model_keys, strategies, reps)
    log.info(f"Jobs built : {len(jobs)}")

    if args.dry_run:
        log.info("DRY RUN — first 10 job IDs:")
        for j in jobs[:10]:
            done = "✓" if job_done(j["jid"]) else "·"
            log.info(f"  [{done}] {j['jid']}")
        log.info("Exiting (dry run).")
        return

    run_jobs(jobs)


if __name__ == "__main__":
    main()
# Patent Benchmark -- Claude Handoff Document

This document brings a new Claude instance fully up to speed on an ongoing
academic research project. Read it completely before responding to any requests.

---

## Project identity

**Title:** Can LLMs Check If Your Design Infringes on Patents?
**Type:** Academic benchmark paper
**Authors:** Avitesh Kesharwani, M.S. (IEEE Senior Member, Genpact) + Bjarne Berg, Ph.D. (UNC Charlotte, Belk College of Business)
**Companion paper:** "Can LLMs Check Your Architecture Decisions?" (Kesharwani & Berg, 2026) -- best macro-F1=0.48, best kappa=0.26 across 100 ADRs

**Research question:** Can multimodal LLMs reliably detect US design patent infringement under the ordinary observer test (Egyptian Goddess v. Swisa, Federal Circuit 2008)?

**Legal standard used:** Ordinary observer test -- would an ordinary purchaser be deceived into thinking one design is the same as another?

---

## Repository location

```
C:\Users\avitesh.kesharvani\LearningGround\patent-benchmark\
```

Python venv at `.venv\`. Always activate before running scripts:
```powershell
.venv\Scripts\activate
```

---

## Dataset -- COMPLETE

### Final composition (per Prof. Berg stratification feedback)

| Class | Count | Source | Label |
|---|---|---|---|
| Positive (in-force patents) | 96 | USPTO, granted 2015-2024 | positive |
| Negative expired (public domain) | 48 | USPTO, granted pre-2008, past term | negative |
| Negative open-source | 48 | Wikimedia Commons, CC0/public domain | negative |
| **Total** | **192** | | 96 positive / 96 negative |

**Stratification:** negative class is exactly 50/50 expired vs open-source,
6 per USPC class each. Requested by Prof. Berg to enable false positive vs
false negative analysis by source type -- key new dimension for top journal.

### USPC classes covered (all 8)

| Class | Domain | Locarno |
|---|---|---|
| D6 | Furniture | 06 |
| D8 | Tools and hardware | 08 |
| D9 | Packaging and containers | 09 |
| D12 | Transportation | 12 |
| D14 | Electronics housings | 14 |
| D23 | Fluid handling | 23 |
| D24 | Medical instruments | 24 (client anchor class) |
| D26 | Lighting | 26 |

### Processed images

All 192 drawings normalized to 1024x1024 grayscale PNG in `data/processed/`.

**PDF processing notes (important for methods section):**
- USPTO design patent PDFs are pure raster scans -- get_text() returns nothing
- Page detection uses blob analysis (numpy): renders each page at 72 DPI,
  measures bounding-box coverage of ink. Drawing pages score 0.25-0.55,
  reference pages 0.02-0.08
- Tesseract OCR masks patent numbers and sheet labels (conf >= 70, metadata
  patterns only)
- This approach was discovered through debugging, not from documentation

---

## Pipeline status -- MODULES 0-6 COMPLETE

| Module | Script | Status | Output |
|---|---|---|---|
| 0 | 00_verify_env.py | done | environment check |
| 1 | 01_pull_uspto.py | done | 96 PDFs in data/raw/positive/ |
| 2 | 02_pull_negative_expired.py | done | 72 PDFs in data/raw/negative_expired/ |
| 3 | 03_collect_negative_opensource.py | done | 48 drawings in data/raw/negative_opensource/ |
| 4 | 04_normalize.py | done | 192 PNGs in data/processed/ |
| 5 | 05_build_manifest.py | done | data/manifest.csv, 192 rows |
| 6 | 06_select_candidates.py | done | candidates/candidates.csv, 960 pairs |
| 7 | 07_run_models.py | NOT WRITTEN | next task |
| 8 | 08_parse_responses.py | not written | after module 7 |
| 9 | 09_compute_stats.py | not written | after module 8 |

---

## What needs to be done next (in order)

### Step 1 -- Add API keys to .env

The .env file needs these four new keys (not yet obtained):

```
USPTO_API_KEY=...            # already have this one
ANTHROPIC_API_KEY=sk-ant-... # console.anthropic.com -> API Keys
GOOGLE_API_KEY=AIza...       # aistudio.google.com -> Get API key
TOGETHER_API_KEY=...         # together.ai -> Settings -> API Keys
OPENAI_API_KEY=sk-...        # platform.openai.com -> API Keys
```

Qwen2.5-VL-7B is hosted via Together AI (OpenAI-compatible API, not local).
Together AI gives $25 free credit on signup -- enough for the full pilot.

### Step 2 -- Update 00_verify_env.py to check new API keys

The existing verify script only checks USPTO_API_KEY. It needs to check all
four model API keys before the pilot run.

### Step 3 -- Write and run module 7 (pilot first, then full)

**Prof. Berg's instruction:** run a pilot of 20 queries with binary verdict
before committing to the full multi-rubric run.

**Pilot spec:**
- 20 queries (stratified: 10 positive + 10 negative, spread across classes)
- 4 models: GPT-5.5, Claude Sonnet 4.6, Gemini 2.5 Pro, Qwen2.5-VL-7B
- Binary verdict only: INFRINGE or NO_INFRINGE
- 1 repetition, temperature=0
- Total: 20 x 4 x 1 = 80 API calls

**Full run spec (after pilot approval):**
- 192 queries x 4 models x 3 strategies x 3 reps = 6,912 calls
- Strategies: zero_shot, few_shot, chain_of_thought
- Temperature: 0
- Cost: ~$354 batch API / ~$614 standard

**Model API details:**

| Model | Provider | API style | Model string |
|---|---|---|---|
| GPT-5.5 | OpenAI | OpenAI SDK | gpt-5.5 |
| Claude Sonnet 4.6 | Anthropic | Anthropic SDK | claude-sonnet-4-6 |
| Gemini 2.5 Pro | Google | Google GenAI SDK | gemini-2.5-pro |
| Qwen2.5-VL-7B | Together AI | OpenAI-compatible | Qwen/Qwen2.5-VL-7B-Instruct |

Together AI base URL: https://api.together.xyz/v1

**Binary verdict prompt (use exactly this):**

```
You are evaluating potential design patent infringement under the ordinary
observer test (Egyptian Goddess v. Swisa, 2008).

An ordinary observer is a typical purchaser of the product, not a patent expert.
The question is: would an ordinary purchaser be deceived into thinking
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
}
```

### Step 4 -- Write module 8 (parse responses)

Extract verdicts from raw JSON responses into results/parsed.csv.
Handle malformed JSON, partial responses, refusals.

### Step 5 -- Write module 9 (statistics)

Per-model, per-strategy metrics:
- Macro-F1
- Cohen kappa
- McNemar test for pairwise model comparisons
- Confusion matrix breakdown by source_type (expired vs open-source negatives)
  -- this is the new dimension Prof. Berg wants for top journal submission

---

## Key confirmed API details

**USPTO Patent File Wrapper API (confirmed working May 2026):**
- Host: https://api.uspto.gov
- Endpoint: POST /api/v1/patent/applications/search
- Auth: X-API-KEY header
- Query syntax: filters + rangeFilters arrays (NOT legacy _and/_eq)
- Response key: patentFileWrapperDataBag
- Rate limit: 45 req/min
- Class field: applicationMetaData.class (values: "D8" not "D08")
- Patent number: applicationMetaData.patentNumber (e.g. "D1049406")

**Patent PDF download (no auth):**
https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/D1049406

**Wikimedia Commons API:**
- https://commons.wikimedia.org/w/api.php
- Requires descriptive User-Agent header on ALL requests (missing = 403)
- Use categorymembers action to list files in a category
- Use imageinfo action with iiprop=url|size|mime|extmetadata for download URLs

---

## config.yaml key values

```yaml
study:
  total_samples: 192
  candidates_per_query: 5
  repetitions: 3
  temperature: 0
  random_seed: 42

date_windows:
  positive_grant_start: "2015-01-01"
  positive_grant_end: "2024-12-31"
  expired_grant_end: "2007-12-31"

negative_composition:
  expired_patent_count: 48    # 50% of negative class, 6 per class
  open_source_count: 48       # 50% of negative class, 6 per class
```

---

## Prof. Berg email thread summary

1. Stratified negative class -- done. 50/50 expired vs open-source.
2. Pilot study first -- agreed, 20 queries binary verdict.
3. Check literature on infringement classification before finalizing rubric.
4. Budget -- $800 ceiling confirmed.
5. Bank transfer request received -- redirected to official university channels.
   Do not process any informal payment requests via email from Prof. Berg.

---

## Common gotchas discovered during build

1. USPTO API returns patentFileWrapperDataBag not patents or results
2. USPC class values are "D8" not "D08" -- zero-padding breaks queries
3. USPTO PDFs are pure raster -- get_text() always returns empty string
4. Wikimedia 403 = missing User-Agent (required on API calls AND downloads)
5. Windows rejects " < > ? * : | \ / in filenames -- strip from Wikimedia titles
6. PIL DecompressionBomb at 300 DPI -- use 150 DPI, raise MAX_IMAGE_PIXELS
7. cairosvg on Windows needs GTK3 runtime (libcairo-2.dll) installed separately
   GTK3 installer: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
8. Tesseract must be in PATH each session:
   $env:PATH += ";C:\Users\avitesh.kesharvani\AppData\Local\Programs\Tesseract-OCR"

---

## candidates.csv structure (960 rows)

Columns: query_id, query_label, query_uspc, query_png,
         candidate_id, candidate_label, candidate_uspc, candidate_png, candidate_rank

candidate_label is ground truth for scoring -- NOT shown to models.
Models see images only; they do not know which candidates are positive or negative.

---

## Cost estimate (confirmed May 2026 pricing)

Per-call token estimate (1 query image + 5 candidate images):
- Input: ~6,800 tokens
- Output: ~400 tokens

| Model | Standard | Batch API |
|---|---|---|
| GPT-5.5 | $331 | $166 |
| Claude Sonnet 4.6 | $190 | $95 |
| Gemini 2.5 Pro | $90 | $90 |
| Qwen2.5-VL-7B (Together) | ~$3 | ~$3 |
| Total | ~$614 | ~$354 |

Use Batch API for GPT-5.5 and Claude (50% discount, 24hr turnaround).
Budget ceiling: $800. Well within budget.

---

## Qwen model background

Qwen2.5-VL-7B is Alibaba's open-weight vision-language model (7B parameters).
Open-weight means the weights are public -- anyone can run it. We use Together
AI's hosted version to avoid local GPU setup. It serves as the open-weight
baseline -- the other three models are closed commercial APIs. Including it lets
the paper address whether open-weight models can perform legal visual reasoning.

---

## Paper framing

First benchmark to evaluate multimodal LLMs on the FIG.1 perspective drawing
comparison task under Egyptian Goddess. Benchmarks the judgment step only, not
retrieval.

Key contribution over the companion ADR paper:
- Visual modality (drawings, not text)
- Precisely defined legal standard (ordinary observer test)
- Stratified negative class enables error analysis by type
- 4 models including open-weight baseline

Expected hypothesis: models perform above chance but below human expert level,
with significant variation across strategies and model size. The expired vs
open-source breakdown will reveal whether errors cluster around visual similarity
(expired USPTO drawings look like positives) vs visual dissimilarity (Wikimedia
drawings look obviously different).
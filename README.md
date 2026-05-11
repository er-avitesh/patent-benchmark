# Patent Infringement Benchmark

Benchmarking multimodal LLMs on US design-patent infringement detection

Authors: Avitesh Kesharwani, M.S., IEEE Senior Member / Bjarne Berg, Ph.D., UNC Charlotte

---

## Progress

```
Dataset
  Positive drawings (in-force patents)    96 /  96   [====================] 100%
  Negative expired  (public domain)       48 /  48   [====================] 100%
  Negative open-source (generated)        48 /  48   [====================] 100%
  Total                                  192 / 192   [====================] 100%
  Processed PNGs                         192 / 192   [====================] 100%
  manifest.csv                           192 / 192   [====================] 100%
  candidates.csv                         960 pairs   [====================] 100%

Negative class stratification (per Berg feedback)
  Expired patents   48 / 96   50%   6 per class
  Open-source       48 / 96   50%   6 per class

Pipeline
  [x] Module 0   Environment check
  [x] Module 1   Positive-class puller         96 PDFs, 8 USPC classes, 2015-2024
  [x] Module 2   Negative expired puller       72 PDFs pulled, 48 used (6/class)
  [x] Module 3   Negative open-source          48 drawings, PIL-generated line drawings
  [x] Module 3b  Negative open-source fix      replaced bad Wikimedia files, PIL generation
  [x] Module 4   Normalize                     192 PNGs, blob detection, Tesseract masking
  [x] Module 5   Build manifest                192 rows, 96 positive / 96 negative, Locarno coded
  [x] Module 6   Candidate selection           960 pairs, 5 per query, title-keyword matching
  [x] Module 7   Model evaluation (pilot)      80 calls complete, 4 models, 20 queries
  [ ] Module 7   Model evaluation (full run)   6,912 calls pending
  [ ] Module 8   Parse responses               parsed.csv
  [ ] Module 9   Statistics                    macro-F1, Cohen kappa, McNemar
```

---

## Pilot results (May 2026)

20 queries x 4 models x 1 strategy (zero_shot) x 1 rep = 80 calls.
All 80 completed with 100% parse rate.

```
Model               parse_ok    INFRINGE    NO_INFRINGE
Claude Sonnet 4.6   20/20       0 (0%)      100 (100%)
GPT-5.5             20/20       1 (1%)      99  (99%)
Gemini 2.5 Pro      20/20       2 (2%)      98  (98%)
Gemma 4 31B         20/20       1 (1%)      99  (99%)
```

Key observation: models exhibit strong conservative bias under the ordinary
observer test, defaulting to NO_INFRINGE in 96-100% of cases even for
within-category title-matched pairs. Three models independently flagged the
same pair (D1051771 vs D1055773, D12 motor vehicle bodies) as INFRINGE.
This inter-model agreement on a true positive pair is the clearest signal
from the pilot.

---

## Dataset notes

### Negative open-source (module 3 / 3b)

Original Wikimedia Commons scrape produced 6 identical non-patent images
duplicated across all 8 USPC classes (physics diagrams, CAD screenshots,
historical engravings — none relevant to the class). These were replaced
in module 3b with 48 PIL-generated grayscale line drawings, 6 per class,
each depicting a recognizable product in the correct category:

  D6  — chair, sofa, table, bed, mirror, pillow
  D8  — hammer, wrench, screwdriver, saw, pliers, spirit level
  D9  — bottle, box, tin can, jar, bag, spray can
  D12 — bicycle, car, motorcycle, bus, truck, wheel
  D14 — laptop, phone, desktop, keyboard, headphones, tablet
  D23 — faucet, showerhead, toilet, bathtub, ceiling fan, bucket
  D24 — syringe, stethoscope, thermometer, pill, bandage, microscope
  D26 — light bulb, candle, flashlight, lantern, sparkle, star

These serve as visually distinct negatives within each class. They are
clearly not in-force US design patents, so the binary label is unambiguous.

### Candidate selection (module 6)

Original module 6 paired candidates by USPC class only. USPC classes at
the 2-digit level are too broad (D14 contains TV mounts, flat screens, GUIs,
and carabiners — all in different visual subcategories). Updated to
title-keyword matching: candidates whose invention_title shares a keyword
with the query are prioritised before random fill. This raised the
keyword-matched candidate rate from 0% to 54%.

---

## Setup

### 1. Python environment

Python 3.11+ required.

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Tesseract (required for module 4 text masking)

- Windows: https://github.com/UB-Mannheim/tesseract/wiki
  Install path: C:\Users\<user>\AppData\Local\Programs\Tesseract-OCR
  Add to PATH before running module 4.
- macOS: brew install tesseract
- Ubuntu: sudo apt install tesseract-ocr

### 3. Cairo (required for SVG rendering in module 4)

- Windows: GTK3 runtime from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases
- macOS: brew install cairo
- Ubuntu: sudo apt install libcairo2

### 4. API keys

```
cp .env.example .env
```

Required keys:

```
USPTO_API_KEY=...         # data.uspto.gov
ANTHROPIC_API_KEY=...     # console.anthropic.com
OPENAI_API_KEY=...        # platform.openai.com
GOOGLE_API_KEY=...        # aistudio.google.com
OPENROUTER_API_KEY=...    # openrouter.ai (Gemma 4 31B)
```

### 5. Verify setup

```bash
python scripts/00_verify_env.py
python scripts/test_connections.py
```

---

## Pipeline

Each script reads config.yaml as its single source of truth and is idempotent
(safe to re-run; already-completed work is skipped).

### Module 0 — environment check

```bash
python scripts/00_verify_env.py
```

### Module 1 — positive-class puller

```bash
python scripts/01_pull_uspto.py --dry-run --class D24
python scripts/01_pull_uspto.py --all
```

Output per patent in data/raw/positive/:

    D1049406.pdf    full patent PDF with all drawing sheets
    D1049406.json   metadata sidecar (grant date, title, USPC class, applicant)
    D1049406.txt    marker file
    D1049406.xml    raw USPTO grant XML (Locarno classification)

USPC classes:

    D6   Furniture           Locarno 06
    D8   Tools and hardware  Locarno 08
    D9   Packaging           Locarno 09
    D12  Transportation      Locarno 12
    D14  Electronics         Locarno 14
    D23  Fluid handling      Locarno 23
    D24  Medical instruments Locarno 24  (client anchor class)
    D26  Lighting            Locarno 26

### Module 2 — negative expired puller

```bash
python scripts/02_pull_negative_expired.py --dry-run --class D24
python scripts/02_pull_negative_expired.py --all
```

Pulls 9 per class (72 total). Module 5 stratifies down to 6 per class (48).

### Module 3 — negative open-source

```bash
python scripts/03_collect_negative_opensource.py --all
python scripts/03b_fix_wikimedia_negatives.py     # replace bad Wikimedia files
```

Original Wikimedia scrape had systematic quality issues (see Dataset notes).
Run 03b after 03 to replace bad images with PIL-generated line drawings.

### Module 4 — normalize

```bash
python scripts/04_normalize.py --dry-run
python scripts/04_normalize.py --id D1049406
python scripts/04_normalize.py --all
python scripts/04_normalize.py --source negative_opensource
python scripts/04_normalize.py --all --no-mask    # skip Tesseract
```

USPTO PDFs are pure raster scans — get_text() returns nothing.
Page detection uses blob analysis (numpy):
- Renders each page at 72 DPI
- Measures bounding-box coverage of ink (drawing pages: 0.25-0.55, reference pages: 0.02-0.08)
- Selects first page within 2% of maximum blob score (FIG. 1 perspective view)

Processing pipeline per image:
1. Find drawing sheet page via blob analysis (PDFs only)
2. Render at 150 DPI
3. Tesseract OCR at conf >= 70, mask patent metadata patterns only
4. Grayscale, pad to square, resize to 1024x1024, strip EXIF
5. Save to data/processed/<id>.png

### Module 5 — build manifest

```bash
python scripts/05_build_manifest.py
python scripts/05_build_manifest.py --check
```

Builds data/manifest.csv from all JSON sidecars. Stratifies expired patents
to 6 per class (48 total). Locarno codes from XMLs where available.

### Module 6 — candidate selection

```bash
python scripts/06_select_candidates.py
python scripts/06_select_candidates.py --check
```

Builds candidates/candidates.csv. For each of 192 queries, selects 5
candidates from the same USPC class using title-keyword matching (Fix B):
candidates whose invention_title shares a keyword with the query are
prioritised, then random fill. Fixed/seeded for reproducibility.

Output columns: query_id, query_label, query_uspc, query_png,
candidate_id, candidate_label, candidate_uspc, candidate_png, candidate_rank

### Module 7 — model evaluation

```bash
# Pilot (20 queries, zero_shot only, 1 rep = 80 calls)
python scripts/07_run_models.py --pilot --dry-run
python scripts/07_run_models.py --pilot --model claude
python scripts/07_run_models.py --pilot

# Full run (192 queries, 3 strategies, 3 reps = 6,912 calls)
python scripts/07_run_models.py --full --model gemma
python scripts/07_run_models.py --full --model claude
python scripts/07_run_models.py --full --model gemini
python scripts/07_run_models.py --full --model openai

# Resume is automatic — rerun same command after any interruption
```

Checkpoints saved to results/raw_responses/<job_id>.json after each call.
Resume logic skips completed jobs. Delete a .json file to force retry.

### Modules 8-9

```bash
python scripts/08_parse_responses.py   # produces results/parsed.csv
python scripts/09_compute_stats.py     # macro-F1, Cohen kappa, McNemar, figures
```

---

## Experiment design

```
Models:      Claude Sonnet 4.6, GPT-5.5, Gemini 2.5 Pro, Gemma 4 31B (OpenRouter)
Strategies:  zero_shot, few_shot, chain_of_thought
Reps:        3 per configuration, temperature=0
Full run:    192 queries x 4 models x 3 strategies x 3 reps = 6,912 calls
Pilot:       20 queries x 4 models x zero_shot x 1 rep = 80 calls
```

Model notes:
- Claude Sonnet 4.6: Anthropic API, max_tokens=512
- GPT-5.5: OpenAI API, max_completion_tokens=2048 (requires this param, not max_tokens)
- Gemini 2.5 Pro: google.generativeai, thinking_budget=512, max_output_tokens=3000
- Gemma 4 31B: OpenRouter (google/gemma-4-31b-it), replaces Qwen2.5-VL which was
  removed from Together AI serverless in May 2026

Cost estimate (standard pricing, May 2026):

```
GPT-5.5              ~$166  (Batch API)
Gemini 2.5 Pro        ~$90
Claude Sonnet 4.6     ~$95  (Batch API)
Gemma 4 31B            ~$8  (OpenRouter)
Total                ~$359
Budget ceiling        $800
```

---

## API notes (confirmed May 2026)

USPTO Patent File Wrapper API:
    Host:        https://api.uspto.gov
    Endpoint:    POST /api/v1/patent/applications/search
    Auth:        X-API-KEY header
    Query:       filters + rangeFilters arrays (NOT legacy _and/_eq syntax)
    Response:    patentFileWrapperDataBag
    Rate limit:  45 requests/minute
    Class field: applicationMetaData.class  (e.g. "D8" not "D08")

Patent PDF download (no auth):
    https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/D1049406

OpenRouter (Gemma 4 31B):
    Base URL:    https://openrouter.ai/api/v1
    Model:       google/gemma-4-31b-it
    Auth:        OPENROUTER_API_KEY
    Note:        Requires HTTP-Referer and X-Title headers

---

## Reproducibility

- random_seed: 42 in config.yaml controls all sampling
- manifest.csv and candidates.csv are committed to git
- Raw PDFs are gitignored but fully regenerable from the scripts
- All API responses saved to results/raw_responses/ for audit and re-parsing
- PIL-generated open-source negatives are deterministic (same shapes, same seed)

---

## Directory structure

    patent-benchmark/
    |-- config.yaml                   single source of truth
    |-- .env                          API keys (gitignored)
    |-- .env.example                  template
    |-- data/
    |   |-- raw/
    |   |   |-- positive/             96 PDFs + sidecars
    |   |   |-- negative_expired/     72 PDFs + sidecars (48 used)
    |   |   `-- negative_opensource/  48 PIL-generated PNGs
    |   |-- processed/                192 normalized PNGs
    |   `-- manifest.csv              192 rows, all metadata
    |-- candidates/
    |   `-- candidates.csv            960 pairs, 5 per query
    |-- results/
    |   |-- raw_responses/            one JSON per API call (module 7)
    |   |-- parsed.csv                module 8 output
    |   |-- run.log                   module 7 execution log
    |   `-- stats/                    module 9 output
    |-- scripts/
    |   |-- _common.py
    |   |-- 00_verify_env.py          done
    |   |-- 01_pull_uspto.py          done
    |   |-- 02_pull_negative_expired.py   done
    |   |-- 03_collect_negative_opensource.py  done
    |   |-- 03b_fix_wikimedia_negatives.py     done (PIL replacement)
    |   |-- 04_normalize.py           done
    |   |-- 05_build_manifest.py      done
    |   |-- 06_select_candidates.py   done (title-keyword matching)
    |   |-- 07_run_models.py          pilot done, full run pending
    |   |-- 08_parse_responses.py     not written
    |   |-- 09_compute_stats.py       not written
    |   |-- test_connections.py       API key verification
    |   |-- validate_pilot.py         pilot result audit
    |   `-- scratch.py                throw-away analysis (overwrite freely)
    `-- docs/
        |-- experiment_plan.md
        |-- sample_size.md
        `-- negative_opensource_protocol.md
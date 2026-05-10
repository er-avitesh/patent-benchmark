# Patent Infringement Benchmark

Benchmarking multimodal LLMs on US design-patent infringement detection


Authors: Avitesh Kesharwani, M.S. IEEE Senior Member / Bjarne Berg, Ph.D., UNC Charlotte

---

## Progress

```
Dataset
  Positive drawings (in-force patents)    96 /  96   [====================] 100%
  Negative expired  (public domain)       48 /  48   [====================] 100%
  Negative open-source (Wikimedia CC0)    48 /  48   [====================] 100%
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
  [x] Module 3   Negative open-source          48 drawings, Wikimedia Commons CC0
  [x] Module 4   Normalize                     192 PNGs, blob detection, Tesseract masking
  [x] Module 5   Build manifest                192 rows, 96 positive / 96 negative, Locarno coded
  [x] Module 6   Candidate selection           960 pairs, 5 per query, class-restricted
  [ ] Module 7   Model evaluation              next -- pilot 20 queries, then full run
  [ ] Module 8   Parse responses               parsed.csv
  [ ] Module 9   Statistics                    macro-F1, Cohen kappa, McNemar
```

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

### 4. USPTO API key

Register at https://data.uspto.gov/apis/getting-started for a free API key.

```
cp .env.example .env
# Add: USPTO_API_KEY=your_key_here
```

### 5. Verify setup

```bash
python scripts/00_verify_env.py
```

---

## Pipeline

Each script reads config.yaml as its single source of truth and is idempotent
(safe to re-run; already-completed work is skipped).

### Module 0 -- environment check

```bash
python scripts/00_verify_env.py
```

### Module 1 -- positive-class puller

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

Note: XMLs were only retained for the initial D24 pull. Re-run --all to
regenerate missing XMLs for full 4-digit Locarno codes (no PDF re-download).

### Module 2 -- negative expired puller

```bash
python scripts/02_pull_negative_expired.py --dry-run --class D24
python scripts/02_pull_negative_expired.py --all
```

Pulls 9 per class (72 total). Module 5 stratifies down to 6 per class (48).

### Module 3 -- negative open-source

```bash
python scripts/03_collect_negative_opensource.py --dry-run --class D24
python scripts/03_collect_negative_opensource.py --all
```

Downloads technical drawings from Wikimedia Commons (public domain / CC0).
Target: 6 per class, 48 total. Configured via negative_composition.open_source_count.

Key implementation notes:
- Wikimedia requires a descriptive User-Agent on ALL requests (API + downloads).
  Missing User-Agent returns HTTP 403.
- Category names must be exact. Generic categories (Technical_drawings,
  Engineering_diagrams) return mixed content filtered by drawing keywords.
- Windows: strip illegal filename characters from Wikimedia titles before saving.

### Module 4 -- normalize

```bash
python scripts/04_normalize.py --dry-run
python scripts/04_normalize.py --id D1049406
python scripts/04_normalize.py --all
python scripts/04_normalize.py --source negative_opensource
python scripts/04_normalize.py --all --no-mask    # skip Tesseract
```

USPTO PDFs are pure raster scans -- get_text() returns nothing on any page.
Page detection uses blob analysis (numpy):
- Renders each page at 72 DPI
- Measures bounding-box coverage of ink as a fraction of page area
- Drawing pages: one large centered figure, blob score 0.25-0.55
- Reference pages: scattered text fragments, blob score 0.02-0.08
- Selects first page within 2% of maximum blob score (FIG. 1 perspective view)

Processing pipeline per image:
1. Find drawing sheet page (PDFs only, via blob analysis)
2. Render at 150 DPI
3. Tesseract OCR at conf >= 70, mask patent metadata patterns only
4. Grayscale, pad to square, resize to 1024x1024, strip EXIF
5. Save to data/processed/<id>.png

### Module 5 -- build manifest

```bash
python scripts/05_build_manifest.py
python scripts/05_build_manifest.py --check
```

Builds data/manifest.csv from all JSON sidecars. Stratifies expired patents
to 6 per class (48 total) to match open-source count. Locarno codes from
XMLs where available; derived from USPC class otherwise (e.g. D9 -> 0900).

### Module 6 -- candidate selection

```bash
python scripts/06_select_candidates.py
python scripts/06_select_candidates.py --check
```

Builds candidates/candidates.csv. For each of 192 queries, selects 5
candidates from the same USPC class. Fixed at build time (seeded) for
reproducibility across models, strategies, and repetitions.

Output columns: query_id, query_label, query_uspc, query_png,
candidate_id, candidate_label, candidate_uspc, candidate_png, candidate_rank

### Modules 7-9

    Module 7   07_run_models.py        pilot: 20 queries; full: 7,200 calls
    Module 8   08_parse_responses.py   parsed.csv
    Module 9   09_compute_stats.py     macro-F1, Cohen kappa, McNemar, figures

---

## Experiment design

    Models:      GPT-5.5, Claude Sonnet 4.6, Gemini 2.5 Pro, Qwen2.5-VL-7B
    Strategies:  zero_shot, few_shot, chain_of_thought
    Repetitions: 3 per configuration, temperature=0
    Full run:    192 queries x 4 models x 3 strategies x 3 reps = 6,912 calls
    Pilot:       20 queries x 4 models x binary verdict (no rubric)

    Cost estimate (standard pricing, May 2026):
      GPT-5.5              $331   ($166 with Batch API)
      Claude Sonnet 4.6    $190   ( $95 with Batch API)
      Gemini 2.5 Pro        $90
      Qwen2.5-VL-7B          $0   (self-hosted)
      Standard total       ~$611
      Batch API total      ~$351

    Per Prof. Berg: run pilot first, then full run if results are tractable.
    Budget ceiling: $800.

---

## API notes (confirmed May 2026)

USPTO Patent File Wrapper API:
    Host:        https://api.uspto.gov
    Endpoint:    POST /api/v1/patent/applications/search
    Auth:        X-API-KEY header
    Query:       filters + rangeFilters arrays (NOT legacy _and/_eq syntax)
    Response:    patentFileWrapperDataBag
    Rate limit:  45 requests/minute
    Class field: applicationMetaData.class  (e.g. "D8", not "D08")

Patent PDF download (no auth):
    https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/D1049406

---

## Reproducibility

- random_seed: 42 in config.yaml controls all sampling
- manifest.csv and candidates.csv are committed to git
- Raw PDFs are gitignored but fully regenerable from the scripts
- All API responses saved to raw_responses/ for audit and re-parsing

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
    |   |   `-- negative_opensource/  48 drawings + sidecars
    |   |-- processed/                192 normalized PNGs + norm sidecars
    |   `-- manifest.csv              192 rows, all metadata
    |-- candidates/
    |   `-- candidates.csv            960 pairs, 5 per query
    |-- prompts/                      ZS, FS, CoT templates
    |-- raw_responses/                module 7 output (one JSON per call)
    |-- results/
    |   |-- parsed.csv                module 8 output
    |   `-- stats/                    module 9 output
    |-- scripts/
    |   |-- _common.py
    |   |-- 00_verify_env.py          done
    |   |-- 01_pull_uspto.py          done
    |   |-- 02_pull_negative_expired.py   done
    |   |-- 03_collect_negative_opensource.py  done
    |   |-- 04_normalize.py           done
    |   |-- 05_build_manifest.py      done
    |   |-- 06_select_candidates.py   done
    |   `-- ...
    `-- docs/
        |-- experiment_plan.md
        |-- sample_size.md
        `-- negative_opensource_protocol.md
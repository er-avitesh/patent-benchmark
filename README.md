# Patent Infringement Benchmark

Benchmarking multimodal LLMs on US design-patent infringement detection
under the ordinary observer test (Egyptian Goddess v. Swisa, 2008).

Authors: Avitesh Kesharwani, M.S. IEEE Senior Member / Bjarne Berg, Ph.D., UNC Charlotte
Companion paper: ADR Compliance Benchmark (Kesharwani & Berg, 2026)

---

## Progress

```
Dataset
  Positive drawings (in-force patents)   96 / 96    [====================] 100%
  Negative expired  (public domain)      72 / 72    [====================] 100%
  Negative open-source (Wikimedia Commons) 23 / 32    [==============      ]  72%
  Total                                 191 / 200   [=================== ]  96%

Pipeline
  [x] Module 0   Environment check
  [x] Module 1   Positive-class puller         96 PDFs across 8 USPC classes
  [x] Module 2   Negative expired puller       72 PDFs across 8 USPC classes
  [x] Module 3   Negative open-source          23 drawings from Wikimedia Commons
  [~] Module 4   Normalize                     running -- PDF to PNG, blob detection, text masking
  [ ] Module 5   Build manifest                manifest.csv
  [ ] Module 6   Candidate selection           candidates.csv
  [ ] Module 7   Model evaluation              7,200 API calls, ~$600
  [ ] Module 8   Parse responses               parsed.csv
  [ ] Module 9   Statistics                    F1, kappa, McNemar
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

- Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
- macOS: brew install tesseract
- Ubuntu: sudo apt install tesseract-ocr

Not needed for modules 0-3. Install before running 04_normalize.py.

### 3. USPTO API key

Register at https://data.uspto.gov/apis/getting-started for a free API key.

```
cp .env.example .env
# Add: USPTO_API_KEY=your_key_here
```

### 4. Verify setup

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

Checks API key, tesseract, directory structure, and USPTO API reachability.

### Module 1 -- positive-class puller

Fetches granted US design patents and downloads the full patent PDF for each.

```bash
python scripts/01_pull_uspto.py --dry-run --class D24   # verify first
python scripts/01_pull_uspto.py --class D24              # one class
python scripts/01_pull_uspto.py --all                    # all 8 classes
```

Output per patent in data/raw/positive/:

    D1049406.pdf    full patent PDF with all drawing sheets
    D1049406.json   metadata sidecar (grant date, title, USPC class, applicant)
    D1049406.txt    marker file
    D1049406.xml    raw USPTO grant XML (includes Locarno classification)

PDF structure (confirmed from live data):
- Page 1: bibliographic front page
- Page 2+: references continuation (0 or more pages)
- First drawing sheet: contains "Sheet 1 of N" -- always FIG. 1 (perspective view)

USPC classes pulled:

    D6   Furniture           (Locarno 06)
    D8   Tools and hardware  (Locarno 08)
    D9   Packaging           (Locarno 09)
    D12  Transportation      (Locarno 12)
    D14  Electronics         (Locarno 14)
    D23  Fluid handling      (Locarno 23)
    D24  Medical instruments (Locarno 24) -- client anchor class
    D26  Lighting            (Locarno 26)

### Module 2 -- negative expired puller

Same approach as module 1, date window before 2008-01-01 (patents past term).

```bash
python scripts/02_pull_negative_expired.py --dry-run --class D24
python scripts/02_pull_negative_expired.py --all
```

Output goes to data/raw/negative_expired/ with label "negative" in JSON sidecar.

### Module 3 -- negative open-source

Script: 03_collect_negative_opensource.py -- complete.

23 technical drawings from Wikimedia Commons (public domain / CC0), 3 per class.
Source categories: Technical_drawings_of_instruments, Historical_surgical_instruments,
Diagrams_of_vehicles, Piping_and_instrumentation_diagrams, and others.
Requires User-Agent header on all API and download requests (missing = 403).

    python scripts/03_collect_negative_opensource.py --dry-run --class D24
    python scripts/03_collect_negative_opensource.py --all

### Module 4 -- normalize

Script: 04_normalize.py -- complete.

USPTO PDFs are pure raster scans -- get_text() returns nothing on any page.
Drawing sheet detection uses blob analysis (numpy) rather than text search:
- Renders each page at 72 DPI to measure the bounding-box fraction of ink
- Drawing pages have one large centered figure (blob score 0.50+)
- Reference/front-matter pages have scattered small text blobs (blob score 0.02-0.05)
- Selects the first page with blob score within 2% of the maximum

For each raw PDF:
1. Scan all pages at 72 DPI, compute blob score per page
2. Select first page with highest blob score (FIG. 1 perspective view)
3. Render that page at 150 DPI
4. Tesseract OCR (conf >= 70, patent metadata patterns only) -- white out matches
5. Grayscale, pad to square, resize to 1024x1024, strip metadata
6. Save to data/processed/<patent_number>.png

For open-source PNG/JPG: load directly, no masking needed.
For open-source SVG: render via cairosvg, then same pipeline.

    python scripts/04_normalize.py --dry-run
    python scripts/04_normalize.py --id D1049406
    python scripts/04_normalize.py --all
    python scripts/04_normalize.py --all --no-mask   # skip Tesseract

### Module 5 -- build manifest

Script: 05_build_manifest.py -- not yet written.

Assembles data/manifest.csv from JSON sidecars and processed images.
Parses XML sidecars to extract Locarno classifications.

### Modules 6-9

    Module 6   06_select_candidates.py    candidates.csv, 5 per query, class-restricted
    Module 7   07_run_models.py           7,200 calls (200 x 4 x 3 x 3), temp=0
    Module 8   08_parse_responses.py      parsed.csv, multi-strategy JSON parser
    Module 9   09_compute_stats.py        macro-F1, Cohen kappa, McNemar, figures

---

## Configuration

All parameters in config.yaml. Edit once, all scripts pick it up.

    total_samples:         200   # 100 positive + 100 negative
    candidates_per_query:    5   # shown to model per query
    repetitions:             3   # per configuration, temperature=0
    random_seed:            42   # controls all sampling

Models (4 multimodal vision models):
    gpt-5.5
    claude-sonnet-4.6
    gemini-2.5-pro
    qwen2.5-vl-7b          # open-weight vision baseline

Strategies: zero_shot, few_shot, chain_of_thought

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
    Patent no.:  applicationMetaData.patentNumber

Patent PDF download (no auth required):
    https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/D1049406

---

## Reproducibility

- random_seed: 42 in config.yaml controls all sampling
- data/manifest.csv and candidates/candidates.csv are committed to git
- Raw PDFs are gitignored but fully regenerable from the scripts
- All API responses saved to raw_responses/ for audit and re-parsing

---

## Directory structure

    patent-benchmark/
    |-- config.yaml                  single source of truth
    |-- .env                         API keys (gitignored)
    |-- .env.example                 template
    |-- data/
    |   |-- raw/
    |   |   |-- positive/            module 1 output (96 PDFs)
    |   |   |-- negative_expired/    module 2 output (72 PDFs)
    |   |   `-- negative_opensource/ module 3 output (~32 drawings)
    |   |-- processed/               module 4 output (normalized PNGs)
    |   `-- manifest.csv             module 5 output
    |-- candidates/
    |   `-- candidates.csv           module 6 output
    |-- prompts/                     ZS, FS, CoT templates
    |-- raw_responses/               module 7 output (one JSON per call)
    |-- results/
    |   |-- parsed.csv               module 8 output
    |   `-- stats/                   module 9 output
    |-- scripts/
    |   |-- _common.py
    |   |-- 00_verify_env.py         done
    |   |-- 01_pull_uspto.py         done
    |   |-- 02_pull_negative_expired.py  done
    |   `-- ...
    `-- docs/
        |-- experiment_plan.md
        |-- sample_size.md
        `-- negative_opensource_protocol.md
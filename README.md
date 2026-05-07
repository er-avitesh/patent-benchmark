# Patent Infringement Benchmark

Benchmarking multimodal LLMs on US design-patent infringement detection
under the ordinary observer test (Egyptian Goddess v. Swisa, 2008).

**Authors:** Avitesh Kesharwani, M.S. IEEE Senior Member · Bjarne Berg, Ph.D., UNC Charlotte
**Companion paper:** ADR Compliance Benchmark (Kesharwani & Berg, 2026)

---

## Status

| Module | Script | Status |
|---|---|---|
| 0 | `00_verify_env.py` | ✅ Complete |
| 1 | `01_pull_uspto.py` | ✅ Complete — 96 positive-class PDFs across 8 USPC classes |
| 2 | `02_pull_negative_expired.py` | 🔜 Next |
| 3 | `03_collect_negative_opensource.py` | 🔜 Pending |
| 4 | `04_normalize.py` | 🔜 Pending |
| 5 | `05_build_manifest.py` | 🔜 Pending |
| 6 | `06_select_candidates.py` | 🔜 Pending |
| 7 | `07_run_models.py` | 🔜 Pending |
| 8 | `08_parse_responses.py` | 🔜 Pending |
| 9 | `09_compute_stats.py` | 🔜 Pending |

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

### 2. Tesseract (required for module 4 text-masking)

- **Windows:** download installer from https://github.com/UB-Mannheim/tesseract/wiki
- **macOS:** `brew install tesseract`
- **Ubuntu:** `sudo apt install tesseract-ocr`

Not needed for modules 0-3. Install before running `04_normalize.py`.

### 3. USPTO API key

Register at https://data.uspto.gov/apis/getting-started to obtain a free API key.

```
cp .env.example .env
# Add your key: USPTO_API_KEY=your_key_here
```

### 4. Verify

```bash
python scripts/00_verify_env.py
```

---

## Pipeline

Each script reads `config.yaml` as its single source of truth and is idempotent
(safe to re-run — already-completed work is skipped).

### Module 0 — environment check

```bash
python scripts/00_verify_env.py
```

Checks API key, tesseract, directory structure, and USPTO API reachability.

---

### Module 1 — positive-class puller

Fetches granted US design patents from the USPTO Open Data Portal and downloads
the full patent PDF (including all drawing sheets) for each.

```bash
# Dry run — verify API response shape, no downloads
python scripts/01_pull_uspto.py --dry-run --class D24

# Pull one class
python scripts/01_pull_uspto.py --class D24

# Pull all 8 classes (~20 min)
python scripts/01_pull_uspto.py --all
```

**Outputs per patent** (in `data/raw/positive/`):

| File | Contents |
|---|---|
| `D1049406.pdf` | Full patent PDF — bibliographic page + all drawing sheets |
| `D1049406.json` | Curated metadata sidecar (grant date, title, USPC class, applicant) |
| `D1049406.txt` | Marker file — `pdf:D1049406.pdf` on success |
| `D1049406.xml` | Raw USPTO grant XML (includes Locarno classification) |

**PDF structure** (confirmed from live data):
- Page 1: bibliographic front page (title, inventor, claim, thumbnail sketches)
- Page 2+: references continuation (may be 0 or more pages)
- First drawing sheet: contains "Sheet 1 of N" in header — always FIG. 1 (perspective view)
- Subsequent sheets: orthographic views (front, rear, side, top, bottom)

`04_normalize.py` extracts the page containing "Sheet 1 of" as the canonical drawing.

**USPC classes pulled** (confirmed field: `applicationMetaData.class`):

| Class | Domain | Locarno equiv |
|---|---|---|
| D6 | Furniture | 06 |
| D8 | Tools and hardware | 08 |
| D9 | Packaging and containers | 09 |
| D12 | Transportation | 12 |
| D14 | Electronics housings | 14 |
| D23 | Fluid handling | 23 |
| D24 | Medical instruments *(client anchor)* | 24 |
| D26 | Lighting | 26 |

---

### Module 2 — negative-class puller (expired patents)

*Script: `02_pull_negative_expired.py` — not yet written*

Same approach as module 1, but with `grant_date < 2008-01-01` to select
patents past their 14/15-year term. These are in the public domain.

Target: ~73 drawings across the same 8 USPC classes.

---

### Module 3 — negative-class open-source (semi-manual)

*Script: `03_collect_negative_opensource.py` — not yet written*

~31 technical drawings from TraceParts and similar CAD repositories.
This portion requires manual curation — see `docs/negative_opensource_protocol.md`.

---

### Module 4 — normalize

*Script: `04_normalize.py` — not yet written*

For each raw PDF:
1. Locate the page containing "Sheet 1 of" (FIG. 1 perspective view)
2. Render that page at 300 DPI as PNG
3. Run Tesseract OCR, mask detected text regions (patent numbers, figure labels, sheet numbers)
4. Convert to grayscale, resize to 1024x1024 with white padding
5. Strip EXIF/metadata
6. Save to `data/processed/<patent_number>.png`

---

### Module 5 — build manifest

*Script: `05_build_manifest.py` — not yet written*

Assembles `data/manifest.csv` from the `.json` sidecars and normalised images.
Also parses the `.xml` sidecars to extract Locarno classifications
(`<classification-locarno><main-classification>2401</main-classification>`).

---

### Modules 6-9

*Scripts not yet written — see experiment plan in `docs/experiment_plan.md`*

| Module | Script | Purpose |
|---|---|---|
| 6 | `06_select_candidates.py` | Build `candidates/candidates.csv` — 5 candidates per query |
| 7 | `07_run_models.py` | 7,200 API calls (200 queries x 4 models x 3 strategies x 3 reps) |
| 8 | `08_parse_responses.py` | Extract verdicts from raw responses → `results/parsed.csv` |
| 9 | `09_compute_stats.py` | Macro-F1, Cohen's kappa, McNemar → tables and figures for paper |

---

## Configuration

All study parameters live in `config.yaml`. Change once, every script picks it up.

Key parameters:

```yaml
study:
  total_samples: 200          # 100 positive + 100 negative
  candidates_per_query: 5     # candidates shown to model per query
  repetitions: 3              # runs per configuration (temperature=0)

models:                       # 4 multimodal vision models
  - gpt-5.5
  - claude-sonnet-4.6
  - gemini-2.5-pro
  - qwen2.5-vl-7b             # open-weight vision baseline (replaces Mistral 7B)

strategies: [zero_shot, few_shot, chain_of_thought]
```

---

## API notes (confirmed from live testing, May 2026)

**USPTO Patent File Wrapper API**
- Host: `https://api.uspto.gov`
- Search endpoint: `POST /api/v1/patent/applications/search`
- Auth: `X-API-KEY` header
- Query syntax: `filters` + `rangeFilters` arrays (NOT the legacy `_and`/`_eq` syntax)
- Response bag key: `patentFileWrapperDataBag`
- Rate limit: 45 requests/minute
- USPC class field: `applicationMetaData.class` (values: `"D6"`, `"D8"`, not `"D06"`, `"D08"`)
- Patent number field: `applicationMetaData.patentNumber` (e.g. `"D1049406"`)
- Drawing URL field: `grantDocumentMetaData.fileLocationURI` (API path, not directly downloadable)

**Patent PDF download (no auth required)**
- URL pattern: `https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/<patent_number>`
- Example: `https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/D1049406`
- Returns a complete multi-page PDF with all drawing sheets
- No API key required — public endpoint

---

## Reproducibility

- `random_seed: 42` in `config.yaml` controls all sampling
- `data/manifest.csv` and `candidates/candidates.csv` are committed to git
- Raw PDFs and images are gitignored but fully regenerable from the scripts
- All API responses saved to `raw_responses/` for audit and re-parsing

---

## Directory structure

```
patent-benchmark/
├── config.yaml               <- single source of truth for all parameters
├── .env                      <- API keys (gitignored)
├── .env.example              <- template
├── data/
│   ├── raw/
│   │   ├── positive/         <- module 1 output (PDFs, JSONs, XMLs)
│   │   ├── negative_expired/ <- module 2 output
│   │   └── negative_opensource/ <- module 3 output
│   ├── processed/            <- module 4 output (normalised PNGs)
│   └── manifest.csv          <- module 5 output
├── candidates/
│   └── candidates.csv        <- module 6 output
├── prompts/                  <- prompt templates (ZS, FS, CoT)
├── raw_responses/            <- module 7 output (one JSON per API call)
├── results/
│   ├── parsed.csv            <- module 8 output
│   └── stats/                <- module 9 output (tables, figures)
├── scripts/
│   ├── _common.py            <- shared utilities
│   ├── 00_verify_env.py
│   ├── 01_pull_uspto.py      <- complete
│   └── ...
└── docs/
    ├── experiment_plan.md
    ├── sample_size.md
    └── negative_opensource_protocol.md
```
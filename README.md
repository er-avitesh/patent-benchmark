# Patent Infringement Benchmark

Benchmarking multimodal LLMs on US design-patent infringement detection
under the ordinary observer test (Egyptian Goddess v. Swisa, 2008).

Companion to the ADR compliance benchmark (Kesharwani & Berg, 2026).

## Status

🚧 **Module 1 in progress** — dataset construction.

## Setup

1. Python 3.11+. Create a venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   # .venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. Install Tesseract for OCR-based text masking:
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
   - Windows: download from https://github.com/UB-Mannheim/tesseract/wiki

3. Get a USPTO API key:
   - Register at https://data.uspto.gov
   - Copy `.env.example` to `.env` and add your key.

4. Verify the environment:
   ```bash
   python scripts/00_verify_env.py
   ```

## Pipeline

| # | Script | Produces |
|---|---|---|
| 0 | `00_verify_env.py` | sanity check — API keys present, tesseract installed |
| 1 | `01_pull_uspto.py` | `data/raw/positive/*.png` + `*.json` metadata |
| 2 | `02_pull_negative_expired.py` | `data/raw/negative_expired/*.png` |
| 3 | `03_collect_negative_opensource.py` | `data/raw/negative_opensource/*.png` (semi-manual) |
| 4 | `04_normalize.py` | `data/processed/*.png` (text-masked, grayscale, 1024px) |
| 5 | `05_build_manifest.py` | `data/manifest.csv` |
| 6 | `06_select_candidates.py` | `candidates/candidates.csv` |
| 7 | `07_run_models.py` | `raw_responses/*.json` |
| 8 | `08_parse_responses.py` | `results/parsed.csv` |
| 9 | `09_compute_stats.py` | `results/stats/*` |

Each script reads `config.yaml` and is idempotent / resumable.

## Config

All study parameters live in `config.yaml`. To change a value (sample size,
model list, USPC classes), edit there — every script reads from this file.

## Reproducibility

`random_seed: 42` in the config controls all sampling. The manifest CSV
and candidates CSV are committed to git; raw drawings are regenerable
from the scripts and are gitignored.
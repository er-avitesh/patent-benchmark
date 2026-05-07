"""
01_pull_uspto.py — positive-class puller.

Pulls design patents from USPTO Open Data Portal (data.uspto.gov), one
USPC class at a time, then downloads the principal drawing for each
patent. Saves both the JSON metadata and the PNG drawing to disk.

USAGE
-----
    # Step 1: smoke-test the API shape with one class, no downloads
    python scripts/01_pull_uspto.py --dry-run --class D24

    # Step 2: pull just one class, save drawings
    python scripts/01_pull_uspto.py --class D24

    # Step 3: pull all 8 classes (after step 2 looks correct)
    python scripts/01_pull_uspto.py --all

WHY THE THREE-STEP RAMP
-----------------------
USPTO migrated their API in March 2026 and the exact endpoint shape may
differ from what's hardcoded here. Run --dry-run first, observe the
JSON response, then adjust _parse_search_response() if field names differ.
Don't kick off a full pull until --class D24 succeeds end-to-end.

OUTPUTS
-------
For each patent successfully pulled:
    data/raw/positive/<patent_number>.png   — drawing
    data/raw/positive/<patent_number>.json  — metadata sidecar

The script is idempotent: if a file already exists, it is skipped.

NOTES FOR FIRST RUN (you, in VS Code)
-------------------------------------
1. Run 00_verify_env.py first to confirm API key works.
2. Run with --dry-run --class D24 and READ the printed JSON response.
3. If field names don't match _parse_search_response(), adjust them and
   commit the fix before proceeding.
4. The drawing-download URL is constructed from the patent number; verify
   on first real pull that the URL pattern is correct. USPTO sometimes
   changes how PDFs/images are served.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from _common import (
    console,
    get_logger,
    load_config,
    load_env,
    require_env_var,
    resolve_path,
)

logger = get_logger("pull_uspto")


# =============================================================================
# Patent search — query USPTO API by USPC class + grant date window
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _search_patents_for_class(
    config: dict,
    uspc_class: str,
    grant_date_start: str,
    grant_date_end: str,
    api_key: str,
    page_size: int = 100,
) -> list[dict]:
    """
    Query USPTO API for design patents in one USPC class within a date window.

    Returns a list of patent records (raw dicts as returned by the API).
    Pagination is handled internally — the function loops until all results
    are fetched or `MAX_RESULTS_PER_CLASS` is hit.

    The exact endpoint and request body shape is approximate and will need
    verification against the live API on first run. Field names are noted
    inline based on USPTO documentation; adjust if the live response differs.
    """
    base_url = config["uspto_api"]["base_url"].rstrip("/")
    endpoint = config["uspto_api"]["patent_search_endpoint"]
    url = f"{base_url}{endpoint}"

    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # -------------------------------------------------------------------------
    # STRATEGY: minimal request first, verify field names, then add filters.
    #
    # We query for Design patents in the date range with NO class filter.
    # The dry-run prints the first record so you can see the actual field
    # names. Once confirmed, we re-add the USPC class filter.
    #
    # NOTE: The USPC class field name for design patents is uncertain —
    # the spec only shows utility patent examples. After the dry-run reveals
    # the actual field names, update the filters block below.
    # -------------------------------------------------------------------------

    # Step 1 body: minimal — just type + date, no class filter.
    # This will return *some* Design patents so we can inspect the schema.
    body = {
        "q": "applicationMetaData.applicationTypeLabelName:Design",
        "rangeFilters": [
            {
                "field": "applicationMetaData.grantDate",
                "valueFrom": grant_date_start,
                "valueTo": grant_date_end,
            }
        ],
        "filters": [
            {
                "name": "applicationMetaData.applicationStatusDescriptionText",
                "value": ["Patented Case"],
            }
        ],
        "sort": [{"field": "applicationMetaData.grantDate", "order": "desc"}],
        "pagination": {"offset": 0, "limit": page_size},
    }

    all_results: list[dict] = []
    MAX_RESULTS_PER_CLASS = 500   # cap to keep things sane for sampling

    while True:
        current_offset = body["pagination"]["offset"]
        logger.info(
            f"  → querying {uspc_class}, offset={current_offset}, "
            f"have {len(all_results)} so far"
        )
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=config["uspto_api"]["request_timeout_seconds"],
        )

        # On 400, dump the response body so you can see what field name is wrong
        if response.status_code == 400:
            logger.error(
                f"  400 Bad Request. Response body:\n{response.text[:2000]}"
            )
        response.raise_for_status()
        data = response.json()

        # *** RESPONSE KEY — verify on first run ***
        # ODP Patent File Wrapper API uses "patentFileWrapperDataBag"
        # If you see a different top-level key in the printed debug output,
        # update this line and the pagination check below.
        page_results = (
            data.get("patentFileWrapperDataBag")
            or data.get("patents")
            or data.get("results")
            or []
        )

        # On first dry-run: print the raw response top-level keys so we can
        # verify the response key name above is correct.
        if current_offset == 0:
            logger.info(
                f"  Response top-level keys: {list(data.keys())}"
            )
            if page_results:
                logger.info(
                    f"  First record keys: {list(page_results[0].keys())}"
                )

        if not page_results:
            break

        all_results.extend(page_results)

        if len(all_results) >= MAX_RESULTS_PER_CLASS:
            logger.info(f"  reached cap of {MAX_RESULTS_PER_CLASS}, stopping pagination")
            all_results = all_results[:MAX_RESULTS_PER_CLASS]
            break

        if len(page_results) < page_size:
            # Last page
            break

        body["pagination"]["offset"] += page_size
        # Polite pacing — USPTO documents 45 req/min for PatentSearch.
        time.sleep(60.0 / config["uspto_api"]["rate_limit_per_minute"])

    return all_results


# =============================================================================
# Drawing download — given a patent number, fetch its principal drawing
# =============================================================================


def _build_drawing_url(patent_number: str) -> str:
    """
    Construct the URL for a design patent's drawing image.

    USPTO serves design patent drawings in several places:
      1. Patent Center (https://ppubs.uspto.gov) — needs session
      2. Bulk data PDFs (https://bulkdata.uspto.gov) — full grant XML+PDF
      3. PatFT image server (legacy)

    The cleanest approach for design patents is to fetch the front-page
    image from the design patent's bibliographic page. This URL pattern
    is approximate and MUST be verified on first run.
    """
    # Strip leading 'D' if present, pad as needed.
    pn = patent_number.upper().lstrip("D")
    # Example legacy pattern (verify):
    return f"https://pdfpiw.uspto.gov/.piw?Docid=D{pn:>07s}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _download_drawing(patent_number: str, save_path: Path) -> bool:
    """
    Download a single drawing. Returns True on success.

    NOTE: design patent drawings are typically distributed as multi-page
    PDFs (one figure per page) inside a TIFF/PDF wrapper. For v1 we fetch
    the principal drawing as PNG when available; otherwise PDF and convert
    later in 04_normalize.py.

    *** This function will likely need rework once you see the actual
    response from USPTO. ***
    """
    url = _build_drawing_url(patent_number)
    response = requests.get(url, timeout=30, allow_redirects=True)
    if response.status_code != 200:
        logger.warning(f"  ✗ drawing download failed for {patent_number}: HTTP {response.status_code}")
        return False

    save_path.write_bytes(response.content)
    return True


# =============================================================================
# Per-class orchestration: search → sample → download
# =============================================================================


def _select_random_sample(
    patents: list[dict], n: int, seed: int
) -> list[dict]:
    """Pick n patents at random with a fixed seed for reproducibility."""
    rng = random.Random(seed)
    if len(patents) < n:
        logger.warning(f"  only {len(patents)} patents available, wanted {n}")
        return patents
    return rng.sample(patents, n)


def _patent_number_field(patent: dict) -> str | None:
    """
    Extract the patent number from an ODP Patent File Wrapper record.

    ODP returns applicationNumberText as the primary ID. Design patents
    are typically prefixed with 'D' in their display form (e.g. D1234567)
    but stored without prefix in the API. We normalise to 'D<number>'
    for consistency with USPTO drawing filenames.
    """
    raw = (
        patent.get("applicationNumberText")
        or patent.get("patent_number")
        or patent.get("patent_id")
    )
    if not raw:
        return None
    # Normalise: strip non-digits for the number portion, re-add 'D' prefix
    num = raw.strip().lstrip("Dd").lstrip("/").strip()
    return f"D{num}" if num else None


def pull_class(
    config: dict,
    uspc_class: str,
    api_key: str,
    sample_size: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Pull positive-class patents for one USPC class. Returns summary stats.
    """
    seed = config["random_seed"]
    grant_start = config["date_windows"]["positive_grant_start"]
    grant_end = config["date_windows"]["positive_grant_end"]

    logger.info(f"[bold cyan]Class {uspc_class}[/bold cyan] — searching")
    patents = _search_patents_for_class(
        config=config,
        uspc_class=uspc_class,
        grant_date_start=grant_start,
        grant_date_end=grant_end,
        api_key=api_key,
    )
    logger.info(f"  found {len(patents)} patents matching filters")

    if dry_run:
        if patents:
            rec = patents[0]
            console.print(f"\n[bold green]✓ Got {len(patents)} records[/bold green]")
            console.print("\n[bold yellow]Full first record — copy the field names you need:[/bold yellow]")
            console.print_json(data=rec)

            # Flatten nested keys so we can grep for "uspc", "class", "design"
            def flatten(d, prefix=""):
                keys = []
                for k, v in d.items():
                    full = f"{prefix}.{k}" if prefix else k
                    keys.append(f"{full}: {repr(v)[:60]}")
                    if isinstance(v, dict):
                        keys.extend(flatten(v, full))
                    elif isinstance(v, list) and v and isinstance(v[0], dict):
                        keys.extend(flatten(v[0], f"{full}[0]"))
                return keys

            console.print("\n[bold yellow]Flattened field paths (look for USPC/class fields):[/bold yellow]")
            for line in flatten(rec):
                if any(kw in line.lower() for kw in ["uspc","class","type","design","d24","d0"]):
                    console.print(f"  [cyan]→ {line}[/cyan]")
                else:
                    console.print(f"    {line}")
        else:
            console.print("[red]No results — check q and filters in _search_patents_for_class()[/red]")
        return {
            "class": uspc_class,
            "found": len(patents),
            "downloaded": 0,
            "dry_run": True,
        }

    # Oversample by 25% to allow for download failures
    target_n = int(sample_size * 1.25)
    sampled = _select_random_sample(patents, target_n, seed)

    out_dir = resolve_path(config["paths"]["raw_positive"])
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failures: list[str] = []
    for patent in sampled:
        if downloaded >= sample_size:
            break

        patent_number = _patent_number_field(patent)
        if not patent_number:
            logger.warning(f"  skipping record without patent number: {patent}")
            continue

        # Idempotency: skip if already downloaded
        png_path = out_dir / f"{patent_number}.png"
        json_path = out_dir / f"{patent_number}.json"
        if png_path.exists() and json_path.exists():
            logger.info(f"  • {patent_number} already on disk, skipping")
            downloaded += 1
            continue

        # Save metadata sidecar first
        json_path.write_text(json.dumps(patent, indent=2, sort_keys=True))

        # Then the drawing
        ok = _download_drawing(patent_number, png_path)
        if ok:
            logger.info(f"  ✓ {patent_number}")
            downloaded += 1
        else:
            failures.append(patent_number)
            json_path.unlink(missing_ok=True)  # don't keep orphan metadata

        # Polite pacing
        time.sleep(60.0 / config["uspto_api"]["rate_limit_per_minute"])

    return {
        "class": uspc_class,
        "found": len(patents),
        "sampled": len(sampled),
        "downloaded": downloaded,
        "failures": failures,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Pull positive-class patent drawings from USPTO.")
    parser.add_argument("--class", dest="uspc_class", help="USPC class to pull (e.g., D24)")
    parser.add_argument("--all", action="store_true", help="Pull all 8 classes from config")
    parser.add_argument("--dry-run", action="store_true", help="Search only, no downloads, print sample record")
    args = parser.parse_args()

    if not args.uspc_class and not args.all:
        parser.error("Specify --class <id> or --all")

    config = load_config()
    load_env()
    api_key = require_env_var(config["uspto_api"]["api_key_env_var"])

    if args.all:
        classes = [c["id"] for c in config["uspc_classes"]]
    else:
        classes = [args.uspc_class]

    sample_size = config["positive_per_class"]

    summaries = []
    for cls in classes:
        try:
            summary = pull_class(
                config=config,
                uspc_class=cls,
                api_key=api_key,
                sample_size=sample_size,
                dry_run=args.dry_run,
            )
            summaries.append(summary)
        except Exception as e:
            logger.error(f"Class {cls} failed: {e}", exc_info=True)
            summaries.append({"class": cls, "error": str(e)})

    console.rule("[bold]Pull summary[/bold]")
    for s in summaries:
        console.print(s)


if __name__ == "__main__":
    main()
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
    # Request body — field names confirmed from live API response May 2026.
    #
    # Key confirmed fields:
    #   applicationMetaData.class         — USPC class, e.g. "D8", "D24"
    #   applicationMetaData.patentNumber  — patent number, e.g. "D1055675"
    #   applicationMetaData.grantDate     — grant date
    #   applicationMetaData.inventionTitle — title
    #   grantDocumentMetaData.fileLocationURI — XML file with drawings
    # -------------------------------------------------------------------------
    body = {
        "q": "applicationMetaData.applicationTypeLabelName:Design",
        "filters": [
            {
                # Granted and in-force only
                "name": "applicationMetaData.applicationStatusDescriptionText",
                "value": ["Patented Case"],
            },
            {
                # USPC class — confirmed field name from live response
                # Values: "D6", "D8", "D9", "D12", "D14", "D23", "D24", "D26"
                # Note: API stores without leading zero ("D8" not "D08")
                "name": "applicationMetaData.class",
                "value": [uspc_class],
            },
        ],
        "rangeFilters": [
            {
                "field": "applicationMetaData.grantDate",
                "valueFrom": grant_date_start,
                "valueTo": grant_date_end,
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _download_drawing(patent: dict, marker_path: Path, api_key: str) -> bool:
    """
    Download the patent PDF from image-ppubs.uspto.gov and save it.

    Confirmed public endpoint (no auth required):
        https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/D1051385

    Returns a multi-page PDF containing all patent drawings.
    The PDF is saved as <patent_number>.pdf and normalize.py (module 4)
    will extract page 2 (first drawing figure) as a PNG.

    This approach is simpler and more reliable than trying to get TIF files
    from the bulk data system, which requires signed URLs.
    """
    patent_number = _patent_number_field(patent)
    if not patent_number:
        logger.warning("  no patent number found in record")
        return False

    out_dir = marker_path.parent

    # image-ppubs.uspto.gov is a public endpoint — no API key needed
    url = f"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{patent_number}"
    logger.info(f"  fetching PDF: {patent_number}")

    response = requests.get(url, timeout=60, allow_redirects=True)

    if response.status_code != 200:
        logger.warning(f"  PDF fetch failed: HTTP {response.status_code} — {url}")
        marker_path.write_text(f"pdf_download_failed:{response.status_code}")
        return False

    content = response.content

    # Sanity check — a real patent PDF is at minimum ~50KB
    if len(content) < 50_000:
        logger.warning(f"  PDF too small ({len(content):,} bytes) for {patent_number}")
        marker_path.write_text(f"pdf_too_small:{len(content)}")
        return False

    # Verify it's actually a PDF
    if not content.startswith(b"%PDF"):
        logger.warning(f"  response is not a PDF for {patent_number}")
        marker_path.write_text(f"not_a_pdf:{content[:80]}")
        return False

    pdf_path = out_dir / f"{patent_number}.pdf"
    pdf_path.write_bytes(content)
    logger.info(f"  saved {patent_number}.pdf ({len(content):,} bytes)")

    marker_path.write_text(f"pdf:{patent_number}.pdf")
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
    Extract the patent number. Confirmed field: applicationMetaData.patentNumber
    Returns e.g. "D1055675". Falls back to applicationNumberText if needed.
    """
    meta = patent.get("applicationMetaData", {})
    return (
        meta.get("patentNumber")
        or patent.get("applicationNumberText")
    )


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

        # Idempotency: skip if marker file or image already on disk
        marker_path = out_dir / f"{patent_number}.txt"   # created by _download_drawing
        json_path = out_dir / f"{patent_number}.json"
        if marker_path.exists() and json_path.exists():
            logger.info(f"  • {patent_number} already on disk, skipping")
            downloaded += 1
            continue

        # Save metadata sidecar — includes full API record for traceability
        # Key fields for manifest: patentNumber, grantDate, class, inventionTitle
        meta = patent.get("applicationMetaData", {})
        sidecar = {
            "patent_number": patent_number,
            "application_number": patent.get("applicationNumberText"),
            "grant_date": meta.get("grantDate"),
            "filing_date": meta.get("filingDate"),
            "uspc_class": meta.get("class"),
            "uspc_symbol": meta.get("uspcSymbolText"),
            "invention_title": meta.get("inventionTitle"),
            "applicant": meta.get("firstApplicantName"),
            "drawing_xml_url": (patent.get("grantDocumentMetaData") or {}).get("fileLocationURI"),
            "label": "positive",
            "source_type": "in_force_design_patent",
            "raw_record": patent,   # full record for debugging
        }
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))

        # Download the drawing
        ok = _download_drawing(patent, marker_path, api_key)
        if ok:
            logger.info(f"  ✓ {patent_number}")
            downloaded += 1
        else:
            failures.append(patent_number)
            json_path.unlink(missing_ok=True)

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
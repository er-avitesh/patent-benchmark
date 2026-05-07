"""
02_pull_negative_expired.py — negative-class puller (expired patents).

Pulls US design patents whose terms have EXPIRED, making them public domain.
These form the "known-not-patented" negative class for the benchmark.

WHY EXPIRED PATENTS AS NEGATIVES
---------------------------------
A drawing from an expired patent is definitively NOT protected by any active
US design patent. This gives us a clean, verifiable negative class with the
same visual style (USPTO line drawings) as the positive class — meaning models
cannot distinguish classes by drawing style, only by the visual design itself.

TERM LOGIC
----------
Pre-2015 design patents: 14-year term from grant date.
Post-2015 design patents: 15-year term from grant date.
Safe cutoff: patents granted before 2008-01-01 are expired as of 2026
regardless of which term applies (18+ years ago).

USAGE
-----
    # Dry run — verify results look like older patents
    python scripts/02_pull_negative_expired.py --dry-run --class D24

    # Pull one class
    python scripts/02_pull_negative_expired.py --class D24

    # Pull all 8 classes (~20 min)
    python scripts/02_pull_negative_expired.py --all

OUTPUTS
-------
For each expired patent (in data/raw/negative_expired/):
    <patent_number>.pdf   — full patent PDF with all drawing sheets
    <patent_number>.json  — metadata sidecar (label: "negative")
    <patent_number>.txt   — marker file
    <patent_number>.xml   — raw USPTO grant XML
"""

from __future__ import annotations

import argparse
import json
import random
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

logger = get_logger("pull_negative_expired")


# =============================================================================
# Patent search — identical to module 1 except date window
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _search_patents_for_class(
    config: dict,
    uspc_class: str,
    grant_date_end: str,
    api_key: str,
    page_size: int = 100,
) -> list[dict]:
    """
    Query USPTO for expired design patents in one USPC class.

    Key difference from module 1: no grant_date_start — we want everything
    before the expiry cutoff. The API rangeFilter requires both valueFrom and
    valueTo, so we use "1900-01-01" as the open lower bound.
    """
    base_url = config["uspto_api"]["base_url"].rstrip("/")
    endpoint = config["uspto_api"]["patent_search_endpoint"]
    url = f"{base_url}{endpoint}"

    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    body = {
        "q": "applicationMetaData.applicationTypeLabelName:Design",
        "filters": [
            {
                # "Patented Case" is still the status even for expired patents
                # in USPTO records — the patent was granted, which is what
                # matters for our records. We verify expiry via grant date.
                "name": "applicationMetaData.applicationStatusDescriptionText",
                "value": ["Patented Case"],
            },
            {
                "name": "applicationMetaData.class",
                "value": [uspc_class],
            },
        ],
        "rangeFilters": [
            {
                "field": "applicationMetaData.grantDate",
                "valueFrom": "1976-01-01",   # USPTO electronic records start here
                "valueTo": grant_date_end,    # e.g. "2007-12-31" — all expired
            }
        ],
        "sort": [{"field": "applicationMetaData.grantDate", "order": "desc"}],
        "pagination": {"offset": 0, "limit": page_size},
    }

    all_results: list[dict] = []
    MAX_RESULTS_PER_CLASS = 500

    while True:
        current_offset = body["pagination"]["offset"]
        logger.info(
            f"  → querying {uspc_class} (expired), offset={current_offset}, "
            f"have {len(all_results)} so far"
        )
        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=config["uspto_api"]["request_timeout_seconds"],
        )

        if response.status_code == 400:
            logger.error(f"  400 Bad Request:\n{response.text[:2000]}")
        response.raise_for_status()
        data = response.json()

        page_results = (
            data.get("patentFileWrapperDataBag")
            or data.get("patents")
            or data.get("results")
            or []
        )

        if current_offset == 0:
            logger.info(f"  Response top-level keys: {list(data.keys())}")
            if page_results:
                meta = page_results[0].get("applicationMetaData", {})
                logger.info(
                    f"  First record: patent={meta.get('patentNumber')}, "
                    f"grantDate={meta.get('grantDate')}, "
                    f"class={meta.get('class')}"
                )

        if not page_results:
            break

        all_results.extend(page_results)

        if len(all_results) >= MAX_RESULTS_PER_CLASS:
            logger.info(f"  reached cap of {MAX_RESULTS_PER_CLASS}, stopping")
            all_results = all_results[:MAX_RESULTS_PER_CLASS]
            break

        if len(page_results) < page_size:
            break

        body["pagination"]["offset"] += page_size
        time.sleep(60.0 / config["uspto_api"]["rate_limit_per_minute"])

    return all_results


# =============================================================================
# PDF download — identical to module 1
# =============================================================================


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def _download_pdf(patent_number: str, out_dir: Path, marker_path: Path) -> bool:
    """Download the patent PDF from image-ppubs.uspto.gov."""
    url = f"https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/{patent_number}"
    logger.info(f"  fetching PDF: {patent_number}")

    response = requests.get(url, timeout=60, allow_redirects=True)

    if response.status_code != 200:
        logger.warning(f"  PDF fetch failed: HTTP {response.status_code}")
        marker_path.write_text(f"pdf_download_failed:{response.status_code}")
        return False

    content = response.content

    if len(content) < 50_000:
        logger.warning(f"  PDF too small ({len(content):,} bytes)")
        marker_path.write_text(f"pdf_too_small:{len(content)}")
        return False

    if not content.startswith(b"%PDF"):
        logger.warning(f"  response is not a PDF for {patent_number}")
        marker_path.write_text(f"not_a_pdf")
        return False

    pdf_path = out_dir / f"{patent_number}.pdf"
    pdf_path.write_bytes(content)
    logger.info(f"  saved {patent_number}.pdf ({len(content):,} bytes)")

    marker_path.write_text(f"pdf:{patent_number}.pdf")
    return True


# =============================================================================
# Helpers
# =============================================================================


def _patent_number_field(patent: dict) -> str | None:
    meta = patent.get("applicationMetaData", {})
    return meta.get("patentNumber") or patent.get("applicationNumberText")


def _select_random_sample(patents: list[dict], n: int, seed: int) -> list[dict]:
    # Use a different seed offset from module 1 to ensure independent sampling.
    # Module 1 uses seed=42; module 2 uses seed=43 so the two samples don't
    # accidentally draw from the same underlying random sequence.
    rng = random.Random(seed + 1)
    if len(patents) < n:
        logger.warning(f"  only {len(patents)} patents available, wanted {n}")
        return patents
    return rng.sample(patents, n)


# =============================================================================
# Per-class orchestration
# =============================================================================


def pull_class(
    config: dict,
    uspc_class: str,
    api_key: str,
    sample_size: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Pull expired-patent negatives for one USPC class."""
    seed = config["random_seed"]
    # Use the expired_grant_end date from config
    grant_date_end = config["date_windows"]["expired_grant_end"]

    logger.info(f"[bold yellow]Class {uspc_class}[/bold yellow] — searching (expired, pre-{grant_date_end})")
    patents = _search_patents_for_class(
        config=config,
        uspc_class=uspc_class,
        grant_date_end=grant_date_end,
        api_key=api_key,
    )
    logger.info(f"  found {len(patents)} expired patents matching filters")

    if dry_run:
        if patents:
            rec = patents[0]
            meta = rec.get("applicationMetaData", {})
            console.print(f"\n[bold green]✓ Got {len(patents)} records[/bold green]")
            console.print(f"  Sample: {meta.get('patentNumber')} — "
                          f"{meta.get('inventionTitle')} — "
                          f"granted {meta.get('grantDate')}")
            console.print(f"  USPC class: {meta.get('class')}")
            console.print(f"  [dim]This patent expired in approx. "
                          f"{int(meta.get('grantDate','2000')[:4]) + 14}[/dim]")
        else:
            console.print("[red]No results — check date window in config.yaml[/red]")
        return {"class": uspc_class, "found": len(patents), "downloaded": 0, "dry_run": True}

    # Oversample by 25%
    target_n = int(sample_size * 1.25)
    sampled = _select_random_sample(patents, target_n, seed)

    out_dir = resolve_path(config["paths"]["raw_negative_expired"])
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failures: list[str] = []

    for patent in sampled:
        if downloaded >= sample_size:
            break

        patent_number = _patent_number_field(patent)
        if not patent_number:
            logger.warning("  skipping record without patent number")
            continue

        marker_path = out_dir / f"{patent_number}.txt"
        json_path = out_dir / f"{patent_number}.json"

        if marker_path.exists() and json_path.exists():
            logger.info(f"  • {patent_number} already on disk, skipping")
            downloaded += 1
            continue

        # Metadata sidecar — same structure as module 1, label = "negative"
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
            "label": "negative",
            "source_type": "expired_design_patent",
            "expiry_note": f"Granted {meta.get('grantDate')} — past 14/15yr term as of 2026",
            "raw_record": patent,
        }
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))

        ok = _download_pdf(patent_number, out_dir, marker_path)
        if ok:
            logger.info(f"  ✓ {patent_number}")
            downloaded += 1
        else:
            failures.append(patent_number)
            json_path.unlink(missing_ok=True)

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
    parser = argparse.ArgumentParser(
        description="Pull expired design patents (negative class) from USPTO."
    )
    parser.add_argument("--class", dest="uspc_class", help="USPC class to pull (e.g., D24)")
    parser.add_argument("--all", action="store_true", help="Pull all 8 classes from config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search only, no downloads, print sample record")
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

    sample_size = config["negative_composition"]["expired_patent_count"] // len(
        config["uspc_classes"]
    )
    logger.info(f"Target per class: {sample_size} expired patents")

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

    console.rule("[bold]Pull summary — expired negatives[/bold]")
    for s in summaries:
        console.print(s)
    total = sum(s.get("downloaded", 0) for s in summaries)
    console.print(f"\n[bold green]Total downloaded: {total}[/bold green]")


if __name__ == "__main__":
    main()
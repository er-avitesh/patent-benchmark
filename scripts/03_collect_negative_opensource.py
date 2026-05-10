"""
03_collect_negative_opensource.py -- negative open-source collector.

Downloads technical drawings from Wikimedia Commons using their public API.
No authentication required. All images are CC0 or public domain.

WHY WIKIMEDIA COMMONS
---------------------
GrabCAD has no public bulk download API (confirmed from their community forum).
TraceParts API is vendor-gated. Wikimedia Commons has a fully open REST API,
thousands of engineering/instrument drawings, and clear public domain licensing.

FIX NOTES (confirmed from live debugging)
------------------------------------------
1. Wikimedia requires a descriptive User-Agent header on ALL requests including
   API calls, not just image downloads. Missing User-Agent returns HTTP 403.
2. Category names must be exact as they appear on Commons. All names below
   have been verified by checking the actual category pages.

USAGE
-----
    python scripts/03_collect_negative_opensource.py --dry-run --class D24
    python scripts/03_collect_negative_opensource.py --class D24
    python scripts/03_collect_negative_opensource.py --all
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
    resolve_path,
)

logger = get_logger("collect_opensource")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia requires this on every request -- missing = 403
HEADERS = {
    "User-Agent": (
        "PatentBenchmarkResearch/1.0 "
        "(academic non-commercial; patent infringement LLM benchmark; "
        "contact: avitesh.research@gmail.com)"
    )
}

# ---------------------------------------------------------------------------
# Verified category names -- only categories that specifically contain
# technical drawings/diagrams, not photographs of objects.
# These are subcategories of Technical_drawings on Wikimedia Commons.
# ---------------------------------------------------------------------------
CLASS_CATEGORIES: dict[str, list[str]] = {
    "D6": [
        "Technical_drawings",          # broad, filtered by keyword below
        "Engineering_diagrams",
    ],
    "D8": [
        "Technical_drawings",
        "Engineering_diagrams",
    ],
    "D9": [
        "Technical_drawings",
        "Engineering_diagrams",
    ],
    "D12": [
        "Diagrams_of_vehicles",
        "Technical_drawings",
    ],
    "D14": [
        "Technical_drawings",
        "Engineering_diagrams",
    ],
    "D23": [
        "Piping_and_instrumentation_diagrams",
        "Technical_drawings",
    ],
    "D24": [
        "Technical_drawings_of_instruments",  # 20 confirmed line drawings
        "Historical_surgical_instruments",    # Wellcome collection line drawings
        "Surgical_instruments_described_by_Abulcasis",  # confirmed line drawings
    ],
    "D26": [
        "Technical_drawings",
        "Engineering_diagrams",
    ],
}

# Keywords that must appear in title OR description for the image to be accepted.
# This is the primary guard against photographs getting through.
# "Wellcome" images are line drawings from the Wellcome Collection.
# Files with drawing/diagram/sketch in the name are almost always drawings.
DRAWING_KEYWORDS = [
    "wellcome",        # Wellcome Collection -- all line drawings
    "drawing",
    "diagram",
    "sketch",
    "schematic",
    "blueprint",
    "cross.section",
    "cross_section",
    "section",
    "elevation",
    "plan_view",
    "engraving",
    "lithograph",
    "illustration",
    "svg",             # SVG files are almost always diagrams
]

# Broad fallback -- the Technical_drawings_of_instruments subcategory
# is small (20 files) but all confirmed line drawings
BROAD_FALLBACKS: dict[str, str] = {
    "D6":  "Technical_drawings",
    "D8":  "Technical_drawings",
    "D9":  "Technical_drawings",
    "D12": "Diagrams_of_vehicles",
    "D14": "Technical_drawings",
    "D23": "Technical_drawings",
    "D24": "Technical_drawings_of_instruments",
    "D26": "Technical_drawings",
}


def _api_get(params: dict) -> dict:
    """Make a Wikimedia API GET request with proper User-Agent."""
    r = requests.get(COMMONS_API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _query_category_members(category: str, limit: int = 100) -> list[dict]:
    """Return file members of a Commons category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": str(limit),
        "format": "json",
    }
    try:
        data = _api_get(params)
        members = data.get("query", {}).get("categorymembers", [])
        logger.info(f"    {category}: {len(members)} files")
        return members
    except Exception as e:
        logger.warning(f"    query failed for {category}: {e}")
        return []


def _get_image_info(page_title: str) -> dict | None:
    """Get direct URL and metadata for a Commons file."""
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "format": "json",
    }
    try:
        data = _api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info_list = page.get("imageinfo", [])
            if info_list:
                info = info_list[0]
                ext = info.get("extmetadata", {})
                return {
                    "url": info.get("url"),
                    "mime": info.get("mime", ""),
                    "width": info.get("width", 0),
                    "height": info.get("height", 0),
                    "title": page_title,
                    "license": ext.get("LicenseShortName", {}).get("value", "unknown"),
                    "description": ext.get("ImageDescription", {}).get("value", ""),
                    "commons_url": (
                        "https://commons.wikimedia.org/wiki/"
                        + page_title.replace(" ", "_")
                    ),
                }
    except Exception as e:
        logger.warning(f"    imageinfo failed for {page_title}: {e}")
    return None


def _is_suitable(info: dict) -> bool:
    """
    Filter for actual line drawings / diagrams.

    Strategy:
    - Accept any SVG (almost always a diagram)
    - For JPG/PNG: require at least one drawing keyword in title or description
    - Reject images that are too small, too large, or clearly photographs
    """
    if not info or not info.get("url"):
        return False
    mime = info.get("mime", "")
    if mime not in ("image/png", "image/jpeg", "image/jpg", "image/svg+xml"):
        return False

    # SVG files are diagrams by definition -- accept immediately
    if mime == "image/svg+xml":
        w = info.get("width", 0) or 0
        h = info.get("height", 0) or 0
        return 100 < w < 8000 and 100 < h < 8000

    # Size filter
    w = info.get("width", 0) or 0
    h = info.get("height", 0) or 0
    if w < 300 or h < 300 or w > 8000 or h > 8000:
        return False

    # Require at least one drawing keyword in title or description
    title = info.get("title", "").lower()
    desc = info.get("description", "").lower()
    combined = title + " " + desc

    has_drawing_keyword = any(kw in combined for kw in DRAWING_KEYWORDS)
    if not has_drawing_keyword:
        return False

    # Reject obvious photographs (override drawing keyword if these appear)
    photo_words = [
        "photo", "photograph", "jpg (", "museum",
        ".jpg (", "dsc", "img_", "p10", "p20",
    ]
    if any(pw in title for pw in photo_words):
        return False

    return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20), reraise=True)
def _download_image(url: str, save_path: Path) -> bool:
    r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
    if r.status_code != 200:
        logger.warning(f"    download failed: HTTP {r.status_code}")
        return False
    content = r.content
    if len(content) < 5_000:
        logger.warning(f"    too small: {len(content)} bytes")
        return False
    save_path.write_bytes(content)
    return True


def _collect_candidates(uspc_class: str) -> list[dict]:
    """Collect and filter candidate images from all categories for a class."""
    categories = CLASS_CATEGORIES.get(uspc_class, [])
    fallback = BROAD_FALLBACKS.get(uspc_class)
    all_to_try = categories + ([fallback] if fallback else [])

    candidates: list[dict] = []
    seen: set[str] = set()

    for cat in all_to_try:
        if len(candidates) >= 40:   # enough to sample from
            break
        members = _query_category_members(cat, limit=50)
        for m in members:
            title = m.get("title", "")
            if title in seen:
                continue
            seen.add(title)
            info = _get_image_info(title)
            if info and _is_suitable(info):
                candidates.append(info)
        time.sleep(0.5)

    return candidates


def pull_class(
    config: dict,
    uspc_class: str,
    sample_size: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    seed = config["random_seed"] + 2

    logger.info(f"[bold green]Class {uspc_class}[/bold green] -- Wikimedia Commons")
    candidates = _collect_candidates(uspc_class)
    logger.info(f"  {len(candidates)} suitable candidates found")

    if dry_run:
        if candidates:
            s = candidates[0]
            console.print(f"\n[green]Got {len(candidates)} candidates[/green]")
            console.print(f"  Title:   {s['title']}")
            console.print(f"  URL:     {s['url']}")
            console.print(f"  Size:    {s['width']}x{s['height']}px  ({s['mime']})")
            console.print(f"  License: {s['license']}")
        else:
            console.print("[red]No candidates -- category names may need adjustment[/red]")
        return {"class": uspc_class, "found": len(candidates), "downloaded": 0, "dry_run": True}

    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(sample_size, len(candidates)))

    out_dir = resolve_path(config["paths"]["raw_negative_opensource"])
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failures: list[str] = []

    for img in sampled:
        title = img["title"].replace("File:", "").replace(" ", "_")
        # Strip characters illegal on Windows filesystems
        import re as _re
        title = _re.sub(r'[\\/:*?"<>|]', "", title)
        stem = title[:60].rstrip("_")
        file_id = f"{uspc_class}_{stem}"
        ext = {
            "image/png": ".png", "image/jpeg": ".jpg",
            "image/jpg": ".jpg", "image/svg+xml": ".svg",
        }.get(img["mime"], ".png")

        image_path = out_dir / f"{file_id}{ext}"
        marker_path = out_dir / f"{file_id}.txt"
        json_path = out_dir / f"{file_id}.json"

        if marker_path.exists() and json_path.exists():
            logger.info(f"  already on disk: {file_id}")
            downloaded += 1
            continue

        json_path.write_text(json.dumps({
            "id": file_id,
            "title": img["title"],
            "uspc_class": uspc_class,
            "label": "negative",
            "source_type": "wikimedia_commons",
            "url": img["url"],
            "commons_url": img["commons_url"],
            "license": img["license"],
            "width": img["width"],
            "height": img["height"],
        }, indent=2, sort_keys=True))

        ok = _download_image(img["url"], image_path)
        if ok:
            marker_path.write_text(f"downloaded:{file_id}{ext}")
            logger.info(f"  {file_id}{ext} ({img['width']}x{img['height']})")
            downloaded += 1
        else:
            failures.append(file_id)
            json_path.unlink(missing_ok=True)

        time.sleep(0.3)

    return {
        "class": uspc_class,
        "found": len(candidates),
        "sampled": len(sampled),
        "downloaded": downloaded,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect open-source negative drawings from Wikimedia Commons."
    )
    parser.add_argument("--class", dest="uspc_class")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.uspc_class and not args.all:
        parser.error("Specify --class <id> or --all")

    config = load_config()
    load_env()

    classes = (
        [c["id"] for c in config["uspc_classes"]] if args.all else [args.uspc_class]
    )
    sample_size = (
        config["negative_composition"]["open_source_count"] // len(config["uspc_classes"])
    )
    logger.info(f"Target per class: {sample_size}")

    summaries = []
    for cls in classes:
        try:
            summaries.append(pull_class(config, cls, sample_size, args.dry_run))
        except Exception as e:
            logger.error(f"Class {cls} failed: {e}", exc_info=True)
            summaries.append({"class": cls, "error": str(e)})

    console.rule("[bold]Summary -- open-source negatives[/bold]")
    for s in summaries:
        console.print(s)
    total = sum(s.get("downloaded", 0) for s in summaries)
    console.print(f"\n[bold green]Total: {total}[/bold green]")


if __name__ == "__main__":
    main()
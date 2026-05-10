"""
05_build_manifest.py -- build data/manifest.csv from all sidecar files.

Reads:
  data/raw/positive/<id>.json          + <id>.xml
  data/raw/negative_expired/<id>.json  + <id>.xml
  data/raw/negative_opensource/<id>.json
  data/processed/<id>.json             (normalization record)

Writes:
  data/manifest.csv   -- one row per drawing, all columns needed for modules 6-9

COLUMNS
-------
  id                  patent number or open-source image ID
  label               "positive" or "negative"
  source_type         "positive_patent" | "expired_patent" | "wikimedia_commons"
  uspc_class          D6, D8, D9, D12, D14, D23, D24, D26
  locarno_class       e.g. "2401" (from XML) or "" for open-source
  grant_date          YYYY-MM-DD or "" for open-source
  invention_title     from USPTO metadata or "" for open-source
  png_path            relative path to data/processed/<id>.png
  drawing_page        page number FIG.1 was extracted from (PDFs only)
  text_masked         true/false

USAGE
-----
    python scripts/05_build_manifest.py
    python scripts/05_build_manifest.py --check   # validate all PNGs exist
"""

from __future__ import annotations

import argparse
import csv
import random
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from _common import console, get_logger, load_config, load_env, resolve_path

logger = get_logger("build_manifest")

MANIFEST_COLUMNS = [
    "id",
    "label",
    "source_type",
    "uspc_class",
    "locarno_class",
    "grant_date",
    "invention_title",
    "png_path",
    "drawing_page",
    "text_masked",
]


# ---------------------------------------------------------------------------
# Locarno extraction from USPTO grant XML
# ---------------------------------------------------------------------------

def _extract_locarno(xml_path: Path) -> str:
    """
    Parse the USPTO grant XML to extract the Locarno classification code.

    Confirmed XML structure (from D1049406.xml):
      <classification-locarno>
        <edition>14</edition>
        <main-classification>2401</main-classification>   <-- we want this
      </classification-locarno>
      <classification-national>
        <main-classification>D24214</main-classification> <-- NOT this
      </classification-national>

    Strategy: find <classification-locarno> element, then get its
    <main-classification> child directly. This avoids matching the
    national classification which has a different format (e.g. "D24214").
    """
    if not xml_path.exists():
        return ""
    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
        # Find classification-locarno element (no namespace in USPTO XMLs)
        locarno_elem = root.find(".//classification-locarno")
        if locarno_elem is not None:
            main = locarno_elem.find("main-classification")
            if main is not None and main.text:
                return main.text.strip()
    except Exception as e:
        logger.warning(f"  XML parse failed for {xml_path.name}: {e}")
    return ""


# ---------------------------------------------------------------------------
# Row builders per source type
# ---------------------------------------------------------------------------

def _row_from_patent(
    raw_json: Path,
    raw_xml: Path,
    processed_json: Path,
    processed_dir: Path,
    label: str,
    source_type: str,
) -> dict | None:
    """Build a manifest row from a patent JSON + XML + processed JSON."""
    try:
        meta = json.loads(raw_json.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  could not read {raw_json}: {e}")
        return None

    try:
        proc = json.loads(processed_json.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  could not read {processed_json}: {e}")
        proc = {}

    patent_id = meta.get("patent_number") or raw_json.stem
    locarno = _extract_locarno(raw_xml)

    # If XML missing or Locarno not found, derive main class from USPC class.
    # USPC D6->06, D8->08, D9->09, D12->12, D14->14, D23->23, D24->24, D26->26.
    # Subclass digits (last 2) require the XML; we use "00" as placeholder.
    if not locarno:
        uspc = meta.get("uspc_class", "")
        digits = uspc.lstrip("D").lstrip("0") or ""
        locarno = f"{int(digits):02d}00" if digits.isdigit() else ""

    png_path = processed_dir / f"{patent_id}.png"

    return {
        "id": patent_id,
        "label": label,
        "source_type": source_type,
        "uspc_class": meta.get("uspc_class", ""),
        "locarno_class": locarno,
        "grant_date": meta.get("grant_date", ""),
        "invention_title": (meta.get("invention_title") or "").replace("\n", " ").strip(),
        "png_path": str(png_path.relative_to(png_path.parents[2])),
        "drawing_page": proc.get("drawing_page_number", ""),
        "text_masked": proc.get("text_masked", ""),
    }


def _row_from_opensource(
    raw_json: Path,
    processed_json: Path,
    processed_dir: Path,
) -> dict | None:
    """Build a manifest row from an open-source image JSON + processed JSON."""
    try:
        meta = json.loads(raw_json.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"  could not read {raw_json}: {e}")
        return None

    try:
        proc = json.loads(processed_json.read_text(encoding="utf-8"))
    except Exception as e:
        proc = {}

    item_id = meta.get("id") or raw_json.stem
    png_path = processed_dir / f"{item_id}.png"

    return {
        "id": item_id,
        "label": "negative",
        "source_type": "wikimedia_commons",
        "uspc_class": meta.get("uspc_class", ""),
        "locarno_class": "",
        "grant_date": "",
        "invention_title": meta.get("title", ""),
        "png_path": str(png_path.relative_to(png_path.parents[2])),
        "drawing_page": "",
        "text_masked": proc.get("text_masked", False),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_manifest(config: dict, check: bool = False) -> Path:
    raw_positive   = resolve_path(config["paths"]["raw_positive"])
    raw_expired    = resolve_path(config["paths"]["raw_negative_expired"])
    raw_opensource = resolve_path(config["paths"]["raw_negative_opensource"])
    processed_dir  = resolve_path(config["paths"]["processed"])
    manifest_path  = resolve_path(config["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing_pngs: list[str] = []

    # --- Positive patents ---
    logger.info("Reading positive patents...")
    for jf in sorted(raw_positive.glob("*.json")):
        xf = jf.with_suffix(".xml")
        pf = processed_dir / f"{jf.stem}.json"
        row = _row_from_patent(jf, xf, pf, processed_dir,
                                label="positive",
                                source_type="positive_patent")
        if row:
            rows.append(row)

    # --- Negative expired patents (stratified: cap at expired_patent_count) ---
    logger.info("Reading negative expired patents...")
    expired_cap = config.get("negative_composition", {}).get("expired_patent_count", 999)
    expired_per_class = expired_cap // len(config.get("uspc_classes", [1]*8))
    expired_class_counts: dict = {}
    rng_expired = random.Random(config.get("random_seed", 42) + 5)
    all_expired_jsons = sorted(raw_expired.glob("*.json"))
    rng_expired.shuffle(all_expired_jsons)
    for jf in all_expired_jsons:
        xf = jf.with_suffix(".xml")
        pf = processed_dir / f"{jf.stem}.json"
        row = _row_from_patent(jf, xf, pf, processed_dir,
                                label="negative",
                                source_type="expired_patent")
        if row:
            cls = row.get("uspc_class", "")
            if expired_class_counts.get(cls, 0) >= expired_per_class:
                continue
            expired_class_counts[cls] = expired_class_counts.get(cls, 0) + 1
            rows.append(row)
    logger.info(f"  expired patents after stratification: {len([r for r in rows if r["source_type"]=="expired_patent"])} (cap {expired_per_class}/class)")

    # --- Negative open-source ---
    logger.info("Reading negative open-source images...")
    for jf in sorted(raw_opensource.glob("*.json")):
        pf = processed_dir / f"{jf.stem}.json"
        row = _row_from_opensource(jf, pf, processed_dir)
        if row:
            rows.append(row)

    logger.info(f"Total rows: {len(rows)}")

    # --- Validation ---
    for row in rows:
        png = resolve_path(row["png_path"])
        if not png.exists():
            missing_pngs.append(row["id"])

    if missing_pngs:
        logger.warning(f"  {len(missing_pngs)} missing PNGs: {missing_pngs[:10]}")
    else:
        logger.info("  all PNGs present")

    # --- Write CSV ---
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote {manifest_path}")

    # --- Summary ---
    positives = sum(1 for r in rows if r["label"] == "positive")
    negatives = sum(1 for r in rows if r["label"] == "negative")
    classes   = sorted(set(r["uspc_class"] for r in rows if r["uspc_class"]))

    console.rule("[bold]Manifest summary[/bold]")
    console.print(f"  Total rows    : {len(rows)}")
    console.print(f"  Positive      : {positives}")
    console.print(f"  Negative      : {negatives}")
    console.print(f"  USPC classes  : {', '.join(classes)}")
    console.print(f"  Missing PNGs  : {len(missing_pngs)}")
    console.print(f"  Output        : {manifest_path}")

    if check:
        console.print("\n[bold]Sample rows:[/bold]")
        for row in rows[:3]:
            console.print(f"  {row['id']:20s}  {row['label']:8s}  {row['uspc_class']:4s}  "
                          f"{row['locarno_class']:6s}  {row['grant_date']:12s}  "
                          f"{row['invention_title'][:40]}")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build data/manifest.csv from all sidecar files."
    )
    parser.add_argument("--check", action="store_true",
                        help="Validate all PNGs exist and print sample rows")
    args = parser.parse_args()

    config = load_config()
    load_env()
    build_manifest(config, check=args.check)


if __name__ == "__main__":
    main()
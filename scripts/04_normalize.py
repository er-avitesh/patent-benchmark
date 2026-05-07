"""
04_normalize.py -- normalize all raw drawings to 1024x1024 grayscale PNG.

CONFIRMED FROM LIVE DATA (D1049406, 11 pages):
- PDFs are pure raster scans -- get_text() returns nothing on any page
- Page structure varies: D1049406 has multiple front-matter pages before drawings
- Drawing pages are identified by image density: one large drawing per page,
  lots of white space, ink coverage < 15% of the page area
- Front matter pages (bibliographic, references) have dense text blocks:
  ink coverage > 15%, or contain many small image regions

PAGE DETECTION STRATEGY:
  For each page starting from page 1 (skip page 0 front page):
    1. Render at 72 DPI (fast, just for analysis)
    2. Convert to grayscale
    3. Measure ink coverage (% of non-white pixels)
    4. Drawing pages: ink < 15%, content is centered, sparse
    5. Take the first page that looks like a drawing sheet

TEXT MASKING:
  Confidence threshold raised from 40 to 70 to avoid masking drawing lines.
  Only mask tokens that are plausibly patent metadata (numbers, "FIG", "Sheet").

USAGE
-----
    python scripts/04_normalize.py --dry-run
    python scripts/04_normalize.py --id D1049406
    python scripts/04_normalize.py --all
    python scripts/04_normalize.py --source positive
    python scripts/04_normalize.py --no-mask   # skip Tesseract, faster
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Raise PIL pixel limit before any PIL import
try:
    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = 400_000_000
except Exception:
    pass

from _common import console, get_logger, load_config, load_env, resolve_path

logger = get_logger("normalize")

TARGET_SIZE = 1024
BACKGROUND = 255  # white


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _fitz():
    try:
        import fitz
        return fitz
    except ImportError:
        logger.error("PyMuPDF not installed: pip install PyMuPDF")
        sys.exit(1)

def _pil():
    try:
        from PIL import Image, ImageDraw
        return Image, ImageDraw
    except ImportError:
        logger.error("Pillow not installed: pip install Pillow")
        sys.exit(1)

def _tesseract():
    try:
        import pytesseract
        # Quick check that tesseract binary is reachable
        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception:
        return None

def _cairosvg():
    try:
        import cairosvg
        return cairosvg
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# PDF page analysis
# ---------------------------------------------------------------------------

def _ink_coverage(pil_img) -> float:
    """
    Fraction of pixels that are not white (ink coverage).
    Used to distinguish drawing pages (sparse) from text pages (dense).
    """
    import struct
    Image, _ = _pil()
    gray = pil_img.convert("L")
    # Count pixels darker than 200 (ink) vs total
    pixels = list(gray.getdata())
    ink = sum(1 for p in pixels if p < 200)
    return ink / len(pixels)


def _render_pdf_page(pdf_path: Path, page_index: int, dpi: int = 150):
    """Render a single PDF page to PIL Image at given DPI."""
    fitz = _fitz()
    Image, _ = _pil()
    doc = fitz.open(str(pdf_path))
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    doc.close()
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def _largest_blob_fraction(pil_img) -> float:
    """
    Measure the size of the largest connected ink region as a fraction
    of total page area. Drawing pages have one large blob (the drawing).
    Text/reference pages have many small blobs (individual characters/words).
    Uses a simple flood-fill approach via bounding-box of all ink pixels.
    """
    import numpy as np
    gray = pil_img.convert("L")
    arr = np.array(gray)
    ink = arr < 200  # True where ink exists

    if not ink.any():
        return 0.0

    # Find bounding box of all ink -- crude but fast
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Bounding box area as fraction of page
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    page_h, page_w = arr.shape
    bbox_fraction = (bbox_h * bbox_w) / (page_h * page_w)

    # Also measure how "spread" the ink is vertically
    # Drawing: ink spread across full height (large figure)
    # Text page: ink concentrated in top portion (few lines of text)
    ink_rows = rows.sum()
    row_coverage = ink_rows / page_h

    return bbox_fraction * row_coverage


def _find_drawing_sheet_page(pdf_path: Path) -> int | None:
    """
    Find the 0-based page index of the first FIG.1 drawing sheet.

    USPTO design patent PDFs are pure raster -- no extractable text.

    KEY INSIGHT from D1049406 ink coverage data:
      Page 0 (front):      ink=0.036  (thumbnails + dense text)
      Page 1 (references): ink=0.006  (sparse text -- MISLEADS simple threshold)
      Pages 2-10 (drawings): ink=0.005-0.007

    Simple ink coverage fails because references pages are also sparse.

    BETTER SIGNAL: largest blob fraction.
    - Drawing pages: one large figure spanning most of the page -> large blob
    - References pages: scattered text words -> many tiny blobs, small bbox
    - Front page: dense grid of thumbnails -> moderate blob

    We pick the page (from index 1 onward) with the LARGEST blob fraction,
    which corresponds to the most spatially dominant single drawing.
    """
    fitz = _fitz()
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    doc.close()

    if n_pages == 0:
        return None
    if n_pages == 1:
        return 0
    if n_pages == 2:
        return 1

    try:
        import numpy as np
    except ImportError:
        logger.warning("  numpy not available -- using page 1 as fallback")
        return 1

    blob_scores: list[float] = []
    ink_scores: list[float] = []

    for i in range(n_pages):
        try:
            img = _render_pdf_page(pdf_path, i, dpi=72)
            blob = _largest_blob_fraction(img)
            ink = _ink_coverage(img)
            blob_scores.append(blob)
            ink_scores.append(ink)
        except Exception:
            blob_scores.append(0.0)
            ink_scores.append(0.0)

    logger.info(f"  ink per page:  {[f'{c:.3f}' for c in ink_scores]}")
    logger.info(f"  blob per page: {[f'{c:.3f}' for c in blob_scores]}")

    # Find the FIRST page with the highest blob score, starting from page 1.
    # (skip page 0 which is always the front/bibliographic page)
    # Using first-occurrence of max ensures we get FIG. 1 when multiple
    # drawing pages have similar blob scores.
    scores_from_1 = blob_scores[1:]
    max_score = max(scores_from_1)
    # Find first index in scores_from_1 that equals max_score
    best_local = next(i for i, s in enumerate(scores_from_1) if s >= max_score * 0.98)
    best_idx = 1 + best_local

    logger.info(f"  selected page {best_idx + 1} (blob={blob_scores[best_idx]:.3f})")
    return best_idx


# ---------------------------------------------------------------------------
# Text masking
# ---------------------------------------------------------------------------

# Patterns that indicate patent metadata text worth masking
_MASK_PATTERNS = re.compile(
    r"(sheet|fig\.?|u\.?s\.?|patent|des\.?|\d{6,})",
    re.IGNORECASE
)

def _mask_text_regions(pil_image):
    """
    OCR the image and white-out regions containing patent metadata text.
    Confidence threshold: 70 (high) to avoid masking drawing lines.
    Only masks tokens matching known patent metadata patterns.
    """
    tess = _tesseract()
    if tess is None:
        return pil_image

    Image, ImageDraw = _pil()

    try:
        data = tess.image_to_data(
            pil_image,
            output_type=tess.Output.DICT,
            config="--psm 6",
        )
    except Exception as e:
        logger.warning(f"  Tesseract OCR failed: {e} -- skipping text masking")
        return pil_image

    draw = ImageDraw.Draw(pil_image)
    masked_count = 0

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        # High confidence only, and only patent metadata patterns
        if not text or conf < 70:
            continue
        if not _MASK_PATTERNS.search(text):
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        pad = 6
        draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=BACKGROUND)
        masked_count += 1

    if masked_count:
        logger.info(f"  masked {masked_count} text regions")

    return pil_image


# ---------------------------------------------------------------------------
# Image normalization
# ---------------------------------------------------------------------------

def _to_grayscale_square_png(img, target: int = TARGET_SIZE):
    """Grayscale -> pad to square -> resize to target x target."""
    Image, _ = _pil()
    gray = img.convert("L")
    w, h = gray.size
    side = max(w, h)
    square = Image.new("L", (side, side), BACKGROUND)
    square.paste(gray, ((side - w) // 2, (side - h) // 2))
    return square.resize((target, target), Image.LANCZOS)


def _save_clean_png(img, out_path: Path):
    """Save as PNG with no metadata."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    Image, _ = _pil()
    clean = Image.open(buf)
    clean.load()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(str(out_path), format="PNG")


# ---------------------------------------------------------------------------
# Per-source processors
# ---------------------------------------------------------------------------

class NormResult(NamedTuple):
    id: str
    source: str
    success: bool
    drawing_page: int | None
    note: str


def _process_pdf(pdf_path: Path, out_dir: Path, mask_text: bool = True, force: bool = False) -> NormResult:
    item_id = pdf_path.stem
    out_png = out_dir / f"{item_id}.png"
    out_json = out_dir / f"{item_id}.json"

    if out_png.exists() and out_json.exists() and not force:
        return NormResult(item_id, "pdf", True, None, "already done")

    page_idx = _find_drawing_sheet_page(pdf_path)
    if page_idx is None:
        logger.warning(f"  {item_id}: no drawing sheet found")
        return NormResult(item_id, "pdf", False, None, "no drawing sheet found")

    logger.info(f"  {item_id}: drawing sheet on page {page_idx + 1} of {_page_count(pdf_path)}")

    try:
        img = _render_pdf_page(pdf_path, page_idx, dpi=150)
    except Exception as e:
        logger.warning(f"  {item_id}: render failed: {e}")
        return NormResult(item_id, "pdf", False, page_idx, str(e))

    if mask_text:
        img = _mask_text_regions(img)

    img = _to_grayscale_square_png(img)
    _save_clean_png(img, out_png)

    out_json.write_text(json.dumps({
        "id": item_id,
        "source": "pdf",
        "drawing_page_index": page_idx,
        "drawing_page_number": page_idx + 1,
        "text_masked": mask_text,
        "output_size": TARGET_SIZE,
    }, indent=2))

    return NormResult(item_id, "pdf", True, page_idx, "ok")


def _page_count(pdf_path: Path) -> int:
    fitz = _fitz()
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    doc.close()
    return n


def _process_raster(img_path: Path, out_dir: Path, force: bool = False) -> NormResult:
    item_id = img_path.stem
    out_png = out_dir / f"{item_id}.png"
    out_json = out_dir / f"{item_id}.json"

    if out_png.exists() and out_json.exists() and not force:
        return NormResult(item_id, "raster", True, None, "already done")

    Image, _ = _pil()
    try:
        img = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        return NormResult(item_id, "raster", False, None, str(e))

    img = _to_grayscale_square_png(img)
    _save_clean_png(img, out_png)
    out_json.write_text(json.dumps({"id": item_id, "source": "raster", "text_masked": False, "output_size": TARGET_SIZE}, indent=2))
    return NormResult(item_id, "raster", True, None, "ok")


def _process_svg(svg_path: Path, out_dir: Path, force: bool = False) -> NormResult:
    item_id = svg_path.stem
    out_png = out_dir / f"{item_id}.png"
    out_json = out_dir / f"{item_id}.json"

    if out_png.exists() and out_json.exists() and not force:
        return NormResult(item_id, "svg", True, None, "already done")

    cairosvg = _cairosvg()
    if cairosvg is None:
        return NormResult(item_id, "svg", False, None, "cairosvg not installed")

    Image, _ = _pil()
    try:
        png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=1024, output_height=1024)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except Exception as e:
        return NormResult(item_id, "svg", False, None, str(e))

    img = _to_grayscale_square_png(img)
    _save_clean_png(img, out_png)
    out_json.write_text(json.dumps({"id": item_id, "source": "svg", "text_masked": False, "output_size": TARGET_SIZE}, indent=2))
    return NormResult(item_id, "svg", True, None, "ok")


# ---------------------------------------------------------------------------
# Collection + orchestration
# ---------------------------------------------------------------------------

def _collect_items(config: dict, source_filter: str | None) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if source_filter in (None, "positive"):
        for p in sorted(resolve_path(config["paths"]["raw_positive"]).glob("*.pdf")):
            items.append(("positive", p))
    if source_filter in (None, "negative_expired"):
        for p in sorted(resolve_path(config["paths"]["raw_negative_expired"]).glob("*.pdf")):
            items.append(("negative_expired", p))
    if source_filter in (None, "negative_opensource"):
        d = resolve_path(config["paths"]["raw_negative_opensource"])
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for p in sorted(d.glob(ext)):
                items.append(("negative_opensource_raster", p))
        for p in sorted(d.glob("*.svg")):
            items.append(("negative_opensource_svg", p))
    return items


def run(config, source_filter=None, target_id=None, dry_run=False, force=False, mask_text=True):
    out_dir = resolve_path(config["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _collect_items(config, source_filter)

    if target_id:
        items = [(s, p) for s, p in items if p.stem == target_id]
        if not items:
            logger.error(f"ID '{target_id}' not found")
            return

    logger.info(f"Items to process: {len(items)}")

    tess_available = _tesseract() is not None
    if mask_text and not tess_available:
        logger.warning("Tesseract not found -- text masking disabled. Add Tesseract to PATH.")

    if dry_run:
        for s, p in items[:20]:
            console.print(f"  [{s}] {p.name}")
        if len(items) > 20:
            console.print(f"  ... and {len(items) - 20} more")
        return

    results: list[NormResult] = []
    for source, path in items:
        ext = path.suffix.lower()
        if ext == ".pdf":
            r = _process_pdf(path, out_dir, mask_text=mask_text and tess_available, force=force)
        elif ext == ".svg":
            r = _process_svg(path, out_dir, force=force)
        elif ext in (".png", ".jpg", ".jpeg"):
            r = _process_raster(path, out_dir, force=force)
        else:
            continue
        status = "ok" if r.success and r.note != "already done" else ("skip" if r.note == "already done" else "FAIL")
        if status == "FAIL":
            logger.warning(f"  FAIL {r.id}: {r.note}")
        elif status == "ok":
            logger.info(f"  ok   {r.id}")
        results.append(r)

    done = sum(1 for r in results if r.success and r.note != "already done")
    skipped = sum(1 for r in results if r.note == "already done")
    failed = sum(1 for r in results if not r.success)
    console.rule("[bold]Normalize summary[/bold]")
    console.print(f"  Processed : {done}")
    console.print(f"  Skipped   : {skipped}")
    console.print(f"  Failed    : {failed}")
    if failed:
        for r in results:
            if not r.success:
                console.print(f"    {r.id}: {r.note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--source", choices=["positive", "negative_expired", "negative_opensource"])
    parser.add_argument("--id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-mask", action="store_true")
    args = parser.parse_args()
    if not any([args.all, args.source, args.id]):
        parser.error("Specify --all, --source <name>, or --id <id>")
    config = load_config()
    load_env()
    run(config, source_filter=args.source, target_id=args.id,
        dry_run=args.dry_run, force=args.force, mask_text=not args.no_mask)

if __name__ == "__main__":
    main()
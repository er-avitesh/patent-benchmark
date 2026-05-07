"""
diag_pdf_text.py -- dump text from every page of a patent PDF.

Run this to see exactly what text USPTO PDFs contain on each page,
so we can identify the correct pattern for the drawing sheet header.

Usage:
    python scripts/diag_pdf_text.py D1049406
    python scripts/diag_pdf_text.py D1049406 --raw   # show raw bytes too
"""
import sys
import re
import fitz
from pathlib import Path

patent_id = sys.argv[1] if len(sys.argv) > 1 else "D1049406"
raw_mode = "--raw" in sys.argv

# Search both positive and negative_expired folders
search_dirs = [
    Path("data/raw/positive"),
    Path("data/raw/negative_expired"),
]
pdf_path = None
for d in search_dirs:
    p = d / f"{patent_id}.pdf"
    if p.exists():
        pdf_path = p
        break

if not pdf_path:
    print(f"PDF not found for {patent_id}")
    sys.exit(1)

print(f"PDF: {pdf_path}  ({pdf_path.stat().st_size:,} bytes)")
doc = fitz.open(str(pdf_path))
print(f"Pages: {len(doc)}\n")

for i, page in enumerate(doc):
    text = page.get_text()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    print(f"--- Page {i+1} ({len(lines)} non-empty lines) ---")
    for line in lines[:30]:   # first 30 lines per page
        print(f"  {repr(line)}")
    if len(lines) > 30:
        print(f"  ... ({len(lines) - 30} more lines)")
    print()

doc.close()
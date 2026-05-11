"""
validate_pilot.py
-----------------
Quick validation of pilot results.
Run from project root:
    python scripts/validate_pilot.py
"""
import json
import pathlib
from collections import defaultdict

RAW_DIR = pathlib.Path("results/raw_responses")
files = list(RAW_DIR.glob("*.json"))

print(f"\n{'='*55}")
print(f"  Pilot Validation")
print(f"{'='*55}")

# ── 1. File count and errors ──────────────────────────────
errors = []
for f in files:
    d = json.loads(f.read_text())
    if d.get("error"):
        errors.append((f.name, d["error"]))

print(f"\nTotal checkpoint files : {len(files)}")
print(f"Jobs with errors       : {len(errors)}")
for name, err in errors:
    print(f"  ✗  {name}")
    print(f"     {str(err)[:120]}")

# ── 2. parse_ok per model ─────────────────────────────────
print(f"\n{'─'*55}")
print("  parse_ok rate per model")
print(f"{'─'*55}")
stats = defaultdict(lambda: {"ok": 0, "fail": 0, "empty_raw": 0})
for f in files:
    d = json.loads(f.read_text())
    m = d.get("model_key", "unknown")
    if d.get("parse_ok"):
        stats[m]["ok"] += 1
    else:
        stats[m]["fail"] += 1
    if not d.get("raw_response"):
        stats[m]["empty_raw"] += 1

for m, s in sorted(stats.items()):
    total = s["ok"] + s["fail"]
    bar = "✓" if s["fail"] == 0 else "✗"
    print(f"  {bar}  {m:10}  parse_ok={s['ok']}/{total}  empty_raw={s['empty_raw']}")

# ── 3. Verdict distribution per model ────────────────────
print(f"\n{'─'*55}")
print("  Verdict distribution (across all 20 queries x 5 candidates)")
print(f"{'─'*55}")
verdict_stats = defaultdict(lambda: {"INFRINGE": 0, "NO_INFRINGE": 0, "OTHER": 0})
for f in files:
    d = json.loads(f.read_text())
    m = d.get("model_key", "unknown")
    for k, v in d.get("verdicts", {}).items():
        if v == "INFRINGE":
            verdict_stats[m]["INFRINGE"] += 1
        elif v == "NO_INFRINGE":
            verdict_stats[m]["NO_INFRINGE"] += 1
        else:
            verdict_stats[m]["OTHER"] += 1

for m, s in sorted(verdict_stats.items()):
    total = s["INFRINGE"] + s["NO_INFRINGE"] + s["OTHER"]
    if total == 0:
        print(f"  {m:10}  no verdicts")
        continue
    inf_pct = 100 * s["INFRINGE"] / total
    print(f"  {m:10}  INFRINGE={s['INFRINGE']} ({inf_pct:.0f}%)  "
          f"NO_INFRINGE={s['NO_INFRINGE']}  OTHER={s['OTHER']}  total={total}")

# ── 4. Spot-check one raw response per model ─────────────
print(f"\n{'─'*55}")
print("  Raw response sample (first file per model)")
print(f"{'─'*55}")
seen = set()
for f in sorted(files):
    d = json.loads(f.read_text())
    m = d.get("model_key", "unknown")
    if m in seen:
        continue
    seen.add(m)
    raw = str(d.get("raw_response", ""))[:200]
    print(f"\n  [{m}]  {f.name}")
    print(f"  verdicts : {d.get('verdicts')}")
    print(f"  raw[200] : {raw!r}")

print(f"\n{'='*55}\n")
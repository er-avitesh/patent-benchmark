import json, pathlib
import pandas as pd

df = pd.read_csv("candidates/candidates.csv")
failed_qids = ["D1042123", "D12_Arca_Swiss_Profile.svg", "D555945"]

for qid in failed_qids:
    rows = df[df["query_id"] == qid]
    if rows.empty:
        print(f"{qid}: NOT FOUND in candidates.csv")
        continue
    row = rows.iloc[0]
    print(f"{qid}")
    print(f"  uspc  : {row['query_uspc']}")
    print(f"  label : {row['query_label']}")
    print(f"  png   : {row['query_png']}")
    print()

# Also show finish_reason for failed openai jobs
print("--- finish_reason from checkpoints ---")
for f in sorted(pathlib.Path("results/raw_responses").glob("*__openai__*.json")):
    d = json.loads(f.read_text())
    if not d.get("parse_ok"):
        raw = d.get("raw_response", "")
        print(f"{f.name}")
        print(f"  parse_ok    : {d.get('parse_ok')}")
        print(f"  raw_response: {repr(raw[:200])}")
        print(f"  error       : {d.get('error')}")
        print()
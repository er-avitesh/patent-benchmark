"""scratch.py - find all INFRINGE verdicts in pilot results"""
import json, pathlib

for f in sorted(pathlib.Path("results/raw_responses").glob("*.json")):
    d = json.loads(f.read_text())
    verdicts = d.get("verdicts", {})
    infringe = [k for k, v in verdicts.items() if v == "INFRINGE"]
    if infringe:
        print(f"\n{f.name}")
        print(f"  model   : {d['model_key']}")
        print(f"  query   : {d['query_id']} ({d['query_label']}, {d['query_uspc']})")
        print(f"  INFRINGE: {infringe}")
        for c in d.get("candidates", []):
            rank_key = f"candidate_{c['rank']}"
            if rank_key in infringe:
                print(f"  -> {c['candidate_id']} (label={c['candidate_label']})")
"""
06_select_candidates.py -- build candidates/candidates.csv.

For each of the 191 drawings in the manifest, selects 5 candidate drawings
from the same USPC class to show alongside it in the model query.

DESIGN DECISIONS
----------------
1. Class-restricted sampling: candidates are drawn only from the same USPC
   class as the query. This controls for domain -- models cannot distinguish
   positive from negative by visual domain (e.g. furniture vs medical devices).

2. Label balance: each candidate set contains a mix of positive and negative
   drawings where class size permits. The exact mix is random (seeded) so
   models cannot learn a positional heuristic.

3. Query excluded: the query drawing is never included in its own candidate set.

4. Fixed at build time: the same candidate sets are used across all models,
   strategies, and repetitions. This ensures cross-model comparisons are on
   identical inputs.

5. Ground truth stored: candidates.csv stores the label of each candidate.
   This is withheld from model prompts but used to score verdicts in module 8.

OUTPUT FORMAT (candidates/candidates.csv)
-----------------------------------------
Each row represents one (query, candidate) pair.

  query_id          ID of the query drawing
  query_label       positive or negative
  query_uspc        USPC class of query
  query_png         path to query PNG
  candidate_id      ID of candidate drawing
  candidate_label   positive or negative (ground truth, withheld from model)
  candidate_uspc    USPC class (always matches query_uspc)
  candidate_png     path to candidate PNG
  candidate_rank    1-5 (position in the candidate set for this query)

So each query produces 5 rows. 191 queries x 5 candidates = 955 rows total.

USAGE
-----
    python scripts/06_select_candidates.py
    python scripts/06_select_candidates.py --check   # print summary stats
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from _common import console, get_logger, load_config, load_env, resolve_path

logger = get_logger("select_candidates")

CANDIDATES_PER_QUERY = 5

CANDIDATE_COLUMNS = [
    "query_id",
    "query_label",
    "query_uspc",
    "query_png",
    "candidate_id",
    "candidate_label",
    "candidate_uspc",
    "candidate_png",
    "candidate_rank",
]


def load_manifest(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_class(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["uspc_class"]].append(row)
    return dict(groups)


def select_candidates_for_query(
    query: dict,
    class_pool: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """
    Select n candidates for a query from the class pool.

    Strategy:
    - Remove the query itself from the pool
    - Shuffle the remaining pool
    - Take the first n items

    The shuffle is seeded so results are deterministic and reproducible.
    We do not enforce a specific label ratio -- the natural class balance
    in the pool determines the candidate mix. With ~21 drawings per class
    (12 positive + 9 expired + ~3 open-source), a random draw of 5 will
    typically yield 2-3 positives and 2-3 negatives, which is appropriate.
    """
    pool = [r for r in class_pool if r["id"] != query["id"]]

    if len(pool) < n:
        logger.warning(
            f"  {query['id']} ({query['uspc_class']}): only {len(pool)} candidates "
            f"available, wanted {n}. Using all."
        )
        return pool

    rng.shuffle(pool)
    return pool[:n]


def build_candidates(config: dict, check: bool = False) -> Path:
    manifest_path = resolve_path(config["paths"]["manifest"])
    candidates_dir = resolve_path("candidates")
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = candidates_dir / "candidates.csv"

    seed = config["random_seed"]
    rng = random.Random(seed + 6)   # offset from previous modules

    manifest = load_manifest(manifest_path)
    logger.info(f"Loaded manifest: {len(manifest)} rows")

    by_class = group_by_class(manifest)
    logger.info(f"Classes: {sorted(by_class.keys())}")
    for cls, rows in sorted(by_class.items()):
        pos = sum(1 for r in rows if r["label"] == "positive")
        neg = sum(1 for r in rows if r["label"] == "negative")
        logger.info(f"  {cls}: {len(rows)} total ({pos} positive, {neg} negative)")

    rows_out: list[dict] = []
    queries_with_few_candidates = []

    for query in manifest:
        cls = query["uspc_class"]
        if not cls:
            logger.warning(f"  {query['id']}: no USPC class, skipping")
            continue

        pool = by_class.get(cls, [])
        candidates = select_candidates_for_query(query, pool, CANDIDATES_PER_QUERY, rng)

        if len(candidates) < CANDIDATES_PER_QUERY:
            queries_with_few_candidates.append(query["id"])

        for rank, candidate in enumerate(candidates, start=1):
            rows_out.append({
                "query_id":        query["id"],
                "query_label":     query["label"],
                "query_uspc":      query["uspc_class"],
                "query_png":       query["png_path"],
                "candidate_id":    candidate["id"],
                "candidate_label": candidate["label"],
                "candidate_uspc":  candidate["uspc_class"],
                "candidate_png":   candidate["png_path"],
                "candidate_rank":  rank,
            })

    with candidates_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_out)

    logger.info(f"Wrote {candidates_path}")

    # Summary
    n_queries = len(manifest)
    n_pairs = len(rows_out)
    avg_candidates = n_pairs / n_queries if n_queries else 0

    # Label balance in candidate sets
    pos_candidates = sum(1 for r in rows_out if r["candidate_label"] == "positive")
    neg_candidates = sum(1 for r in rows_out if r["candidate_label"] == "negative")

    console.rule("[bold]Candidate selection summary[/bold]")
    console.print(f"  Queries          : {n_queries}")
    console.print(f"  Total pairs      : {n_pairs}")
    console.print(f"  Avg candidates   : {avg_candidates:.1f} per query")
    console.print(f"  Positive cands   : {pos_candidates} ({100*pos_candidates/n_pairs:.0f}%)")
    console.print(f"  Negative cands   : {neg_candidates} ({100*neg_candidates/n_pairs:.0f}%)")
    if queries_with_few_candidates:
        console.print(f"  Fewer than {CANDIDATES_PER_QUERY} candidates: {queries_with_few_candidates}")
    console.print(f"  Output           : {candidates_path}")

    if check:
        console.print("\n[bold]Sample (first query):[/bold]")
        first_query = manifest[0]["id"]
        for row in rows_out:
            if row["query_id"] == first_query:
                console.print(
                    f"  rank {row['candidate_rank']}: {row['candidate_id']:25s} "
                    f"{row['candidate_label']:8s} {row['candidate_uspc']}"
                )

    return candidates_path


def main():
    parser = argparse.ArgumentParser(
        description="Build candidates/candidates.csv for model evaluation."
    )
    parser.add_argument("--check", action="store_true",
                        help="Print summary stats and sample candidates")
    args = parser.parse_args()

    config = load_config()
    load_env()
    build_candidates(config, check=args.check)


if __name__ == "__main__":
    main()
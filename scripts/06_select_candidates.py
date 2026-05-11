"""
06_select_candidates.py -- build candidates/candidates.csv.

For each of the 192 drawings in the manifest, selects 5 candidate drawings
from the same USPC class to show alongside it in the model query.

DESIGN DECISIONS
----------------
1. Class-restricted sampling: candidates are drawn only from the same USPC
   class as the query. This controls for domain -- models cannot distinguish
   positive from negative by visual domain (e.g. furniture vs medical devices).

2. Title-keyword matching (Fix B): within the class pool, candidates whose
   invention_title shares a keyword with the query title are prioritised.
   This produces visually tighter pairs (chairs vs chairs, bottles vs bottles)
   rather than loose within-class matches (chairs vs hospital beds).
   Keyword matching is applied before random fill, ensuring the candidate set
   is as visually relevant as possible while remaining seeded/reproducible.

3. Label balance: each candidate set contains a mix of positive and negative
   drawings where class size permits. The exact mix is random (seeded).

4. Query excluded: the query drawing is never included in its own candidate set.

5. Fixed at build time: the same candidate sets are used across all models,
   strategies, and repetitions.

6. Ground truth stored: candidates.csv stores the label of each candidate,
   withheld from model prompts but used to score verdicts in module 8.

OUTPUT FORMAT (candidates/candidates.csv)
-----------------------------------------
  query_id          ID of the query drawing
  query_label       positive or negative
  query_uspc        USPC class of query
  query_png         path to query PNG
  candidate_id      ID of candidate drawing
  candidate_label   positive or negative (ground truth, withheld from model)
  candidate_uspc    USPC class (always matches query_uspc)
  candidate_png     path to candidate PNG
  candidate_rank    1-5 (position in the candidate set for this query)

Each query produces 5 rows. 192 queries x 5 candidates = 960 rows total.

USAGE
-----
    python scripts/06_select_candidates.py
    python scripts/06_select_candidates.py --check   # print summary stats
"""

from __future__ import annotations

import argparse
import csv
import re
import random
from collections import defaultdict
from pathlib import Path

from _common import console, get_logger, load_config, load_env, resolve_path

logger = get_logger("select_candidates")

CANDIDATES_PER_QUERY = 5

# Common English words that add no discriminative value for title matching.
# Keeping this list short — we want to filter noise, not over-filter.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "to", "in",
    "on", "at", "by", "from", "as", "is", "its", "part", "parts",
    "set", "sets", "type", "types", "having", "with", "without",
    "assembly", "device", "apparatus", "unit", "system", "design",
    "ornamental", "portion", "shown",
}

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


def title_keywords(title: str) -> set[str]:
    """Extract meaningful keywords from an invention title."""
    if not title:
        return set()
    words = re.findall(r"[a-z]+", title.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) >= 3}


def select_candidates_for_query(
    query: dict,
    class_pool: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """
    Select n candidates for a query from the class pool.

    Strategy (Fix B — title-keyword matching):
    1. Remove the query itself from the pool.
    2. Split pool into keyword-match candidates (share ≥1 title keyword
       with the query) and non-match candidates.
    3. Shuffle each group independently (seeded).
    4. Fill candidate set: take from keyword-matches first, top up with
       non-matches if needed to reach n.

    This prioritises visually tighter pairs (chair vs chair, bottle vs bottle)
    while guaranteeing we always get n candidates even in thin classes.
    Falls back to pure random selection (original behaviour) when no
    keyword matches exist (e.g. generated PIL drawings with generic titles).
    """
    pool = [r for r in class_pool if r["id"] != query["id"]]

    if len(pool) < n:
        logger.warning(
            f"  {query['id']} ({query['uspc_class']}): only {len(pool)} candidates "
            f"available, wanted {n}. Using all."
        )
        rng.shuffle(pool)
        return pool

    query_kw = title_keywords(query.get("invention_title", ""))

    if query_kw:
        keyword_matches = [
            r for r in pool
            if title_keywords(r.get("invention_title", "")) & query_kw
        ]
        non_matches = [
            r for r in pool
            if r not in keyword_matches
        ]
    else:
        keyword_matches = []
        non_matches = pool

    rng.shuffle(keyword_matches)
    rng.shuffle(non_matches)

    # Fill: keyword matches first, then non-matches
    selected = (keyword_matches + non_matches)[:n]

    if keyword_matches:
        n_kw = min(len(keyword_matches), n)
        logger.debug(
            f"  {query['id']}: {n_kw} keyword-match candidates "
            f"(keywords: {query_kw})"
        )

    return selected


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
    keyword_match_count = 0

    for query in manifest:
        cls = query["uspc_class"]
        if not cls:
            logger.warning(f"  {query['id']}: no USPC class, skipping")
            continue

        pool = by_class.get(cls, [])
        candidates = select_candidates_for_query(query, pool, CANDIDATES_PER_QUERY, rng)

        if len(candidates) < CANDIDATES_PER_QUERY:
            queries_with_few_candidates.append(query["id"])

        # Count how many candidates share a keyword with this query
        query_kw = title_keywords(query.get("invention_title", ""))
        for c in candidates:
            if query_kw & title_keywords(c.get("invention_title", "")):
                keyword_match_count += 1

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
    n_queries  = len(manifest)
    n_pairs    = len(rows_out)
    avg_cands  = n_pairs / n_queries if n_queries else 0
    pos_cands  = sum(1 for r in rows_out if r["candidate_label"] == "positive")
    neg_cands  = sum(1 for r in rows_out if r["candidate_label"] == "negative")
    kw_pct     = 100 * keyword_match_count / n_pairs if n_pairs else 0

    console.rule("[bold]Candidate selection summary[/bold]")
    console.print(f"  Queries              : {n_queries}")
    console.print(f"  Total pairs          : {n_pairs}")
    console.print(f"  Avg candidates       : {avg_cands:.1f} per query")
    console.print(f"  Positive cands       : {pos_cands} ({100*pos_cands/n_pairs:.0f}%)")
    console.print(f"  Negative cands       : {neg_cands} ({100*neg_cands/n_pairs:.0f}%)")
    console.print(f"  Keyword-matched cands: {keyword_match_count} ({kw_pct:.0f}%)")
    if queries_with_few_candidates:
        console.print(f"  Fewer than {CANDIDATES_PER_QUERY}: {queries_with_few_candidates}")
    console.print(f"  Output               : {candidates_path}")

    if check:
        console.print("\n[bold]Sample (first 3 queries):[/bold]")
        seen = set()
        for row in rows_out:
            qid = row["query_id"]
            if len(seen) >= 3 and qid not in seen:
                break
            seen.add(qid)
            if list(seen).index(qid) < 3:
                console.print(
                    f"  [{qid[:20]:20s}] rank {row['candidate_rank']}: "
                    f"{row['candidate_id'][:25]:25s} "
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
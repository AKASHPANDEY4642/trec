"""
Step 5: Reciprocal Rank Fusion (RRF)
======================================
Combines multiple ranked lists (e.g., raw BM25 + processed BM25)
using Reciprocal Rank Fusion to get a better final ranking.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
where k=60 is the standard constant and rank_i(d) is the rank of 
document d in ranked list i.

Usage:
    python 05_rrf_fusion.py

Output:
    runs/rrf_fused.txt  — TREC run file with RRF-fused results
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, RUN_ID


def load_trec_run(filepath: Path) -> dict[str, list[tuple[str, int, float]]]:
    """
    Load a TREC run file.
    Returns dict mapping topic_id -> list of (docid, rank, score) tuples.
    """
    results = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            topic_id, _, docid, rank, score, _ = parts
            results[topic_id].append((docid, int(rank), float(score)))
    return dict(results)


def rrf_fusion(
    runs: list[dict[str, list[tuple[str, int, float]]]],
    k: int = 60,
    max_docs: int = 100
) -> dict[str, list[tuple[str, float]]]:
    """
    Reciprocal Rank Fusion of multiple ranked lists.
    
    Args:
        runs: List of run dicts (topic_id -> list of (docid, rank, score))
        k: RRF constant (default 60, per Cormack et al. 2009)
        max_docs: Maximum documents per topic in output
    
    Returns:
        dict mapping topic_id -> list of (docid, rrf_score) sorted by score desc
    """
    # Collect all topic IDs
    all_topics = set()
    for run in runs:
        all_topics.update(run.keys())
    
    fused = {}
    for topic_id in sorted(all_topics):
        doc_scores = defaultdict(float)
        
        for run in runs:
            if topic_id not in run:
                continue
            for docid, rank, score in run[topic_id]:
                doc_scores[docid] += 1.0 / (k + rank)
        
        # Sort by RRF score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused[topic_id] = sorted_docs[:max_docs]
    
    return fused


def write_fused_run(fused: dict[str, list[tuple[str, float]]], output_path: Path, run_id: str):
    """Write RRF-fused results in TREC run file format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in sorted(fused.keys(), key=lambda x: int(x.split("-")[1])):
            for rank, (docid, score) in enumerate(fused[topic_id], start=1):
                f.write(f"{topic_id} Q0 {docid} {rank} {score:.10f} {run_id}\n")
                total += 1
    print(f"Wrote {total} lines to {output_path}")


def main():
    print("=" * 70)
    print("TREC RAG 2026 — Step 5: Reciprocal Rank Fusion")
    print("=" * 70)
    
    # --- Find available run files to fuse ---
    available_runs = []
    run_files = [
        ("BM25 Raw", OUTPUT_DIR / "bm25_raw.txt"),
        ("BM25 Processed", OUTPUT_DIR / "query_rewritten_bm25.txt"),
        ("BM25 Reranked", OUTPUT_DIR / "bm25_reranked.txt"),
    ]
    
    for name, path in run_files:
        if path.exists():
            print(f"  Found: {name} -> {path}")
            available_runs.append((name, path))
        else:
            print(f"  Missing: {name} -> {path}")
    
    if len(available_runs) < 2:
        print(f"\nNeed at least 2 run files for fusion. Found {len(available_runs)}.")
        print("Run more retrieval variants first (01, 03, or 04).")
        sys.exit(1)
    
    # --- Load runs ---
    runs = []
    for name, path in available_runs:
        run = load_trec_run(path)
        runs.append(run)
        print(f"  Loaded {name}: {len(run)} topics")
    
    # --- Fuse ---
    print(f"\nFusing {len(runs)} runs with RRF (k=60)...")
    fused = rrf_fusion(runs, k=60, max_docs=100)
    
    # --- Write output ---
    output_path = OUTPUT_DIR / "rrf_fused.txt"
    fused_run_id = RUN_ID.replace("bm25", "rrf")
    write_fused_run(fused, output_path, fused_run_id)
    
    print(f"\nFusion complete! {len(fused)} topics")
    print(f"Next: Run 07_validate_run.py to validate before submission")


if __name__ == "__main__":
    main()

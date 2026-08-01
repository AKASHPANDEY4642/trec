"""
Step 6: Spot-Check Results Quality
====================================
Before submitting, manually inspect a few topics to see if the
retrieved documents look relevant. This gives you confidence that
the pipeline is working correctly.

Usage:
    python 06_spot_check.py [--run runs/bm25_raw.txt] [--topics 3]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR, TMP_DIR, QUERIES_FILE


def load_queries(filepath: Path) -> dict[str, str]:
    """Load queries as dict."""
    queries = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0].strip()] = parts[1].strip()
    return queries


def load_trec_run(filepath: Path) -> dict[str, list[tuple[str, int, float]]]:
    """Load a TREC run file."""
    from collections import defaultdict
    results = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 6:
                topic_id, _, docid, rank, score, _ = parts
                results[topic_id].append((docid, int(rank), float(score)))
    return dict(results)


def main():
    parser = argparse.ArgumentParser(description="Spot-check retrieval results")
    parser.add_argument("--run", default=str(OUTPUT_DIR / "bm25_raw.txt"),
                        help="Path to the run file to inspect")
    parser.add_argument("--topics", type=int, default=3,
                        help="Number of topics to inspect")
    parser.add_argument("--docs", type=int, default=5,
                        help="Number of top docs to show per topic")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TREC RAG 2026 — Step 6: Spot-Check Results")
    print("=" * 70)
    
    # Load queries
    queries = load_queries(QUERIES_FILE)
    
    # Load run
    run_path = Path(args.run)
    if not run_path.exists():
        print(f"ERROR: Run file not found: {run_path}")
        sys.exit(1)
    
    run = load_trec_run(run_path)
    print(f"Loaded run: {len(run)} topics from {run_path}")
    
    # Load cached JSON results if available (for doc text)
    doc_cache = {}
    for cache_file in TMP_DIR.glob("bm25*results.json"):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for topic_id, docs in data.items():
                for doc in docs:
                    doc_cache[doc["docid"]] = doc.get("doc", "")
    
    # Spot check
    topic_ids = sorted(run.keys(), key=lambda x: int(x.split("-")[1]))[:args.topics]
    
    for topic_id in topic_ids:
        print(f"\n{'=' * 70}")
        print(f"TOPIC: {topic_id}")
        print(f"{'=' * 70}")
        
        query = queries.get(topic_id, "N/A")
        print(f"\nNarrative: {query[:300]}...")
        
        results = run[topic_id][:args.docs]
        print(f"\nTop-{args.docs} retrieved documents:")
        
        for docid, rank, score in results:
            print(f"\n  Rank {rank} | Score: {score:.4f} | DocID: {docid}")
            
            doc_text = doc_cache.get(docid, "")
            if doc_text:
                if isinstance(doc_text, dict):
                    doc_text = doc_text.get("contents", doc_text.get("text", str(doc_text)))
                doc_text = str(doc_text)[:300]
                print(f"  Doc: {doc_text}...")
            else:
                print(f"  Doc: [text not cached — run retrieval with doc caching]")
    
    print(f"\n{'=' * 70}")
    print("Review the results above. Do the top documents look relevant to the queries?")
    print("If yes, proceed to submit. If not, consider:")
    print("  1. Running query processing (02_query_processing.py)")
    print("  2. Running reranking (04_rerank.py)")
    print("  3. Using RRF fusion (05_rrf_fusion.py)")


if __name__ == "__main__":
    main()

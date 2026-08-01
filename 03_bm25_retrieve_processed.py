"""
Step 3: BM25 Retrieval with Processed Queries
===============================================
Re-runs BM25 retrieval using the cleaned/focused queries from Step 2
instead of the raw narrative text.

Usage:
    python 03_bm25_retrieve_processed.py

Prerequisites:
    - Run 02_query_processing.py first
    - API token set in .env.local

Output:
    runs/query_rewritten_bm25.txt  — TREC run file with processed queries
"""

import sys
import time
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    API_BASE_URL, INDEX_NAME, API_TOKEN, TMP_DIR,
    BM25_HITS, QUERY_REWRITTEN_RUN, RUN_ID,
    REQUEST_DELAY_SECONDS
)
from importlib import import_module

# Reuse the search and write functions from step 1
step1 = import_module("01_bm25_retrieve")
search_pyserini = step1.search_pyserini
write_trec_run = step1.write_trec_run


def main():
    print("=" * 70)
    print("TREC RAG 2026 — Step 3: BM25 Retrieval with Processed Queries")
    print("=" * 70)
    
    # Load processed queries
    processed_path = TMP_DIR / "processed_queries.json"
    if not processed_path.exists():
        print(f"ERROR: Processed queries not found at {processed_path}")
        print(f"Run 02_query_processing.py first")
        sys.exit(1)
    
    with open(processed_path, "r", encoding="utf-8") as f:
        processed_queries = json.load(f)
    
    print(f"Loaded {len(processed_queries)} processed queries")
    
    if not API_TOKEN:
        print(f"ERROR: No API token. Set PYSERINI_API_TOKEN in .env.local")
        sys.exit(1)
    
    # --- Choose which query variant to use ---
    # Options: "original", "first_sentence", "core_question", "keywords", "combined"
    # "combined" generally works best for BM25
    QUERY_VARIANT = "combined"
    print(f"Using query variant: {QUERY_VARIANT}")
    
    # --- Run retrieval ---
    all_results = {}
    failed = []
    
    for entry in tqdm(processed_queries, desc=f"BM25 ({QUERY_VARIANT})"):
        narrative_id = entry["narrative_id"]
        query_text = entry[QUERY_VARIANT]
        
        candidates = search_pyserini(query_text, hits=BM25_HITS)
        
        if candidates:
            all_results[narrative_id] = candidates
        else:
            failed.append(narrative_id)
        
        time.sleep(REQUEST_DELAY_SECONDS)
    
    # --- Report ---
    print(f"\nRetrieval complete: {len(all_results)}/{len(processed_queries)} successful")
    if failed:
        print(f"Failed: {failed}")
    
    # --- Write run file ---
    run_id_variant = RUN_ID.replace("bm25", f"bm25_{QUERY_VARIANT}")
    write_trec_run(all_results, QUERY_REWRITTEN_RUN, run_id_variant)
    
    # --- Cache results for reranking ---
    json_cache = TMP_DIR / f"bm25_{QUERY_VARIANT}_results.json"
    serializable = {}
    for topic_id, candidates in all_results.items():
        serializable[topic_id] = [
            {
                "docid": c.get("docid", ""),
                "score": c.get("score", 0.0),
                "rank": c.get("rank", 0),
                "doc": c.get("doc", "")
            }
            for c in candidates
        ]
    
    with open(json_cache, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Cached results to {json_cache}")
    
    print(f"\nNext: Run 04_rerank.py to rerank these results")


if __name__ == "__main__":
    main()

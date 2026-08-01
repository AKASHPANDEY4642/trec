"""
Step 1: BM25 Retrieval via Pyserini REST API
=============================================
This is the core retrieval script. It reads the 119 test narratives,
sends each one as a BM25 search query to the Pyserini REST API,
and writes the results in standard TREC run file format.

Usage:
    python 01_bm25_retrieve.py

Output:
    runs/bm25_raw.txt  — Standard TREC run file

Prerequisites:
    1. Clone the data repo: git clone https://github.com/TREC-RAG/trec-rag-data.git
    2. Set your API token in .env.local: PYSERINI_API_TOKEN=your_token_here
"""

import sys
import time
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    API_BASE_URL, INDEX_NAME, API_TOKEN, QUERIES_FILE,
    BM25_HITS, BM25_RAW_RUN, RUN_ID,
    REQUEST_DELAY_SECONDS, MAX_RETRIES, RETRY_DELAY_SECONDS
)


def load_queries(filepath: Path) -> list[tuple[str, str]]:
    """
    Load test narratives from TSV file.
    Returns list of (narrative_id, narrative_text) tuples.
    """
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # TSV format: narrative_id<TAB>narrative_text
            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"WARNING: Skipping malformed line: {line[:80]}...")
                continue
            narrative_id, narrative_text = parts
            queries.append((narrative_id.strip(), narrative_text.strip()))
    
    print(f"Loaded {len(queries)} narratives from {filepath}")
    return queries


def search_pyserini(query_text: str, hits: int = 100) -> list[dict]:
    """
    Send a BM25 search request to the Pyserini REST API.
    
    API: GET /v1/{index}/search?query={text}&hits={n}
    Headers: Authorization: Bearer {token}
    
    Returns list of result dicts with keys: docid, rank, score, doc
    """
    url = f"{API_BASE_URL}/v1/{INDEX_NAME}/search"
    params = {
        "query": query_text,
        "hits": hits
    }
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=120)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # The API returns results in a 'candidates' array
                    candidates = data.get("candidates", data.get("results", []))
                    return candidates
                except json.JSONDecodeError as e:
                    print(f"JSON decode error on attempt {attempt}/{MAX_RETRIES}: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return []
            
            elif response.status_code == 401:
                print(f"ERROR: Authentication failed (401). Check your API token in .env.local")
                print(f"If you don't have a token, email get-pyserini@googlegroups.com")
                sys.exit(1)
            
            elif response.status_code == 429:
                # Rate limited — wait and retry
                wait_time = RETRY_DELAY_SECONDS * attempt
                print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt}/{MAX_RETRIES}...")
                time.sleep(wait_time)
                continue
            
            elif response.status_code >= 500:
                # Server error — retry
                wait_time = RETRY_DELAY_SECONDS * attempt
                print(f"Server error ({response.status_code}). Retry {attempt}/{MAX_RETRIES} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            else:
                print(f"ERROR: Unexpected status {response.status_code}: {response.text[:200]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                return []
        
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt}/{MAX_RETRIES}. Retrying...")
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error on attempt {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(RETRY_DELAY_SECONDS * attempt)
            continue
    
    print(f"FAILED after {MAX_RETRIES} attempts")
    return []


def write_trec_run(results: dict[str, list[dict]], output_path: Path, run_id: str):
    """
    Write results in standard TREC run file format.
    
    Format: topic_id Q0 docid rank score run_id
    One line per retrieved document, space-separated.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in sorted(results.keys(), key=lambda x: int(x.split("-")[1])):
            candidates = results[topic_id]
            for rank_idx, candidate in enumerate(candidates, start=1):
                docid = candidate.get("docid", "")
                score = candidate.get("score", 0.0)
                # Write: topic_id Q0 docid rank score run_id
                f.write(f"{topic_id} Q0 {docid} {rank_idx} {score} {run_id}\n")
                total_lines += 1
    
    print(f"Wrote {total_lines} lines to {output_path}")
    print(f"  Topics covered: {len(results)}")
    print(f"  Avg docs per topic: {total_lines / max(len(results), 1):.1f}")


def main():
    print("=" * 70)
    print("TREC RAG 2026 — Step 1: BM25 Retrieval via Pyserini REST API")
    print("=" * 70)
    
    # --- Check prerequisites ---
    if not QUERIES_FILE.exists():
        print(f"\nERROR: Queries file not found: {QUERIES_FILE}")
        print(f"\nFix: Clone the data repository first:")
        print(f"  git clone https://github.com/TREC-RAG/trec-rag-data.git")
        sys.exit(1)
    
    if not API_TOKEN:
        print(f"\nERROR: No API token found. Set PYSERINI_API_TOKEN in .env.local")
        print(f"\nTo get a token, email: get-pyserini@googlegroups.com")
        print(f"\nThen create .env.local with:")
        print(f"  PYSERINI_API_TOKEN=your_token_here")
        sys.exit(1)
    
    # --- Test API connectivity ---
    print(f"\nTesting API connectivity at {API_BASE_URL}...")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/", timeout=10)
        print(f"  API status: {health_resp.status_code} — {'OK' if health_resp.status_code == 200 else 'Check URL'}")
    except Exception as e:
        print(f"  WARNING: Could not reach API: {e}")
        print(f"  Will attempt queries anyway...")
    
    # --- Load queries and existing cache ---
    queries = load_queries(QUERIES_FILE)
    if not queries:
        print("ERROR: No queries loaded. Check the file format.")
        sys.exit(1)
        
    json_cache = Path("tmp") / "bm25_results.json"
    all_results = {}
    failed_queries = []
    
    if json_cache.exists():
        try:
            with open(json_cache, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            sample_len = max([len(v) for v in cached_data.values()]) if cached_data else 0
            if (BM25_HITS > 100 and sample_len > 100) or (BM25_HITS <= 100):
                print(f"Found cached results for {len(cached_data)} queries with sample depth {sample_len}. Resuming...")
                for topic_id, candidates in cached_data.items():
                    all_results[topic_id] = [
                        {
                            "docid": c.get("docid", ""),
                            "score": c.get("score", 0.0),
                            "rank": c.get("rank", 0),
                            "doc": c.get("doc", "")
                        }
                        for c in candidates
                    ]
            else:
                print(f"Cached data has depth {sample_len}, but we requested {BM25_HITS} hits. Starting fresh.")
        except Exception as e:
            print(f"Error loading cache: {e}. Starting fresh.")
    
    # --- Run BM25 retrieval for all queries ---
    remaining_queries = [(nid, ntxt) for nid, ntxt in queries if nid not in all_results]
    print(f"\nRetrieving top-{BM25_HITS} documents for {len(remaining_queries)}/{len(queries)} remaining narratives...")
    print(f"Estimated time: ~{len(remaining_queries) * (REQUEST_DELAY_SECONDS + 2):.0f} seconds")
    print()
    
    for narrative_id, narrative_text in tqdm(remaining_queries, desc="BM25 Retrieval"):
        candidates = search_pyserini(narrative_text, hits=BM25_HITS)
        
        if candidates:
            all_results[narrative_id] = candidates
            # Incremental save to cache
            serializable_results = {}
            for topic_id, candidates_list in all_results.items():
                serializable_results[topic_id] = []
                for c in candidates_list:
                    serializable_results[topic_id].append({
                        "docid": c.get("docid", ""),
                        "score": c.get("score", 0.0),
                        "rank": c.get("rank", 0),
                        "doc": c.get("doc", "")
                    })
            try:
                json_cache.parent.mkdir(exist_ok=True)
                with open(json_cache, "w", encoding="utf-8") as f:
                    json.dump(serializable_results, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to write incremental cache: {e}")
        else:
            failed_queries.append(narrative_id)
            print(f"\n  WARNING: No results for {narrative_id}")
        
        # Rate limiting — be polite to the API
        time.sleep(REQUEST_DELAY_SECONDS)
    
    # --- Report results ---
    print(f"\n{'=' * 70}")
    print(f"BM25 Retrieval Complete!")
    print(f"  Successful: {len(all_results)}/{len(queries)}")
    if failed_queries:
        print(f"  Failed: {len(failed_queries)} — {failed_queries}")
    
    # --- Write TREC run file ---
    write_trec_run(all_results, BM25_RAW_RUN, RUN_ID)
    
    # --- Also save raw JSON for reranking ---
    json_cache = Path("tmp") / "bm25_results.json"
    json_cache.parent.mkdir(exist_ok=True)
    
    # Save a serializable version (doc text + metadata needed for reranking)
    serializable_results = {}
    for topic_id, candidates in all_results.items():
        serializable_results[topic_id] = []
        for c in candidates:
            serializable_results[topic_id].append({
                "docid": c.get("docid", ""),
                "score": c.get("score", 0.0),
                "rank": c.get("rank", 0),
                "doc": c.get("doc", "")  # Document text for reranking
            })
    
    with open(json_cache, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nCached raw results (with doc text) to {json_cache} for reranking")
    
    print(f"\nNext step: Run 04_rerank.py to rerank with a cross-encoder")
    print(f"Or run 07_validate_run.py to validate the BM25 baseline run file")


if __name__ == "__main__":
    main()

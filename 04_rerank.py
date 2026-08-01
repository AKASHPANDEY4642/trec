"""
Step 4: Cross-Encoder Reranking
================================
This is the BIGGEST quality improvement over BM25.
Takes the top-100 BM25 results and re-scores each (query, document) pair
using a neural cross-encoder model, then re-sorts by the new scores.

Typical improvement: +20-40% on nDCG@10

Usage:
    python 04_rerank.py [--input bm25_raw|bm25_combined] [--gpu] [--flashrank]

Options:
    --input bm25_raw        Rerank the raw BM25 results (default)
    --input bm25_combined   Rerank the processed-query BM25 results
    --gpu                   Force GPU usage
    --flashrank             Use FlashRank (much faster, CPU-friendly)

Output:
    runs/bm25_reranked.txt  — TREC run file with reranked results

Prerequisites:
    - Run 01_bm25_retrieve.py first (or 03_bm25_retrieve_processed.py)
    - pip install sentence-transformers torch  (for cross-encoder)
    - OR pip install flashrank  (for FlashRank)
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TMP_DIR, BM25_RERANKED_RUN, RUN_ID, RERANK_MODEL, RERANK_DEPTH
)


def rerank_with_cross_encoder(
    queries: dict[str, str],
    results: dict[str, list[dict]],
    model_name: str,
    use_gpu: bool = False,
    rerank_depth: int = 100
) -> dict[str, list[dict]]:
    """
    Rerank BM25 results using a cross-encoder model.
    
    Args:
        queries: dict mapping topic_id -> query text
        results: dict mapping topic_id -> list of {docid, score, doc} dicts
        model_name: HuggingFace model name for the cross-encoder
        use_gpu: Whether to use GPU
        rerank_depth: How many top documents to rerank per query
    
    Returns:
        dict mapping topic_id -> reranked list of {docid, score, doc} dicts
    """
    from sentence_transformers import CrossEncoder
    import torch
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Loading cross-encoder: {model_name}")
    print(f"Device: {device}")
    
    model = CrossEncoder(model_name, device=device)
    
    reranked_results = {}
    
    for topic_id in tqdm(sorted(results.keys()), desc="Reranking"):
        candidates = results[topic_id][:rerank_depth]
        query_text = queries.get(topic_id, "")
        
        if not query_text or not candidates:
            reranked_results[topic_id] = candidates
            continue
        
        # Prepare (query, document) pairs for the cross-encoder
        pairs = []
        for c in candidates:
            doc_text = c.get("doc", "")
            if isinstance(doc_text, dict):
                # Some responses may return doc as an object
                doc_text = doc_text.get("contents", doc_text.get("text", str(doc_text)))
            # Truncate very long documents (cross-encoders have token limits)
            doc_text = str(doc_text)[:2048]
            pairs.append((query_text, doc_text))
        
        # Score all pairs
        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"  WARNING: Reranking failed for {topic_id}: {e}")
            reranked_results[topic_id] = candidates
            continue
        
        # Attach new scores and sort
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        
        # Sort by rerank_score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Replace score with rerank_score for the run file
        for rank, c in enumerate(candidates, start=1):
            c["score"] = c["rerank_score"]
            c["rank"] = rank
        
        reranked_results[topic_id] = candidates
    
    return reranked_results


def rerank_with_flashrank(
    queries: dict[str, str],
    results: dict[str, list[dict]],
    rerank_depth: int = 100
) -> dict[str, list[dict]]:
    """
    Rerank BM25 results using FlashRank (fast, CPU-friendly).
    
    pip install flashrank
    """
    from flashrank import Ranker, RerankRequest
    
    print("Loading FlashRank ranker...")
    ranker = Ranker()
    
    reranked_results = {}
    
    for topic_id in tqdm(sorted(results.keys()), desc="Reranking (FlashRank)"):
        candidates = results[topic_id][:rerank_depth]
        query_text = queries.get(topic_id, "")
        
        if not query_text or not candidates:
            reranked_results[topic_id] = candidates
            continue
        
        # Prepare passages for FlashRank
        passages = []
        for c in candidates:
            doc_text = c.get("doc", "")
            if isinstance(doc_text, dict):
                doc_text = doc_text.get("contents", doc_text.get("text", str(doc_text)))
            doc_text = str(doc_text)[:2048]
            passages.append({"id": c["docid"], "text": doc_text})
        
        try:
            rerank_request = RerankRequest(query=query_text, passages=passages)
            reranked = ranker.rerank(rerank_request)
            
            # Map back to our format
            docid_to_candidate = {c["docid"]: c for c in candidates}
            new_candidates = []
            for rank, item in enumerate(reranked, start=1):
                docid = item["id"]
                if docid in docid_to_candidate:
                    c = docid_to_candidate[docid]
                    c["score"] = float(item["score"])
                    c["rank"] = rank
                    new_candidates.append(c)
            
            reranked_results[topic_id] = new_candidates
        except Exception as e:
            print(f"  WARNING: FlashRank failed for {topic_id}: {e}")
            reranked_results[topic_id] = candidates
    
    return reranked_results


def write_trec_run(results: dict[str, list[dict]], output_path: Path, run_id: str):
    """Write results in TREC run file format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in sorted(results.keys(), key=lambda x: int(x.split("-")[1])):
            for rank_idx, c in enumerate(results[topic_id], start=1):
                docid = c.get("docid", "")
                score = c.get("score", 0.0)
                f.write(f"{topic_id} Q0 {docid} {rank_idx} {score} {run_id}\n")
                total += 1
    print(f"Wrote {total} lines to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Rerank BM25 results with cross-encoder")
    parser.add_argument("--input", default="bm25_raw",
                        choices=["bm25_raw", "bm25_combined"],
                        help="Which BM25 results to rerank")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--flashrank", action="store_true",
                        help="Use FlashRank instead of cross-encoder")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TREC RAG 2026 — Step 4: Cross-Encoder Reranking")
    print("=" * 70)
    
    # --- Determine input file ---
    if args.input == "bm25_combined":
        input_file = TMP_DIR / "bm25_combined_results.json"
    else:
        input_file = TMP_DIR / "bm25_results.json"
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print(f"Run 01_bm25_retrieve.py first")
        sys.exit(1)
    
    # --- Load cached BM25 results ---
    print(f"Loading BM25 results from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        bm25_results = json.load(f)
    
    print(f"Loaded results for {len(bm25_results)} topics")
    
    # --- Load original queries (needed for cross-encoder) ---
    queries_file = Path("trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv")
    queries = {}
    if queries_file.exists():
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0].strip()] = parts[1].strip()
    else:
        print(f"WARNING: Original queries file not found. Using empty queries.")
    
    print(f"Loaded {len(queries)} original queries")
    
    # --- Check for documents with text ---
    sample_topic = next(iter(bm25_results))
    sample_doc = bm25_results[sample_topic][0] if bm25_results[sample_topic] else {}
    has_doc_text = bool(sample_doc.get("doc", ""))
    
    if not has_doc_text:
        print("\nWARNING: BM25 results don't contain document text ('doc' field).")
        print("Reranking requires document text. The BM25 results were either:")
        print("  1. Not cached with doc text, or")
        print("  2. The API didn't return doc text")
        print("\nSkipping reranking. Use the BM25 baseline as your submission.")
        sys.exit(1)
    
    # --- Rerank ---
    if args.flashrank:
        reranked = rerank_with_flashrank(queries, bm25_results, RERANK_DEPTH)
        run_id = RUN_ID.replace("bm25", "flashrank")
    else:
        reranked = rerank_with_cross_encoder(
            queries, bm25_results, RERANK_MODEL,
            use_gpu=args.gpu, rerank_depth=RERANK_DEPTH
        )
        run_id = RUN_ID.replace("bm25", "reranked")
    
    # --- Write run file ---
    write_trec_run(reranked, BM25_RERANKED_RUN, run_id)
    
    print(f"\nReranking complete!")
    print(f"Next: Run 07_validate_run.py to validate before submission")


if __name__ == "__main__":
    main()

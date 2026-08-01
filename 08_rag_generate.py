"""
Step 8: RAG Answer Generation
===============================
Generates cited answers for the TREC RAG 2026 RAG task.

For each of the 119 narratives:
  1. Takes top-K reranked documents as evidence
  2. Sends narrative + evidence to a local LLM on GPU
  3. Parses LLM response into cited answer sentences
  4. Outputs official TREC RAG JSONL format

Usage:
    python 08_rag_generate.py [--gpu] [--evidence-depth 20] [--resume]

Output:
    runs/rag_output_trec_rag_2026.jsonl

Prerequisites:
    - BM25 retrieval completed (tmp/bm25_results.json must exist)
    - A reranked run file (runs/bm25_reranked.txt) or BM25 run (runs/bm25_raw.txt)
    - pip install transformers accelerate bitsandbytes
    - GPU with >= 16GB VRAM recommended
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    QUERIES_FILE, OUTPUT_DIR, TMP_DIR, BM25_RAW_RUN, BM25_RERANKED_RUN,
    RAG_RUN_ID, RAG_EVIDENCE_DEPTH, RAG_MODEL_NAME, RAG_MAX_NEW_TOKENS,
    RAG_OUTPUT_FILE, RAG_DOC_MAX_CHARS, RAG_TEAM_ID
)


# =============================================================================
# Prompt Template
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant writing answers for the TREC RAG 2026 evaluation.
You MUST follow these rules EXACTLY:
1. Write a comprehensive answer addressing ALL aspects of the narrative.
2. Write multiple sentences (aim for 5-15 sentences).
3. EVERY factual sentence MUST cite at least one evidence document using [N] notation.
4. Citation numbers refer to the evidence document numbers provided.
5. Only cite a document if it genuinely supports your statement.
6. Be thorough, accurate, and concise.
7. If the evidence does not cover some aspect, say so explicitly."""

USER_PROMPT_TEMPLATE = """NARRATIVE:
{narrative}

EVIDENCE DOCUMENTS:
{evidence}

Write a comprehensive, well-cited answer addressing all aspects of the narrative above. Cite evidence using [N] notation."""


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_narratives(filepath: Path) -> list[tuple[str, str]]:
    """Load narratives from TSV file. Returns [(narrative_id, narrative_text), ...]."""
    narratives = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            narratives.append((parts[0].strip(), parts[1].strip()))
    print(f"[DATA] Loaded {len(narratives)} narratives from {filepath}")
    return narratives


def load_run_file(filepath: Path, max_per_topic: int = 1000) -> dict[str, list[str]]:
    """
    Load a TREC run file and return {topic_id: [docid1, docid2, ...]} in ranked order.
    """
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            topic_id, _, docid, rank, score, _ = parts
            if topic_id not in results:
                results[topic_id] = []
            if len(results[topic_id]) < max_per_topic:
                results[topic_id].append(docid)
    print(f"[DATA] Loaded run file {filepath}: {len(results)} topics")
    return results


def load_doc_texts(json_cache: Path, needed_docids: set) -> dict[str, str]:
    """
    Load document texts from the BM25 results JSON cache.
    Only keeps documents in the needed_docids set to save memory.
    """
    print(f"[DATA] Loading BM25 results cache from {json_cache}")
    print(f"[DATA] File size: {json_cache.stat().st_size / (1024**3):.1f} GB")
    print(f"[DATA] Need text for {len(needed_docids)} unique documents")
    print(f"[DATA] This may take 1-2 minutes for large files...")

    start = time.time()
    with open(json_cache, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    doc_text_lookup = {}
    for topic_id, candidates in all_results.items():
        for c in candidates:
            docid = c.get("docid", "")
            if docid in needed_docids and docid not in doc_text_lookup:
                doc_text = c.get("doc", "")
                if isinstance(doc_text, dict):
                    doc_text = doc_text.get("contents", doc_text.get("text", str(doc_text)))
                doc_text_lookup[docid] = str(doc_text)

    del all_results  # Free memory
    elapsed = time.time() - start
    print(f"[DATA] Loaded text for {len(doc_text_lookup)}/{len(needed_docids)} documents in {elapsed:.1f}s")

    missing = needed_docids - set(doc_text_lookup.keys())
    if missing:
        print(f"[WARN] Missing text for {len(missing)} documents (will skip these)")

    return doc_text_lookup


def load_already_processed(output_file: Path) -> set[str]:
    """Load narrative IDs already processed (for resume support)."""
    done = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    nid = obj.get("metadata", {}).get("narrative_id", "")
                    if nid:
                        done.add(nid)
                except json.JSONDecodeError:
                    continue
    return done


# =============================================================================
# LLM Functions
# =============================================================================
def load_model(model_name: str, use_gpu: bool = True):
    """Load the LLM with 4-bit quantization for GPU inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'=' * 70}")
    print(f"[MODEL] Loading {model_name}")

    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"[MODEL] GPU: {gpu_name} ({vram:.1f} GB VRAM)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[MODEL] Using 4-bit quantization (NF4)")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
    else:
        device = "cpu"
        print(f"[MODEL] Running on CPU (will be slow!)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Print model memory usage
    if use_gpu and torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / (1024**3)
        print(f"[MODEL] GPU memory used: {mem_used:.1f} GB")

    print(f"[MODEL] Model loaded successfully on {device}")
    print(f"{'=' * 70}\n")

    return model, tokenizer


def build_prompt(narrative: str, evidence_docs: list[tuple[str, str]], max_chars: int) -> str:
    """
    Build the user prompt with narrative and numbered evidence documents.

    Args:
        narrative: The narrative text
        evidence_docs: List of (docid, doc_text) tuples
        max_chars: Maximum characters per document
    """
    evidence_parts = []
    for i, (docid, doc_text) in enumerate(evidence_docs, start=1):
        truncated = doc_text[:max_chars].strip()
        if len(doc_text) > max_chars:
            truncated += "..."
        evidence_parts.append(f"[{i}] {truncated}")

    evidence_str = "\n\n".join(evidence_parts)
    return USER_PROMPT_TEMPLATE.format(narrative=narrative, evidence=evidence_str)


def generate_answer(model, tokenizer, narrative: str, evidence_docs: list[tuple[str, str]],
                    max_chars: int, max_new_tokens: int) -> str:
    """Generate a cited answer using the LLM."""
    import torch

    user_prompt = build_prompt(narrative, evidence_docs, max_chars)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback for models without chat template
        text = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user_prompt}\n<|assistant|>\n"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=28000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


# =============================================================================
# Answer Parsing
# =============================================================================
def parse_answer_into_sentences(raw_text: str, num_references: int) -> list[dict]:
    """
    Parse LLM-generated text into structured sentences with citation indices.

    Returns:
        [{"text": "Sentence text [1][2].", "citations": [0, 1]}, ...]
    """
    # Clean up the raw text
    text = raw_text.strip()

    # Split into sentences (handle common patterns)
    # Split on period/exclamation/question followed by space or newline, but not inside [N]
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)

    # If splitting produced nothing useful, split on newlines
    if len(sentences) <= 1 and "\n" in text:
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 5:
            continue

        # Extract citation numbers [1], [2], etc.
        citation_nums = re.findall(r'\[(\d+)\]', sent)
        citation_indices = []
        for c in citation_nums:
            idx = int(c) - 1  # Convert 1-indexed to 0-indexed
            if 0 <= idx < num_references:
                citation_indices.append(idx)
        citation_indices = sorted(set(citation_indices))

        result.append({
            "text": sent,
            "citations": citation_indices
        })

    # If no sentences parsed, create a single entry from the whole text
    if not result and text:
        citation_nums = re.findall(r'\[(\d+)\]', text)
        citation_indices = sorted(set(
            int(c) - 1 for c in citation_nums
            if 0 <= int(c) - 1 < num_references
        ))
        result.append({
            "text": text[:2000],
            "citations": citation_indices
        })

    return result


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="TREC RAG 2026 — RAG Answer Generation")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="Use GPU for inference (default: True)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference")
    parser.add_argument("--evidence-depth", type=int, default=RAG_EVIDENCE_DEPTH,
                        help=f"Number of top documents to use as evidence (default: {RAG_EVIDENCE_DEPTH})")
    parser.add_argument("--model", type=str, default=RAG_MODEL_NAME,
                        help=f"HuggingFace model name (default: {RAG_MODEL_NAME})")
    parser.add_argument("--max-tokens", type=int, default=RAG_MAX_NEW_TOKENS,
                        help=f"Max new tokens to generate (default: {RAG_MAX_NEW_TOKENS})")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from previously generated answers (default: True)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file path")
    args = parser.parse_args()

    use_gpu = args.gpu and not args.cpu
    output_file = Path(args.output) if args.output else RAG_OUTPUT_FILE

    print("=" * 70)
    print("TREC RAG 2026 — Step 8: RAG Answer Generation")
    print("=" * 70)
    print(f"  Model:          {args.model}")
    print(f"  Evidence depth: {args.evidence_depth} docs per narrative")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  GPU:            {'Yes' if use_gpu else 'No (CPU)'}")
    print(f"  Output:         {output_file}")
    print(f"  Team ID:        {RAG_TEAM_ID}")
    print(f"  Run ID:         {RAG_RUN_ID}")
    print(f"  Resume:         {args.resume}")
    print()

    # --- Step 1: Load narratives ---
    if not QUERIES_FILE.exists():
        print(f"[ERROR] Queries file not found: {QUERIES_FILE}")
        print(f"  Fix: git clone https://github.com/TREC-RAG/trec-rag-data.git")
        sys.exit(1)

    narratives = load_narratives(QUERIES_FILE)
    narrative_dict = {nid: ntxt for nid, ntxt in narratives}

    # --- Step 2: Check for resume ---
    already_done = set()
    if args.resume:
        already_done = load_already_processed(output_file)
        if already_done:
            print(f"[RESUME] Found {len(already_done)} already-processed narratives")

    remaining = [(nid, ntxt) for nid, ntxt in narratives if nid not in already_done]
    print(f"[INFO] {len(remaining)} narratives remaining to process")

    if not remaining:
        print("[DONE] All narratives already processed!")
        sys.exit(0)

    # --- Step 3: Find best run file for document ordering ---
    run_file = None
    for candidate_file, label in [
        (BM25_RERANKED_RUN, "Reranked"),
        (OUTPUT_DIR / "rrf_fused.txt", "RRF Fused"),
        (BM25_RAW_RUN, "BM25 Raw"),
    ]:
        if candidate_file.exists():
            run_file = candidate_file
            print(f"[DATA] Using {label} run file: {candidate_file}")
            break

    if run_file is None:
        print("[ERROR] No run file found. Run the retrieval pipeline first (01_bm25_retrieve.py)")
        sys.exit(1)

    ranked_docs = load_run_file(run_file, max_per_topic=args.evidence_depth)

    # --- Step 4: Collect needed docids and load texts ---
    bm25_cache = TMP_DIR / "bm25_results.json"
    if not bm25_cache.exists():
        print(f"[ERROR] BM25 results cache not found: {bm25_cache}")
        print(f"  Fix: Run 01_bm25_retrieve.py first")
        sys.exit(1)

    needed_docids = set()
    for nid, _ in remaining:
        if nid in ranked_docs:
            needed_docids.update(ranked_docs[nid][:args.evidence_depth])

    doc_text_lookup = load_doc_texts(bm25_cache, needed_docids)

    # --- Step 5: Load LLM ---
    model, tokenizer = load_model(args.model, use_gpu=use_gpu)

    # --- Step 6: Generate answers ---
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(remaining)
    successes = 0
    failures = 0
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"[GEN] Starting answer generation for {total} narratives...")
    print(f"[GEN] Estimated time: {total * 2:.0f}-{total * 5:.0f} minutes")
    print(f"{'=' * 70}\n")

    for idx, (narrative_id, narrative_text) in enumerate(remaining, start=1):
        iter_start = time.time()

        print(f"\n--- [{idx}/{total}] {narrative_id} ---")
        print(f"  Narrative: {narrative_text[:120]}...")

        # Get top-K evidence documents
        top_docids = ranked_docs.get(narrative_id, [])[:args.evidence_depth]

        if not top_docids:
            print(f"  [WARN] No ranked documents found for {narrative_id}, skipping")
            failures += 1
            continue

        evidence_docs = []
        for docid in top_docids:
            text = doc_text_lookup.get(docid, "")
            if text and len(text.strip()) > 10:
                evidence_docs.append((docid, text))

        if not evidence_docs:
            print(f"  [WARN] No document text found for {narrative_id}, skipping")
            failures += 1
            continue

        print(f"  Evidence: {len(evidence_docs)} documents")

        # Generate answer
        try:
            raw_answer = generate_answer(
                model, tokenizer, narrative_text, evidence_docs,
                max_chars=RAG_DOC_MAX_CHARS, max_new_tokens=args.max_tokens
            )
        except Exception as e:
            print(f"  [ERROR] Generation failed: {e}")
            failures += 1
            continue

        # Parse answer
        answer_sentences = parse_answer_into_sentences(raw_answer, len(evidence_docs))

        # Build references list (docids of the evidence documents used)
        references = [docid for docid, _ in evidence_docs]

        # Build output object
        output_obj = {
            "metadata": {
                "team_id": RAG_TEAM_ID,
                "narrative_id": narrative_id,
                "narrative": narrative_text,
                "run_id": RAG_RUN_ID,
                "run_desc": f"BM25 top-1000 + cross-encoder reranking + {args.model} cited answer generation",
                "type": "automatic"
            },
            "references": references,
            "answer": answer_sentences
        }

        # Append to JSONL (incremental save)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

        successes += 1
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        eta = (elapsed / idx) * (total - idx)

        # Print summary for this narrative
        n_sentences = len(answer_sentences)
        n_cited = sum(1 for s in answer_sentences if s["citations"])
        print(f"  Answer: {n_sentences} sentences, {n_cited} with citations")
        print(f"  Preview: {answer_sentences[0]['text'][:100]}..." if answer_sentences else "  Preview: (empty)")
        print(f"  Time: {iter_time:.1f}s | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    # --- Step 7: Summary ---
    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"RAG Answer Generation Complete!")
    print(f"  Successful: {successes}/{total}")
    print(f"  Failed:     {failures}/{total}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Output:     {output_file}")
    print(f"{'=' * 70}")

    if successes > 0:
        print(f"\nNext step: Run  python 09_validate_rag.py  to validate the output")
    else:
        print(f"\n[ERROR] No answers generated! Check errors above.")


if __name__ == "__main__":
    main()

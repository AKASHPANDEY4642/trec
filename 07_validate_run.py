"""
Step 7: Validate TREC Run File Before Submission
===================================================
Performs comprehensive validation of your run file to make sure it
won't be rejected at submission. Checks:

1. File format (6 space-separated columns per line)
2. All 119 topic IDs present (rag2026-0 through rag2026-118)
3. Q0 constant in column 2
4. Ranks are sequential per topic (1, 2, 3, ...)
5. Scores are in non-increasing order per topic
6. No duplicate (topic_id, docid) pairs
7. Run ID is consistent across all lines
8. Total line count is reasonable

Usage:
    python 07_validate_run.py [path_to_run_file]

Default:
    Validates runs/bm25_raw.txt
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR


# All expected topic IDs
EXPECTED_TOPICS = {f"rag2026-{i}" for i in range(119)}


def validate_run_file(filepath: Path) -> tuple[bool, list[str], list[str]]:
    """
    Validate a TREC run file.
    
    Returns:
        (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    if not filepath.exists():
        errors.append(f"File not found: {filepath}")
        return False, errors, warnings
    
    # Parse the file
    topics_seen = defaultdict(list)  # topic_id -> list of (docid, rank, score, run_id)
    line_count = 0
    run_ids = set()
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            parts = line.split()
            
            # Check column count
            if len(parts) != 6:
                errors.append(f"Line {line_num}: Expected 6 columns, got {len(parts)}: {line[:80]}")
                continue
            
            topic_id, q0, docid, rank_str, score_str, run_id = parts
            
            # Check Q0
            if q0 != "Q0":
                errors.append(f"Line {line_num}: Column 2 should be 'Q0', got '{q0}'")
            
            # Check rank is integer
            try:
                rank = int(rank_str)
            except ValueError:
                errors.append(f"Line {line_num}: Rank '{rank_str}' is not an integer")
                rank = 0
            
            # Check score is float
            try:
                score = float(score_str)
            except ValueError:
                errors.append(f"Line {line_num}: Score '{score_str}' is not a float")
                score = 0.0
            
            topics_seen[topic_id].append((docid, rank, score, run_id))
            run_ids.add(run_id)
    
    # --- Check all topics present ---
    topics_found = set(topics_seen.keys())
    missing_topics = EXPECTED_TOPICS - topics_found
    extra_topics = topics_found - EXPECTED_TOPICS
    
    if missing_topics:
        errors.append(f"Missing {len(missing_topics)} topics: {sorted(missing_topics)[:10]}...")
    
    if extra_topics:
        warnings.append(f"Extra {len(extra_topics)} unexpected topics: {sorted(extra_topics)[:10]}...")
    
    # --- Check per-topic consistency ---
    for topic_id, entries in topics_seen.items():
        # Check for duplicate docids
        docids = [e[0] for e in entries]
        dup_docids = [d for d in set(docids) if docids.count(d) > 1]
        if dup_docids:
            errors.append(f"Topic {topic_id}: Duplicate docids: {dup_docids[:5]}")
        
        # Check ranks are sequential
        ranks = [e[1] for e in entries]
        expected_ranks = list(range(1, len(entries) + 1))
        if ranks != expected_ranks:
            warnings.append(f"Topic {topic_id}: Ranks not sequential 1..{len(entries)}")
        
        # Check scores are non-increasing
        scores = [e[2] for e in entries]
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1]:
                warnings.append(f"Topic {topic_id}: Scores not non-increasing at rank {i+1}")
                break
    
    # --- Check run ID consistency ---
    if len(run_ids) > 1:
        warnings.append(f"Multiple run IDs found: {run_ids}")
    
    # --- Check total line count ---
    if line_count < 119:
        errors.append(f"Only {line_count} lines. Should have at least 119 (1 per topic)")
    
    if line_count > 119 * 1000:
        warnings.append(f"Very large file: {line_count} lines. Consider reducing to top 100 per topic.")
    
    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def main():
    print("=" * 70)
    print("TREC RAG 2026 — Step 7: Validate Run File")
    print("=" * 70)
    
    # Determine which file to validate
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        # Try to find the best run file
        candidates = [
            OUTPUT_DIR / "bm25_reranked.txt",
            OUTPUT_DIR / "rrf_fused.txt",
            OUTPUT_DIR / "query_rewritten_bm25.txt",
            OUTPUT_DIR / "bm25_raw.txt",
            OUTPUT_DIR / "final_submission.txt",
        ]
        filepath = None
        for c in candidates:
            if c.exists():
                filepath = c
                break
        
        if filepath is None:
            print("ERROR: No run files found in runs/")
            print("Run 01_bm25_retrieve.py first")
            sys.exit(1)
    
    print(f"\nValidating: {filepath}")
    is_valid, errors, warnings = validate_run_file(filepath)
    
    # --- Report ---
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  • {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print(f"  • {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more warnings")
    
    if is_valid:
        print(f"\n✅ VALID — Run file is ready for submission!")
        print(f"\n--- Stats ---")
        # Quick stats
        from collections import Counter
        topic_counts = Counter()
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:
                    topic_counts[parts[0]] += 1
        
        counts = list(topic_counts.values())
        print(f"  Topics: {len(topic_counts)}")
        print(f"  Total lines: {sum(counts)}")
        print(f"  Docs per topic: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
        print(f"  Run ID: {parts[5] if parts else 'N/A'}")
        
        print(f"\n📤 Submission steps:")
        print(f"  1. Go to https://ir.nist.gov/evalbase (TREC submission portal)")
        print(f"  2. Log in with your TREC credentials")
        print(f"  3. Upload: {filepath}")
        print(f"  4. Mark as task: Retrieval (R)")
    else:
        print(f"\n❌ INVALID — Fix the errors above before submission")


if __name__ == "__main__":
    main()

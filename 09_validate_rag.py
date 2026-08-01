"""
Step 9: Validate RAG Output
=============================
Validates the RAG JSONL output file for TREC RAG 2026 submission compliance.

Checks:
  - Valid JSONL format
  - All 119 narrative IDs present
  - Required metadata fields
  - Citation indices within bounds
  - Narrative text matches official test data
  - Each answer has at least one sentence

Usage:
    python 09_validate_rag.py [path/to/rag_output.jsonl]

Output:
    Prints validation report to stdout
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import QUERIES_FILE, RAG_OUTPUT_FILE


def load_official_narratives(filepath: Path) -> dict[str, str]:
    """Load official narrative IDs and texts."""
    narratives = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                narratives[parts[0].strip()] = parts[1].strip()
    return narratives


def validate_rag_output(output_file: Path, official_narratives: dict[str, str]) -> bool:
    """
    Validate a RAG JSONL output file.
    Returns True if all checks pass.
    """
    print("=" * 70)
    print("TREC RAG 2026 — RAG Output Validation")
    print("=" * 70)
    print(f"  File: {output_file}")
    print()

    if not output_file.exists():
        print(f"[FAIL] Output file does not exist: {output_file}")
        return False

    errors = []
    warnings = []
    seen_ids = set()
    total_sentences = 0
    total_citations = 0
    total_references = 0
    run_ids = set()
    team_ids = set()

    with open(output_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # Check valid JSON
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON — {e}")
                continue

            # Check metadata
            meta = obj.get("metadata", {})
            if not meta:
                errors.append(f"Line {line_num}: Missing 'metadata' field")
                continue

            narrative_id = meta.get("narrative_id", "")
            if not narrative_id:
                errors.append(f"Line {line_num}: Missing 'metadata.narrative_id'")
                continue

            # Check for duplicates
            if narrative_id in seen_ids:
                errors.append(f"Line {line_num}: Duplicate narrative_id '{narrative_id}'")
            seen_ids.add(narrative_id)

            # Check required metadata fields
            for field in ["team_id", "narrative_id", "narrative", "run_id"]:
                if field not in meta or not meta[field]:
                    errors.append(f"Line {line_num} ({narrative_id}): Missing metadata.{field}")

            if "run_id" in meta:
                run_ids.add(meta["run_id"])
            if "team_id" in meta:
                team_ids.add(meta["team_id"])

            # Check narrative text matches official data
            official_text = official_narratives.get(narrative_id, "")
            submitted_text = meta.get("narrative", "")
            if official_text and submitted_text and official_text != submitted_text:
                # Allow minor whitespace differences
                if official_text.strip() != submitted_text.strip():
                    warnings.append(
                        f"Line {line_num} ({narrative_id}): Narrative text doesn't match official data"
                    )

            # Check references
            references = obj.get("references", [])
            if not isinstance(references, list):
                errors.append(f"Line {line_num} ({narrative_id}): 'references' is not a list")
                continue

            if not references:
                warnings.append(f"Line {line_num} ({narrative_id}): Empty references list")

            total_references += len(references)

            # Check answer
            answer = obj.get("answer", [])
            if not isinstance(answer, list):
                errors.append(f"Line {line_num} ({narrative_id}): 'answer' is not a list")
                continue

            if not answer:
                errors.append(f"Line {line_num} ({narrative_id}): Empty answer (no sentences)")
                continue

            for sent_idx, sent in enumerate(answer):
                if not isinstance(sent, dict):
                    errors.append(f"Line {line_num} ({narrative_id}): answer[{sent_idx}] is not a dict")
                    continue

                if "text" not in sent:
                    errors.append(f"Line {line_num} ({narrative_id}): answer[{sent_idx}] missing 'text'")

                if "citations" not in sent:
                    errors.append(f"Line {line_num} ({narrative_id}): answer[{sent_idx}] missing 'citations'")
                    continue

                citations = sent.get("citations", [])
                if not isinstance(citations, list):
                    errors.append(
                        f"Line {line_num} ({narrative_id}): answer[{sent_idx}].citations is not a list"
                    )
                    continue

                # Check citation indices are within bounds
                for cit_idx in citations:
                    if not isinstance(cit_idx, int):
                        errors.append(
                            f"Line {line_num} ({narrative_id}): citation '{cit_idx}' is not an integer"
                        )
                    elif cit_idx < 0 or cit_idx >= len(references):
                        errors.append(
                            f"Line {line_num} ({narrative_id}): citation index {cit_idx} "
                            f"out of bounds (references has {len(references)} items)"
                        )

                total_citations += len(citations)

            total_sentences += len(answer)

    # --- Check coverage ---
    missing_ids = set(official_narratives.keys()) - seen_ids
    extra_ids = seen_ids - set(official_narratives.keys())

    if missing_ids:
        errors.append(
            f"Missing {len(missing_ids)} narrative(s): "
            + ", ".join(sorted(missing_ids)[:10])
            + ("..." if len(missing_ids) > 10 else "")
        )

    if extra_ids:
        warnings.append(f"Extra {len(extra_ids)} narrative ID(s) not in official data")

    # --- Check run_id format ---
    for rid in run_ids:
        if len(rid) > 12:
            warnings.append(f"run_id '{rid}' exceeds 12 characters (TREC limit)")
        if not rid.isalnum() and not all(c.isalnum() or c == '_' for c in rid):
            warnings.append(f"run_id '{rid}' contains non-alphanumeric characters")

    # --- Print Report ---
    print(f"\n{'=' * 70}")
    print(f"VALIDATION REPORT")
    print(f"{'=' * 70}")
    print(f"  Narratives found:    {len(seen_ids)}/119")
    print(f"  Total sentences:     {total_sentences}")
    print(f"  Avg sentences/narr:  {total_sentences / max(len(seen_ids), 1):.1f}")
    print(f"  Total citations:     {total_citations}")
    print(f"  Total references:    {total_references}")
    print(f"  Avg refs/narrative:  {total_references / max(len(seen_ids), 1):.1f}")
    print(f"  Team ID(s):          {', '.join(team_ids) if team_ids else 'MISSING'}")
    print(f"  Run ID(s):           {', '.join(run_ids) if run_ids else 'MISSING'}")
    print()

    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:20]:
            print(f"    ✗ {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print(f"  ERRORS: 0 ✓")

    if warnings:
        print(f"  WARNINGS: {len(warnings)}")
        for w in warnings[:10]:
            print(f"    ⚠ {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings) - 10} more")
    else:
        print(f"  WARNINGS: 0 ✓")

    print()

    if not errors:
        print(f"  ✓ VALIDATION PASSED — Ready for submission!")
        print(f"  Submit to: https://ir.nist.gov/evalbase")
        return True
    else:
        print(f"  ✗ VALIDATION FAILED — Fix {len(errors)} error(s) before submission")
        return False


def main():
    # Determine input file
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
    else:
        output_file = RAG_OUTPUT_FILE

    # Load official narratives
    if not QUERIES_FILE.exists():
        print(f"[ERROR] Official queries file not found: {QUERIES_FILE}")
        print(f"  Fix: git clone https://github.com/TREC-RAG/trec-rag-data.git")
        sys.exit(1)

    official_narratives = load_official_narratives(QUERIES_FILE)
    print(f"[DATA] Loaded {len(official_narratives)} official narratives")

    # Validate
    success = validate_rag_output(output_file, official_narratives)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

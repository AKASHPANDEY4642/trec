"""
TREC RAG 2026 Retrieval — Main Pipeline Runner
================================================
Runs the complete retrieval pipeline step by step.
Use this if you want to run everything in sequence.

Usage:
    python run_pipeline.py [--steps 1,2,3,4,7] [--gpu] [--flashrank]

Steps:
    1 = BM25 retrieval (raw narratives)
    2 = Query processing
    3 = BM25 retrieval (processed queries)
    4 = Cross-encoder reranking
    5 = RRF fusion
    6 = Spot check
    7 = Validate run file

Default: Steps 1,7 (minimum viable submission)
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_step(script: str, extra_args: list[str] = None):
    """Run a pipeline step as a subprocess."""
    cmd = [sys.executable, script] + (extra_args or [])
    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode != 0:
        print(f"\n❌ Step failed with exit code {result.returncode}")
        print(f"Fix the issue and re-run this step")
        return False
    
    print(f"\n✅ Step completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="TREC RAG 2026 Retrieval Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Step combinations:
  Minimum:  --steps 1,7         BM25 only (valid submission)
  Good:     --steps 1,4,7       BM25 + reranking
  Better:   --steps 1,2,3,4,7   BM25 + query processing + reranking
  Best:     --steps 1,2,3,4,5,7 All steps including fusion
        """
    )
    parser.add_argument("--steps", default="1,7",
                        help="Comma-separated step numbers to run (default: 1,7)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for reranking")
    parser.add_argument("--flashrank", action="store_true",
                        help="Use FlashRank for reranking (CPU-friendly)")
    args = parser.parse_args()
    
    steps = [int(s.strip()) for s in args.steps.split(",")]
    
    print("=" * 70)
    print("TREC RAG 2026 — Retrieval Pipeline")
    print("=" * 70)
    print(f"Steps to run: {steps}")
    print(f"GPU: {args.gpu}")
    print(f"FlashRank: {args.flashrank}")
    
    step_map = {
        1: ("01_bm25_retrieve.py", []),
        2: ("02_query_processing.py", []),
        3: ("03_bm25_retrieve_processed.py", []),
        4: ("04_rerank.py", []),
        5: ("05_rrf_fusion.py", []),
        6: ("06_spot_check.py", []),
        7: ("07_validate_run.py", []),
    }
    
    # Add reranking flags
    if 4 in steps:
        rerank_args = []
        if args.gpu:
            rerank_args.append("--gpu")
        if args.flashrank:
            rerank_args.append("--flashrank")
        step_map[4] = ("04_rerank.py", rerank_args)
    
    for step_num in steps:
        if step_num not in step_map:
            print(f"Unknown step: {step_num}")
            continue
        
        script, extra_args = step_map[step_num]
        success = run_step(script, extra_args)
        
        if not success:
            print(f"\n⚠️  Pipeline stopped at step {step_num}")
            print(f"Fix the issue and re-run with: --steps {','.join(str(s) for s in steps if s >= step_num)}")
            sys.exit(1)
    
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete!")
    print(f"{'=' * 70}")
    print(f"\nYour run files are in: runs/")
    print(f"Next: Submit to TREC at https://ir.nist.gov/evalbase")


if __name__ == "__main__":
    main()

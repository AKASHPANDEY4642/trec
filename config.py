"""
TREC RAG 2026 Retrieval Task — Configuration
=============================================
All constants and settings for the retrieval pipeline.
Edit RUN_ID to your team name before submission.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API token from .env.local
load_dotenv(".env.local")

# =============================================================================
# Pyserini REST API Configuration
# =============================================================================
API_BASE_URL = "http://api.castorini.uwaterloo.ca"
INDEX_NAME = "climbmix-400b"
API_TOKEN = os.getenv("PYSERINI_API_TOKEN", "")

# =============================================================================
# Data Paths
# =============================================================================
PROJECT_DIR = Path(__file__).parent
QUERIES_FILE = PROJECT_DIR / "trec-rag-data" / "trec-rag-2026" / "test-data" / "trec_rag_2026_queries.tsv"
OUTPUT_DIR = PROJECT_DIR / "runs"
TMP_DIR = PROJECT_DIR / "tmp"

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

# =============================================================================
# Retrieval Settings
# =============================================================================
# CHANGE THIS to your team name!
RUN_ID = "IIITDMK_bm25"

# Number of documents to retrieve per query from BM25
BM25_HITS = 1000

# Number of top documents to rerank with cross-encoder
RERANK_DEPTH = 1000

# =============================================================================
# Reranking Model
# =============================================================================
# Option 1: Fast, CPU-friendly (33M params)
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Option 2: Higher quality, needs GPU (568M params)
# RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Option 3: Good balance (110M params)
# RERANK_MODEL = "Alibaba-NLP/gte-reranker-base"

# =============================================================================
# Rate Limiting (be polite to the API)
# =============================================================================
REQUEST_DELAY_SECONDS = 1.0  # Delay between API requests
MAX_RETRIES = 3              # Retry failed requests up to 3 times
RETRY_DELAY_SECONDS = 5.0    # Wait between retries

# =============================================================================
# Output File Paths
# =============================================================================
BM25_RAW_RUN = OUTPUT_DIR / "bm25_raw.txt"
BM25_RERANKED_RUN = OUTPUT_DIR / "bm25_reranked.txt"
QUERY_REWRITTEN_RUN = OUTPUT_DIR / "query_rewritten_bm25.txt"
FINAL_SUBMISSION_RUN = OUTPUT_DIR / "final_submission.txt"

# =============================================================================
# RAG Task Settings
# =============================================================================
RAG_TEAM_ID = "IIITDMK"
RAG_RUN_ID = "IIITDMK_rag"
RAG_EVIDENCE_DEPTH = 20             # Top-K docs to use as evidence per narrative
RAG_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Local LLM for answer generation
RAG_MAX_NEW_TOKENS = 2048           # Max tokens in generated answer
RAG_DOC_MAX_CHARS = 4000            # Max chars per evidence document in prompt
RAG_OUTPUT_FILE = OUTPUT_DIR / "rag_output_trec_rag_2026.jsonl"

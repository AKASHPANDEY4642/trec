# Delivery Checklist — TREC RAG 2026 Retrieval Task

## User Requirements vs Deliverables

| # | User Asked For | Deliverable | Status |
|---|---------------|-------------|--------|
| 1 | "tell me step by step code by code what things I need to do" | [walkthrough.md](file:///C:/Users/kash/.gemini/antigravity/brain/1bbad416-2857-43b3-8fbe-440453111565/walkthrough.md) — 9-step guide | ✅ Done |
| 2 | "which file to use and what code to write" | 12 files created in [project dir](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/) | ✅ Done |
| 3 | "the whole pipeline in detail" | 7-step pipeline: BM25 → Query Processing → Reranking → RRF → Validation → Submit | ✅ Done |
| 4 | "what all github repo to clone" | `git clone https://github.com/TREC-RAG/trec-rag-data.git` (in Step 3) | ✅ Documented |
| 5 | "what model to use" | Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-12-v2` (CPU) or `BAAI/bge-reranker-v2-m3` (GPU) | ✅ In [config.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/config.py) |
| 6 | "what embedding to use" | BM25 (sparse) via Pyserini REST API — no embedding needed for baseline; FlashRank for reranking | ✅ In code |
| 7 | "what setup I need" | Python 3.10+, pip, virtual env, API token, TREC registration | ✅ In walkthrough |
| 8 | "what lib to use" | requests, python-dotenv, tqdm, sentence-transformers/flashrank, torch | ✅ In [requirements.txt](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/requirements.txt) |
| 9 | "what dataset to download" | ClimbMix-400b via API (no download needed); queries from trec-rag-data repo | ✅ Documented |
| 10 | "which line to change" | `RUN_ID` in config.py, `PYSERINI_API_TOKEN` in .env.local | ✅ Highlighted |
| 11 | "end to end till my first submission" | Steps 0-9 in walkthrough cover registration → code → validate → submit | ✅ Done |
| 12 | "a good level of submission" | BM25 + cross-encoder reranking = competitive baseline (not just minimum) | ✅ In pipeline |
| 13 | "retrieval only, not RAG" | Pipeline outputs standard TREC run file (6-column format), not RAG JSON | ✅ Verified |
| 14 | "baseline task like trec 2024/2025" | BM25 + reranking follows the standard TREC baseline approach | ✅ Done |

## Code File Verification

| File | Exists | Key Functionality |
|------|--------|-------------------|
| [config.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/config.py) | ✅ 2.7 KB | API URL, index name, model config, paths |
| [01_bm25_retrieve.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/01_bm25_retrieve.py) | ✅ 8.9 KB | Loads queries, calls API, writes TREC run |
| [02_query_processing.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/02_query_processing.py) | ✅ 7.5 KB | First-sentence, keyword, core-question extraction |
| [03_bm25_retrieve_processed.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/03_bm25_retrieve_processed.py) | ✅ 3.4 KB | BM25 with cleaned queries |
| [04_rerank.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/04_rerank.py) | ✅ 9.6 KB | Cross-encoder + FlashRank reranking |
| [05_rrf_fusion.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/05_rrf_fusion.py) | ✅ 4.4 KB | RRF fusion of multiple ranked lists |
| [06_spot_check.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/06_spot_check.py) | ✅ 4.0 KB | Manual quality inspection |
| [07_validate_run.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/07_validate_run.py) | ✅ 7.1 KB | Format validation, all 119 topics, scoring order |
| [run_pipeline.py](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/run_pipeline.py) | ✅ 3.6 KB | Orchestrator for all steps |
| [requirements.txt](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/requirements.txt) | ✅ 385 B | All pip dependencies |
| [README.md](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/README.md) | ✅ 7.1 KB | Complete documentation |
| [.gitignore](file:///C:/Users/kash/.gemini/antigravity/scratch/trec-rag-2026-retrieval/.gitignore) | ✅ 214 B | Excludes secrets, venv, output |

## Blockers (User Action Required)

> [!CAUTION]
> These cannot be automated and must be done by you:

1. **Install Python 3.12** — https://www.python.org/downloads/
2. **Email for API token** — `get-pyserini@googlegroups.com` (can take 1-2 days)
3. **Register for TREC** — https://trec.nist.gov/cfp.html
4. **Edit RUN_ID** in config.py to your team name

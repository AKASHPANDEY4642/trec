# TREC RAG 2026 — Retrieval Task: End-to-End Submission Plan

> **Goal:** Submit a competitive retrieval run for the TREC RAG 2026 Retrieval (R) task before the **August 8, 2026** deadline.

## Background & Task Summary

- **Task:** Given 119 narrative queries (`rag2026-0` through `rag2026-118`), retrieve and rank the most relevant segments from the **ClimbMix-400b** corpus.
- **Corpus:** NVIDIA ClimbMix-400b (~553M documents, 400B tokens). Accessed via the **Pyserini REST API** at `http://api.castorini.uwaterloo.ca`.
- **Input:** `trec_rag_2026_queries.tsv` — TSV file with `narrative_id<TAB>narrative` per line.
- **Output:** Standard TREC run file — 6 space-separated columns per line:
  ```
  topic_id Q0 docid rank score run_id
  ```
- **Evaluation:** Standard IR metrics (nDCG, MRR, Precision/Recall).

> [!IMPORTANT]
> **Deadline: August 8, 2026.** You have ~8 days. This plan is designed for a fast, high-quality baseline submission.

## User Review Required

> [!IMPORTANT]
> **Pyserini API Token Required:** You must email `get-pyserini@googlegroups.com` to request an API token. This is **mandatory** to access the ClimbMix-400b corpus. Do this IMMEDIATELY if you haven't already.

> [!IMPORTANT]
> **TREC Registration Required:** You must register at [trec.nist.gov](https://trec.nist.gov/cfp.html) to be an active participant and submit runs. Do this NOW if you haven't.

> [!WARNING]
> **Hardware:** The retrieval pipeline below uses the Pyserini REST API (no local GPU needed for BM25 baseline). For the reranking step, you'll need a machine with a GPU (8GB+ VRAM) or use a cloud API. If you only have a CPU, the BM25-only baseline is still a valid submission.

## Open Questions

1. **Do you already have a Pyserini API token?** If not, you need to email `get-pyserini@googlegroups.com` immediately.
2. **Are you registered for TREC 2026?** If not, register at https://trec.nist.gov/cfp.html.
3. **What hardware do you have?** (GPU size determines reranking approach)
4. **Do you have a team name / run ID you want to use?** (e.g., `myteam_bm25`, `myteam_rerank`)

---

## Proposed Changes / Implementation Pipeline

### Phase 0: Prerequisites & Setup

#### Environment Setup
- Install Python 3.10+
- Create a project directory and virtual environment
- Install required libraries

```bash
# Create project directory
mkdir trec-rag-2026-retrieval
cd trec-rag-2026-retrieval

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install requests tqdm python-dotenv
```

#### Clone Data Repository
```bash
git clone https://github.com/TREC-RAG/trec-rag-data.git
```
The test queries are at: `trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv`

#### Get Pyserini API Token
Email `get-pyserini@googlegroups.com` requesting access. Store the token in `.env.local`:
```
PYSERINI_API_TOKEN=your_token_here
```

---

### Phase 1: BM25 Baseline Retrieval (via Pyserini REST API)

This is the simplest valid submission — pure BM25 retrieval using the official API.

#### [NEW] `config.py` — Configuration constants
```python
API_BASE_URL = "http://api.castorini.uwaterloo.ca"
INDEX_NAME = "climbmix-400b"
QUERIES_FILE = "trec-rag-data/trec-rag-2026/test-data/trec_rag_2026_queries.tsv"
OUTPUT_DIR = "runs"
RUN_ID = "myteam_bm25"  # Change to your team name
HITS_PER_QUERY = 100  # Retrieve top 100 per narrative
```

#### [NEW] `01_bm25_retrieve.py` — BM25 retrieval via Pyserini REST API
This script:
1. Reads all 119 narratives from the TSV file
2. For each narrative, sends a search request to the Pyserini REST API
3. Collects the results and writes them in TREC run file format

**API Endpoint:** `GET /v1/climbmix-400b/search?query=<text>&hits=100`
- Requires `Authorization: Bearer <token>` header
- Returns JSON with `candidates` array, each having `docid`, `rank`, `score`, `doc`

#### [NEW] `02_format_run.py` — Format and validate the TREC run file
Ensures the output file is correctly formatted:
```
rag2026-0 Q0 shard_00459_61697 1 12.483799 myteam_bm25
rag2026-0 Q0 shard_00123_45678 2 11.234567 myteam_bm25
...
```

---

### Phase 2: Query Processing (Improves over raw BM25)

#### [NEW] `03_query_processing.py` — Extract key search terms from narratives
The narratives are long (2-3 sentences). Directly using them as BM25 queries may hurt precision. This script:
1. Uses simple keyword extraction (TF-IDF or rule-based) to extract the core information need
2. Optionally splits complex narratives into sub-queries
3. Re-runs BM25 with cleaned/focused queries

**Approach options (pick one or combine):**
- **Simple truncation:** Use first sentence only
- **Keyword extraction:** Extract nouns/entities using spaCy
- **LLM query rewriting:** Use an LLM (GPT-4o-mini / Gemini Flash) to rewrite the narrative into a focused search query

---

### Phase 3: Neural Reranking (Significant Quality Boost)

> [!TIP]
> This is the single biggest improvement you can make over BM25. Cross-encoder reranking typically improves nDCG@10 by 20-40%.

#### Install reranking dependencies
```bash
pip install sentence-transformers torch
# OR for a lighter approach:
pip install flashrank
```

#### [NEW] `04_rerank.py` — Cross-encoder reranking
This script:
1. Reads the BM25 results from Phase 1
2. For each query, takes the top-100 BM25 candidates
3. Scores each (query, document) pair using a cross-encoder model
4. Re-sorts by the cross-encoder score
5. Writes a new TREC run file

**Recommended models (pick one):**

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M params | Good | Fast (CPU OK) |
| `BAAI/bge-reranker-v2-m3` | 568M params | Very Good | Medium (GPU recommended) |
| `Alibaba-NLP/gte-reranker-base` | 110M params | Good | Fast |
| `flashrank` (FlashRank library) | Various | Good | Very fast (CPU) |

**For CPU-only:** Use `flashrank` or `cross-encoder/ms-marco-MiniLM-L-12-v2`
**For GPU:** Use `BAAI/bge-reranker-v2-m3`

---

### Phase 4: Hybrid Retrieval + RRF (Advanced — Optional)

If you have time, add dense retrieval and fuse with BM25:

#### [NEW] `05_dense_retrieve.py` — Dense retrieval using embeddings
- Encode narratives with a bi-encoder (e.g., `BAAI/bge-base-en-v1.5`)
- Search the ClimbMix index (if dense search is available via API)
- OR: Use the BM25 top-1000 candidates and re-score with dense similarity

#### [NEW] `06_rrf_fusion.py` — Reciprocal Rank Fusion
- Combine BM25 and dense retrieval ranked lists using RRF
- Formula: `RRF_score = Σ 1/(k + rank_i)` where `k=60` is typical
- Write the fused result as a new TREC run file

---

### Phase 5: Validation & Submission

#### [NEW] `07_validate_run.py` — Validate the run file
Checks:
- All 119 topic IDs are present (rag2026-0 through rag2026-118)
- Each line has exactly 6 space-separated fields
- Column 2 is always "Q0"
- Ranks are sequential per topic
- Scores are in non-increasing order per topic
- Run ID is consistent
- No duplicate (topic_id, docid) pairs

#### Submission Process
1. Go to TREC submission portal (through Evalbase: `ir.nist.gov/evalbase`)
2. Upload your run file
3. Name your run clearly (e.g., `myteam_bm25_reranked`)

---

## Complete File Structure

```
trec-rag-2026-retrieval/
├── .env.local                     # Pyserini API token (DO NOT COMMIT)
├── .gitignore                     # Ignore .env.local, runs/, etc.
├── config.py                      # Configuration constants
├── 01_bm25_retrieve.py            # BM25 retrieval via Pyserini REST API
├── 02_format_run.py               # Format and validate TREC run file
├── 03_query_processing.py         # Query rewriting/cleaning
├── 04_rerank.py                   # Cross-encoder reranking
├── 05_dense_retrieve.py           # Dense retrieval (optional)
├── 06_rrf_fusion.py               # RRF fusion (optional)
├── 07_validate_run.py             # Validate run file before submission
├── requirements.txt               # All Python dependencies
├── runs/                          # Output directory for run files
│   ├── bm25_raw.txt               # Raw BM25 results
│   ├── bm25_reranked.txt          # After reranking
│   └── hybrid_rrf_reranked.txt    # After hybrid + reranking
└── trec-rag-data/                 # Cloned data repo
    └── trec-rag-2026/
        └── test-data/
            └── trec_rag_2026_queries.tsv
```

---

## Recommended Execution Order (Priority)

| Priority | Step | Time Estimate | Expected Impact |
|----------|------|---------------|-----------------|
| 🔴 P0 | Get API token + Register for TREC | 1-2 days (waiting) | **Blocker** |
| 🔴 P0 | Phase 1: BM25 baseline | 2-3 hours | Valid submission |
| 🟡 P1 | Phase 3: Reranking | 3-4 hours | **+20-40% nDCG** |
| 🟡 P1 | Phase 2: Query processing | 2-3 hours | +5-15% improvement |
| 🟢 P2 | Phase 4: Hybrid + RRF | 4-6 hours | +5-10% more |
| 🟢 P2 | Phase 5: Validation | 30 min | Prevent rejection |

---

## Verification Plan

### Automated Tests
1. Run `07_validate_run.py` to verify format compliance
2. Count unique topic_ids — must be exactly 119
3. Verify scores are monotonically non-increasing per topic
4. Check that all docids match ClimbMix format (`shard_XXXXX_YYYYY`)

### Manual Verification
1. Spot-check 3-5 topics: read the narrative, check if top-5 retrieved docs are relevant
2. Compare BM25 vs reranked results for the same queries
3. Verify run file size is reasonable (should be ~12K lines for 119 topics × 100 docs)

### Submission Checklist
- [ ] TREC registration confirmed
- [ ] Pyserini API token obtained
- [ ] Run file passes validation
- [ ] All 119 topics covered
- [ ] Run file uploaded to TREC portal before August 8

---

## Summary of What You'll Submit

A single text file (e.g., `myteam_bm25_reranked.txt`) containing ~11,900 lines (119 topics × 100 docs each), in standard TREC format:

```
rag2026-0 Q0 shard_00459_61697 1 0.9998 myteam_bm25_reranked
rag2026-0 Q0 shard_00123_45678 2 0.9876 myteam_bm25_reranked
...
rag2026-118 Q0 shard_00789_12345 100 0.0123 myteam_bm25_reranked
```

> [!NOTE]
> **Multiple runs allowed:** You can submit up to 5 runs (check current TREC rules). So you can submit both a BM25-only baseline AND a reranked version.

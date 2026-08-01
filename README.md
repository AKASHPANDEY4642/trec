# TREC RAG 2026 — Retrieval Task Submission Pipeline

> **Deadline: August 8, 2026**

A complete, beginner-friendly pipeline to submit a competitive retrieval run for the TREC RAG 2026 Retrieval (R) task.

## What This Does

Given 119 narrative queries, this pipeline retrieves and ranks the most relevant segments from the **NVIDIA ClimbMix-400b** corpus (553M documents) using the official **Pyserini REST API**.

## Quick Start (Minimum Viable Submission)

```bash
# 1. Clone this project (or copy the files)
cd trec-rag-2026-retrieval

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Install minimal dependencies
pip install requests python-dotenv tqdm

# 4. Clone the test data
git clone https://github.com/TREC-RAG/trec-rag-data.git

# 5. Set your API token (see "Getting Your API Token" below)
echo PYSERINI_API_TOKEN=your_token_here > .env.local

# 6. Edit config.py — change RUN_ID to your team name
#    RUN_ID = "yourteam_bm25"

# 7. Run BM25 retrieval
python 01_bm25_retrieve.py

# 8. Validate the run file
python 07_validate_run.py

# 9. Submit to TREC!
```

## Prerequisites

### 1. Register for TREC (DO THIS FIRST!)
Go to https://trec.nist.gov/cfp.html and register your organization.

### 2. Get Pyserini API Token (DO THIS IMMEDIATELY!)
Email `get-pyserini@googlegroups.com` to request access to the Pyserini REST API.
This may take 1-2 days so **do it now**.

Once you receive the token, create `.env.local`:
```
PYSERINI_API_TOKEN=your_token_here
```

### 3. Python Environment
- Python 3.10 or higher
- pip (comes with Python)

## Pipeline Steps

| Step | Script | What It Does | Time | Impact |
|------|--------|-------------|------|--------|
| 1 | `01_bm25_retrieve.py` | BM25 retrieval via API | ~5 min | **Baseline** |
| 2 | `02_query_processing.py` | Clean/focus the queries | <1 min | Prep for step 3 |
| 3 | `03_bm25_retrieve_processed.py` | BM25 with cleaned queries | ~5 min | +5-15% |
| 4 | `04_rerank.py` | Neural reranking | ~30 min | **+20-40%** |
| 5 | `05_rrf_fusion.py` | Combine multiple runs | <1 min | +5-10% |
| 6 | `06_spot_check.py` | Manual quality check | 2 min | Quality assurance |
| 7 | `07_validate_run.py` | Validate run file format | <1 min | **Required** |

### Recommended Step Combinations

**Fastest (just submit something):**
```bash
python run_pipeline.py --steps 1,7
```

**Good quality (recommended):**
```bash
pip install sentence-transformers torch  # or: pip install flashrank
python run_pipeline.py --steps 1,4,7 --flashrank  # CPU
python run_pipeline.py --steps 1,4,7 --gpu         # GPU
```

**Best quality:**
```bash
python run_pipeline.py --steps 1,2,3,4,5,7 --gpu
```

## Run All Steps Manually

### Step 1: BM25 Retrieval
```bash
python 01_bm25_retrieve.py
```
Sends all 119 narratives to the Pyserini REST API and retrieves the top 100 BM25 results for each.

**Output:** `runs/bm25_raw.txt`

### Step 2: Query Processing
```bash
python 02_query_processing.py
```
Creates focused search queries from the long narratives. Narratives are 2-3 sentences; this extracts the core information need.

**Output:** `tmp/processed_queries.json`

### Step 3: BM25 with Processed Queries
```bash
python 03_bm25_retrieve_processed.py
```
Re-runs BM25 with the cleaned queries for potentially better results.

**Output:** `runs/query_rewritten_bm25.txt`

### Step 4: Cross-Encoder Reranking ⭐
```bash
# CPU (slower but works everywhere):
python 04_rerank.py --flashrank

# GPU (faster, better quality):
python 04_rerank.py --gpu

# Rerank the processed-query results instead:
python 04_rerank.py --input bm25_combined --gpu
```
This is the **single biggest improvement**. Cross-encoders see both query and document together and produce much better relevance scores than BM25.

**Output:** `runs/bm25_reranked.txt`

### Step 5: RRF Fusion
```bash
python 05_rrf_fusion.py
```
Combines multiple ranked lists using Reciprocal Rank Fusion. Needs at least 2 run files.

**Output:** `runs/rrf_fused.txt`

### Step 6: Spot Check
```bash
python 06_spot_check.py --run runs/bm25_reranked.txt --topics 5
```
Shows the narrative and top retrieved documents so you can manually check relevance.

### Step 7: Validate
```bash
python 07_validate_run.py runs/bm25_reranked.txt
```
Checks format compliance. **Must pass before submission.**

## Submission

1. Go to **https://ir.nist.gov/evalbase** (TREC submission portal)
2. Log in with your TREC credentials
3. Find the TREC RAG 2026 track
4. Upload your run file (e.g., `runs/bm25_reranked.txt`)
5. Mark as task: **Retrieval (R)**
6. Give it a descriptive name matching your `RUN_ID`

> **Tip:** You can submit up to 5 runs. Submit both a BM25 baseline and a reranked version!

## File Structure

```
trec-rag-2026-retrieval/
├── .env.local                        # API token (DO NOT COMMIT)
├── .gitignore
├── config.py                         # All settings — EDIT RUN_ID
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # Run all steps at once
├── 01_bm25_retrieve.py               # BM25 via Pyserini API
├── 02_query_processing.py            # Clean/focus queries
├── 03_bm25_retrieve_processed.py     # BM25 with clean queries
├── 04_rerank.py                      # Neural reranking
├── 05_rrf_fusion.py                  # RRF fusion
├── 06_spot_check.py                  # Manual quality check
├── 07_validate_run.py                # Validate before submit
├── runs/                             # Output run files
│   ├── bm25_raw.txt
│   ├── bm25_reranked.txt
│   └── ...
├── tmp/                              # Cached intermediate results
└── trec-rag-data/                    # Cloned official data repo
```

## Troubleshooting

**"No API token"**: Email `get-pyserini@googlegroups.com` and set `PYSERINI_API_TOKEN` in `.env.local`

**"Queries file not found"**: Run `git clone https://github.com/TREC-RAG/trec-rag-data.git` in the project directory

**"Connection error"**: The API at `http://api.castorini.uwaterloo.ca` may be down temporarily. Wait and retry.

**"Rate limited (429)"**: The script handles this automatically with exponential backoff. You can increase `REQUEST_DELAY_SECONDS` in `config.py`.

**"CUDA out of memory"**: Use `--flashrank` flag instead of `--gpu` for CPU-based reranking, or use a smaller model in `config.py`.

## Key Links

- [TREC RAG Website](https://trec-rag.github.io/)
- [Track Guidelines](https://github.com/TREC-RAG/trec-rag-skills/tree/main/skills/trec-rag-2026-track-guidelines)
- [Test Topics](https://github.com/TREC-RAG/trec-rag-data/tree/main/trec-rag-2026/test-data)
- [Pyserini REST API Skill](https://github.com/TREC-RAG/trec-rag-skills/tree/main/skills/pyserini-rest-api)
- [TREC Registration](https://trec.nist.gov/cfp.html)
- [RAGDoll Evaluation](https://github.com/castorini/RAGDoll)
- [Mailing List](https://groups.google.com/g/trec-rag-2026-participants)

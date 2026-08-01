# TREC RAG 2026 — Lab GPU Instructions (RTX 2000 Ada, 16GB VRAM)

> **Team: IIITDMK**

## What This Is

This is the same retrieval pipeline from your hostel laptop, but configured to run neural reranking on your lab GPU for a **much stronger submission** (+20-40% nDCG improvement).

## Your Lab GPU Specs
- **GPU**: RTX 2000 Ada Generation  
- **VRAM**: 16 GB
- **Driver**: 595.84
- **CUDA Support**: 13.2
- **OS**: Ubuntu

This is more than enough for cross-encoder reranking (needs ~4GB VRAM).

---

## Quick Setup (One Command)

```bash
chmod +x lab_gpu_setup/setup.sh
./lab_gpu_setup/setup.sh
```

This installs everything: Python, PyTorch with CUDA, sentence-transformers, etc.

---

## Manual Setup (If Script Fails)

```bash
# 1. System packages
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Core dependencies
pip install requests python-dotenv tqdm pandas

# 4. PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Reranking models
pip install sentence-transformers>=3.0.0
pip install flashrank>=0.2.0

# 6. Verify GPU
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## Files to Copy from Hostel Laptop

**CRITICAL**: Copy these from your hostel laptop to the lab machine:

| File | Why |
|------|-----|
| `.env.local` | Contains your API token |
| `tmp/bm25_results.json` | Cached BM25 results with document text (needed for reranking without re-running API calls) |
| `runs/bm25_raw.txt` | Your BM25 baseline run file |
| All `.py` files | The pipeline scripts |
| `config.py` | Configuration (already set to IIITDMK) |
| `trec-rag-data/` | Test queries (or clone fresh with `git clone`) |

### Easiest Way to Transfer
```bash
# On hostel laptop: zip the project
# Then copy via USB, SCP, or cloud

# Or on lab machine, just copy everything:
scp -r user@hostel-ip:path/to/trec-rag-2026-retrieval/ ./
```

---

## Running the GPU Pipeline

### Option A: Quick Rerank (Recommended First)
Just rerank the BM25 results — biggest bang for the buck.

```bash
source venv/bin/activate

# Rerank with GPU (using cross-encoder/ms-marco-MiniLM-L-12-v2, ~5 min)
python 04_rerank.py --gpu

# Validate
python 07_validate_run.py runs/bm25_reranked.txt
```

### Option B: Full Pipeline (Best Quality)
```bash
source venv/bin/activate

# Query processing (CPU, instant)
python 02_query_processing.py

# BM25 with cleaned queries (API, ~5 min)
python 03_bm25_retrieve_processed.py

# Rerank with GPU
python 04_rerank.py --gpu

# Fuse multiple runs
python 05_rrf_fusion.py

# Spot check
python 06_spot_check.py --run runs/rrf_fused.txt --topics 5

# Validate
python 07_validate_run.py runs/rrf_fused.txt
```

### Option C: Higher Quality Model (Uses More VRAM)
Edit `config.py` to use the bigger model:

```python
# Change this line in config.py:
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # 568M params, needs ~4GB VRAM
```

Then run:
```bash
python 04_rerank.py --gpu
```

With 16GB VRAM, your RTX 2000 Ada can handle this easily.

---

## Expected Outputs

| Run File | Method | Expected Quality |
|----------|--------|-----------------|
| `runs/bm25_raw.txt` | BM25 baseline | Baseline |
| `runs/bm25_reranked.txt` | BM25 + Cross-Encoder | **+20-40% nDCG** |
| `runs/query_rewritten_bm25.txt` | BM25 + Query Processing | +5-15% |
| `runs/rrf_fused.txt` | All combined via RRF | **Best** |

---

## Submission

1. Go to **https://ir.nist.gov/evalbase**
2. Log in with TREC credentials
3. Upload your run file (best one: `runs/rrf_fused.txt` or `runs/bm25_reranked.txt`)
4. Task: **Retrieval (R)**
5. Run ID should match what's in the file (e.g., `IIITDMK_reranked`)

> You can submit up to **5 runs**. Submit both the BM25 baseline and the reranked version!

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA not available` | Check `nvidia-smi`, install CUDA toolkit: `sudo apt install nvidia-cuda-toolkit` |
| `CUDA out of memory` | Use `--flashrank` flag instead of `--gpu`, or use smaller model |
| `No module 'torch'` | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `API token error` | Check `.env.local` has `PYSERINI_API_TOKEN=your_token` |
| `bm25_results.json missing` | Re-run `python 01_bm25_retrieve.py` on the lab machine |

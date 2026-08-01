#!/bin/bash
# ============================================================
# TREC RAG 2026 — Lab GPU Setup Script (Ubuntu + RTX 2000 Ada)
# Team: IIITDMK
# ============================================================
# Run this ONCE on your lab Ubuntu machine to set up everything.
# 
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e

echo "============================================================"
echo "TREC RAG 2026 — Lab GPU Setup for Team IIITDMK"
echo "============================================================"
echo ""

# --- 1. System dependencies ---
echo "[1/6] Installing system dependencies..."
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git curl

# --- 2. Create virtual environment ---
echo "[2/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# --- 3. Install Python dependencies ---
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip

# Core dependencies
pip install requests python-dotenv tqdm pandas

# PyTorch with CUDA support (for RTX 2000 Ada with CUDA 13.2 / driver 595.84)
# Using CUDA 12.1 wheels which are compatible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Neural reranking
pip install sentence-transformers>=3.0.0

# Lightweight reranking alternative
pip install flashrank>=0.2.0

# --- 4. Verify CUDA ---
echo "[4/6] Verifying CUDA setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f'GPU Memory: {total_mem:.1f} GB')
else:
    print('WARNING: CUDA not available! GPU reranking will not work.')
    print('Check your NVIDIA driver and CUDA installation.')
"

# --- 5. Clone data repo (if not already present) ---
echo "[5/6] Setting up data..."
if [ ! -d "trec-rag-data" ]; then
    git clone https://github.com/TREC-RAG/trec-rag-data.git
else
    echo "  trec-rag-data/ already exists, skipping clone"
fi

# --- 6. Check for required files ---
echo "[6/6] Checking required files..."

if [ ! -f ".env.local" ]; then
    echo "  WARNING: .env.local not found!"
    echo "  Create it with: echo 'PYSERINI_API_TOKEN=your_token' > .env.local"
else
    echo "  .env.local found"
fi

if [ ! -f "tmp/bm25_results.json" ]; then
    echo "  WARNING: tmp/bm25_results.json not found!"
    echo "  Copy this file from your hostel laptop (it has cached BM25 results with doc text)"
    echo "  Without it, you'll need to re-run 01_bm25_retrieve.py first"
else
    echo "  tmp/bm25_results.json found ($(du -sh tmp/bm25_results.json | cut -f1))"
fi

echo ""
echo "============================================================"
echo "Setup complete! Next steps:"
echo "============================================================"
echo ""
echo "1. Make sure .env.local has your API token"
echo "2. Copy tmp/bm25_results.json from your hostel laptop"
echo "3. Run the GPU pipeline:"
echo ""
echo "   source venv/bin/activate"
echo ""
echo "   # Option A: Just rerank the existing BM25 results (fastest)"
echo "   python 04_rerank.py --gpu"
echo "   python 07_validate_run.py runs/bm25_reranked.txt"
echo ""
echo "   # Option B: Full pipeline (best quality)"
echo "   python 02_query_processing.py"
echo "   python 03_bm25_retrieve_processed.py"
echo "   python 04_rerank.py --gpu"
echo "   python 05_rrf_fusion.py"
echo "   python 07_validate_run.py runs/rrf_fused.txt"
echo ""
echo "   # Spot check results"
echo "   python 06_spot_check.py --run runs/bm25_reranked.txt --topics 5"
echo ""

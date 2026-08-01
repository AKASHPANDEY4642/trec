# LAB AGENT PLAN (For the Lab GPU AI Agent)

Hello AI Agent! When the user is at the lab and asks you to run the retrieval and reranking pipeline, follow these exact steps:

## 1. Environment Setup
The user will clone this repository on the lab machine (Ubuntu with RTX 2000 Ada, 16GB VRAM).
- Check if `.env.local` exists. If not, ask the user to provide their `PYSERINI_API_TOKEN` and create the `.env.local` file with the contents:
  `PYSERINI_API_TOKEN=<their_token>`
- The `trec-rag-data/` folder is ignored by git. If it's missing, run the download scripts or clone the dataset. (It may automatically download if the scripts handle it).
- Run the setup script to install dependencies:
  ```bash
  chmod +x lab_gpu_setup/setup.sh
  ./lab_gpu_setup/setup.sh
  ```

## 2. Execute the Pipeline
You are running on a machine with a powerful GPU (16GB VRAM). Use the `venv` created by the setup script.
Activate it: `source venv/bin/activate`

Run the pipeline in this order:
1. **Initial BM25 Retrieval**: `python 01_bm25_retrieve.py`
   *(This requires `.env.local` with API token and will cache results in `tmp/`)*
2. **Query Processing**: `python 02_query_processing.py`
3. **Processed Retrieval**: `python 03_bm25_retrieve_processed.py`
4. **GPU Reranking (Crucial step!)**: `python 04_rerank.py --gpu`
   *(Since this machine has a good GPU, we can use the powerful cross-encoders)*
5. **RRF Fusion**: `python 05_rrf_fusion.py`
6. **Validate Run**: `python 07_validate_run.py runs/rrf_fused.txt`

## 3. Results Collection
After running, the best run will be located at `runs/rrf_fused.txt` (and the reranked run at `runs/bm25_reranked.txt`). Ask the user to submit these to the TREC evalbase!

## Note on Ignored Files
The following files are ignored via `.gitignore` and must be regenerated or copied manually on the lab machine:
- `.env.local` (API credentials)
- `tmp/` (Cached results from BM25)
- `runs/` (Final run outputs)
- `trec-rag-data/` (Large data files)
- `venv/` / `__pycache__/` (Python environments and caches)

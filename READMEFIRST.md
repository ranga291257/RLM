# READMEFIRST

## What this repo is

This repo is a **side-by-side demo** of two ways to analyze a long text document (a synthetic plant maintenance log):

- **RLM-style** (`rlm_ollama_demo.py`): query-driven scanning + recursive summarization with evidence chunk IDs.
- **RAG-style** (`rag_demo.py`): pre-chunk + embed + retrieve top‑K chunks + answer from retrieved context only.

The default demo document is `synthetic_maintenance_log.txt`.

---

## Prereqs (Windows)

- **Python** (your repo comments mention Python 3.14 support; 3.12+ is typically fine unless you require 3.14 specifically)
- **Ollama** installed and running locally
  - Start the server: `ollama serve` (or just open the Ollama app on Windows)

---

## Quick start (PowerShell)

From the repo root (`d:\dev\RLM`):

1) **Create venv + install dependencies**

```powershell
.\scripts\setup-rlm-env.ps1
.\rlm_env\Scripts\Activate.ps1
```

1) **Pull local Ollama models (recommended defaults)**

```powershell
ollama pull llama3.2:3b
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

1) **Run the local-only RLM demo (best “first run”)**

```powershell
python .\rlm_ollama_local_demo.py
```

1) **Run the RAG demo**

```powershell
python .\rag_demo.py --no-cloud
```

1) **Compare outputs (after both runs)**

```powershell
python .\compare_rlm_vs_rag.py --compare-only
```

Outputs you should see created/updated in the repo root:

- `rlm_analysis_output.md`
- `rag_analysis_output.md`
- `rlm_vs_rag_comparison_report.md`

---

## Key scripts (what to run when)

- **`rlm_ollama_local_demo.py`**
  - Local-only, default model: `llama3.2:3b`
  - Uses `synthetic_maintenance_log.txt` by default
  - No cloud flags; always calls Ollama locally

- **`rlm_ollama_demo.py`**
  - Full demo that supports **local** and **Ollama Cloud**
  - Default model in this file may be a cloud model; use it when you want cloud behavior

- **`rag_demo.py`**
  - RAG pipeline using Ollama embeddings (`/api/embeddings`) + chat (`/api/chat`)
  - Defaults to repo-local paths:
    - input: `synthetic_maintenance_log.txt`
    - output: `rag_analysis_output.md`

- **`compare_rlm_vs_rag.py`**
  - Reads `rlm_analysis_output.md` and `rag_analysis_output.md`
  - Writes `rlm_vs_rag_comparison_report.md`

- **`synthetic_data.py`**
  - Generates a synthetic maintenance log for repeatable demos

Example:

```powershell
python .\synthetic_data.py --out .\synthetic_maintenance_log.txt --entries 250 --seed 42
```

---

## Optional: GPU / CUDA sanity check (PyTorch)

If you want CUDA-enabled PyTorch installed, `.\scripts\setup-rlm-env.ps1` installs PyTorch from:
`https://download.pytorch.org/whl/cu130` (you can override via `-TorchIndexUrl`).

To verify CUDA visibility:

```powershell
python .\check_cuda.py
```

---

## Optional: Ollama Cloud test

`test_ollama_cloud_model.py` is a **cloud-only** smoke test.

- Requires env var `OLLAMA_API_KEY`
- Uses the `ollama` Python package client

```powershell
$env:OLLAMA_API_KEY="..."
python .\test_ollama_cloud_model.py
```

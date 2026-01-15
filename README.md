RLM vs RAG Demo

## What this repo is

This repo is a **side-by-side demo** of two ways to analyze a long text document (a synthetic plant maintenance log):

- **RLM-style** (`rlm_ollama_demo.py`): query-driven scanning + recursive summarization with evidence chunk IDs.
- **RAG-style** (`rag_demo.py`): pre-chunk + embed + retrieve top‑K chunks + answer from retrieved context only.

The default demo document is `synthetic_maintenance_log.txt`.

---

## Prerequisites (Windows)

- **Python** 3.12+ (3.14 recommended for CUDA 13.0 support)
- **Ollama** installed and running locally
  - Start the server: `ollama serve` (or just open the Ollama app on Windows)
- For **Ollama Cloud**: Set `OLLAMA_API_KEY` environment variable
- **PDF support**: `PyPDF2` is included in `requirements.txt` (installed automatically)

---

## Quick Setup

From the repo root (`d:\dev\RLM`):

### 1. Create virtual environment and install dependencies

```powershell
.\scripts\setup-rlm-env.ps1
.\rlm_env\Scripts\Activate.ps1
```

### 2. Pull local Ollama models (recommended defaults)

```powershell
ollama pull llama3.2:3b
ollama pull llama3.2:latest
ollama pull nomic-embed-text
```

---

## Running the Demos

### RLM Demo (`rlm_ollama_demo.py`)

Supports both **local** and **cloud** modes.

#### Local Mode Examples

**Basic local run (default model):**

```powershell
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b
```

**Local with custom model:**

```powershell
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:latest
```

**Local with custom question and timeout:**

```powershell
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --question "What are the main vibration issues?" --timeout 600
```

**Local with custom document (supports .txt and .pdf):**

```powershell
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file ".\my_log.txt"
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file ".\maintenance_report.pdf"
```

**Local with performance tuning (faster, fewer spans):**

```powershell
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --max-spans 5 --window 400
```

#### Cloud Mode Examples

**Cloud mode (auto-detected from model name ending in `-cloud`):**

Set API key as environment variable (recommended - keeps key out of command history):

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rlm_ollama_demo.py --model gpt-oss:120b-cloud
```

**Cloud mode (explicit flag with env var):**

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rlm_ollama_demo.py --cloud --model gpt-oss:120b-cloud
```

**Cloud mode with API key in command line (alternative):**

If you prefer to pass the key directly (less secure - appears in command history):

```powershell
python .\rlm_ollama_demo.py --cloud --model gpt-oss:120b-cloud --api-key "your_api_key_here"
```

**Note**: The script will use `OLLAMA_API_KEY` environment variable if `--api-key` is not provided. Setting it as an env var is recommended for security.

**Cloud mode with custom parameters:**

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rlm_ollama_demo.py --cloud --model gpt-oss:120b-cloud --num-predict 4000 --max-spans 10
```

#### RLM Command-Line Options

| Option                     | Default                           | Description                                               |
| -------------------------- | --------------------------------- | --------------------------------------------------------- |
| `--model`                | `gpt-oss:120b-cloud`            | Ollama model name (cloud models end with `-cloud`)      |
| `--cloud`                | Auto-detect                       | Force use of Ollama Cloud                                 |
| `--no-cloud`             | Auto-detect                       | Force use of local Ollama server                          |
| `--api-key`              | From env var                      | Ollama Cloud API key (or set `OLLAMA_API_KEY` env var)  |
| `--question`             | Default question                  | Question to answer                                        |
| `--timeout`              | `900`                           | Per-call Ollama read timeout in seconds                   |
| `--num-predict`          | `3000`                          | Max tokens to generate per call                           |
| `--max-spans`            | `8`                             | Max retrieved spans to summarize (lower = faster)         |
| `--window`               | `600`                           | Chars around each keyword hit to extract (lower = faster) |
| `--fallback-chunk-chars` | `2200`                          | Chunk size if keyword search returns too little           |
| `--fallback-overlap`     | `200`                           | Chunk overlap if keyword search returns too little        |
| `--doc-file`             | None                              | Path to a text or PDF file to analyze (.txt or .pdf)     |
| `--synthetic-file`       | `synthetic_maintenance_log.txt` | Path to synthetic .txt file                               |
| `--generate-synthetic`   | False                             | Generate/overwrite the synthetic file before running      |
| `--entries`              | `250`                           | Synthetic entries to generate (only if generating)        |
| `--seed`                 | `42`                            | Synthetic RNG seed (only if generating)                   |
| `--asset`                | `C-201`                         | Asset tag for synthetic doc (only if generating)          |

---

### RAG Demo (`rag_demo.py`)

Supports both **local** and **cloud** modes.

#### Local Mode Examples

**Basic local run:**

```powershell
python .\rag_demo.py --no-cloud
```

**Local with custom model:**

```powershell
python .\rag_demo.py --no-cloud --model llama3.2:3b
```

**Local with custom embedding model:**

```powershell
python .\rag_demo.py --no-cloud --model llama3.2:3b --embedding-model nomic-embed-text
```

**Local with custom chunking parameters:**

```powershell
python .\rag_demo.py --no-cloud --chunk-size 300 --chunk-overlap 30 --top-k 10
```

**Local with custom output file:**

```powershell
python .\rag_demo.py --no-cloud --output-file ".\my_rag_output.md"
```

#### Cloud Mode Examples

**Cloud mode (auto-detected from model name):**

Set API key as environment variable (recommended):

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rag_demo.py --model llama3.2:latest-cloud
```

**Cloud mode (explicit flag):**

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rag_demo.py --cloud --model llama3.2:latest-cloud
```

**Alternative: Pass API key via command line:**

```powershell
python .\rag_demo.py --cloud --model llama3.2:latest-cloud --api-key "your_api_key_here"
```

#### RAG Command-Line Options

| Option                | Default                           | Description                                              |
| --------------------- | --------------------------------- | -------------------------------------------------------- |
| `--model`           | `llama3.2:latest`               | Ollama model name (local model)                          |
| `--cloud`           | Auto-detect                       | Force use of Ollama Cloud                                |
| `--no-cloud`        | Auto-detect                       | Force use of local Ollama server                         |
| `--api-key`         | From env var                      | Ollama Cloud API key (or set `OLLAMA_API_KEY` env var) |
| `--question`        | Default question                  | Question to answer                                       |
| `--timeout`         | `900`                           | Per-call Ollama read timeout in seconds                  |
| `--num-predict`     | `3000`                          | Max tokens to generate                                   |
| `--top-k`           | `15`                            | Number of chunks to retrieve (RAG parameter)             |
| `--chunk-size`      | `500`                           | Chunk size in words (RAG parameter)                      |
| `--chunk-overlap`   | `50`                            | Chunk overlap in words (RAG parameter)                   |
| `--embedding-model` | `nomic-embed-text`              | Ollama embedding model (local)                           |
| `--doc-file`        | None                              | Path to a text or PDF file to analyze (.txt or .pdf)    |
| `--synthetic-file`  | `synthetic_maintenance_log.txt` | Path to synthetic .txt file                              |
| `--output-file`     | `rag_analysis_output.md`        | Output markdown file                                     |

---

### Compare Outputs

After running both demos, compare their outputs:

```powershell
python .\compare_rlm_vs_rag.py --compare-only
```

This creates `rlm_vs_rag_comparison_report.md` comparing:

- `rlm_analysis_output.md` (from RLM demo)
- `rag_analysis_output.md` (from RAG demo)

---

## Key Scripts Overview

### `rlm_ollama_demo.py`

- **Purpose**: RLM-style analysis with query-driven scanning
- **Modes**: Local and Cloud (auto-detects from model name or use `--cloud`/`--no-cloud`)
- **Default model**: `gpt-oss:120b-cloud` (cloud model)
- **Best for**: Finding all recurring patterns across entire document

### `rag_demo.py`

- **Purpose**: RAG-style analysis with pre-chunking and semantic retrieval
- **Modes**: Local and Cloud (auto-detects from model name or use `--cloud`/`--no-cloud`)
- **Default model**: `llama3.2:latest` (local model)
- **Best for**: Fast retrieval of top-K most relevant chunks

### `compare_rlm_vs_rag.py`

- **Purpose**: Compare outputs from both approaches
- **Usage**: Run after both demos complete
- **Output**: `rlm_vs_rag_comparison_report.md`

### `synthetic_data.py`

- **Purpose**: Generate synthetic maintenance log for repeatable demos

**Example:**

```powershell
python .\synthetic_data.py --out .\synthetic_maintenance_log.txt --entries 250 --seed 42
```

---

## Output Files

After running the demos, you'll find these files in the repo root:

- `rlm_analysis_output.md` - RLM analysis results
- `rag_analysis_output.md` - RAG analysis results
- `rlm_vs_rag_comparison_report.md` - Side-by-side comparison

---

## Optional: GPU / CUDA Setup

If you want CUDA-enabled PyTorch installed, `.\scripts\setup-rlm-env.ps1` installs PyTorch from:
`https://download.pytorch.org/whl/cu130` (you can override via `-TorchIndexUrl`).

To verify CUDA visibility:

```powershell
python .\check_cuda.py
```

**Note**: CUDA 13.0 requires NVIDIA driver version 600+. Check with `nvidia-smi`.

---

## Optional: Ollama Cloud Test

`test_ollama_cloud_model.py` is a **cloud-only** smoke test to verify your API key works.

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\test_ollama_cloud_model.py
```

---

## Supported File Formats

Both RLM and RAG demos support:

- **`.txt` files**: Plain text files (default)
- **`.pdf` files**: PDF documents with automatic detection:
  - **Text-based PDFs**: Direct text extraction (fast)
  - **Scanned PDFs**: Automatic OCR using Tesseract (slower, requires setup)

**Example with PDF:**

```powershell
# RLM with PDF (auto-detects text-based vs scanned)
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file ".\report.pdf"

# RAG with PDF
python .\rag_demo.py --no-cloud --doc-file ".\report.pdf"
```

### PDF Processing Details

**Text-based PDFs:**
- Automatically detected and processed directly (fast)
- Uses PyPDF2 for extraction

**Scanned PDFs (image-based):**
- Automatically detected when little/no text is extractable
- Falls back to OCR using Tesseract (if installed)
- Shows progress: `[INFO] OCR page 1/10...`

**OCR Setup (Optional - for scanned PDFs only):**

1. **Install Tesseract OCR** (Windows):
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location (or set `TESSDATA_PREFIX` env var)

2. **Install Python OCR packages:**
   ```powershell
   pip install pytesseract pdf2image pillow
   ```

3. **Verify OCR works:**
   ```powershell
   python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
   ```

**Note**: OCR is **optional**. If not installed, the script will:
- Still work with text-based PDFs
- Warn and return minimal text for scanned PDFs
- Provide installation instructions

**No OpenCV required** - OCR uses Tesseract directly (avoids numpy version conflicts).

---

## Understanding the Approaches

### RLM (Recursive Language Model)

- **Chunking**: Dynamic, query-dependent
- **Access**: Model writes code to search/chunk based on query
- **Retrieval**: Keyword/regex matching
- **Process**: Recursive summarization and combination
- **Best for**: Dense tasks (finding all recurring patterns)

### RAG (Retrieval-Augmented Generation)

- **Chunking**: Pre-chunked, fixed-size (500 words default)
- **Access**: Static indexing with vector embeddings
- **Retrieval**: Semantic similarity (top-K chunks)
- **Process**: Retrieve → Generate answer
- **Best for**: Sparse tasks (finding specific facts quickly)

---

## Troubleshooting

### "Could not connect to Ollama"

- Make sure Ollama server is running: `ollama serve`
- Or open the Ollama desktop app on Windows

### "OLLAMA_API_KEY environment variable required"

- Set it: `$env:OLLAMA_API_KEY="your_key"` (PowerShell)
- Or pass via `--api-key` flag

### Model not found

- Pull the model first: `ollama pull llama3.2:3b`
- For cloud models, ensure you have API access

### Timeout errors

- Increase `--timeout` (default: 900 seconds)
- Reduce `--max-spans` or `--top-k` for faster processing
- Use a smaller model

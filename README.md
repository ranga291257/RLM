RLM vs RAG Demo

## What this repo is

This repo is a **side-by-side demo** of different ways to analyze a long text document (a synthetic plant maintenance log):

- **RLM-style** (`rlm_ollama_demo.py`): Basic RLM with query-driven scanning + recursive summarization with evidence chunk IDs.
- **RLM Advanced** (`rlm_ollama_advanced.py`): Full paper implementation with REPL, iterative refinement, and multi-level recursion.
- **RAG-style** (`rag_demo.py`): Pre-chunk + embed + retrieve top‑K chunks + answer from retrieved context only.

The default demo document is `input/synthetic_maintenance_log.txt`.

---

## Prerequisites (Windows)

- **Python** 3.13 (required for CUDA 13.0 support)
  - Python 3.13 should be installed at `C:\Users\ranga\Python313\python.exe` (or update path in `setup_cuda130_uv.ps1`)
  - Reason: UV is not ready for Python 3.14 at the time of writing the program
- **UV package manager** (installed automatically by setup script if missing)
- **Ollama** installed and running locally
  - Start the server: `ollama serve` (or just open the Ollama app on Windows)
- For **Ollama Cloud**: Set `OLLAMA_API_KEY` environment variable. This key is free but has rate limitations.
- **PDF support**: `PyPDF2` is included in `requirements.txt` (installed automatically)

---

## Directory Structure

The project uses organized directories:

- **`input/`** - Place your input documents here (.txt, .pdf files)
  - **`proj_prompt.txt`** - Customize analysis questions and prompts (see [Customizing Prompts](#customizing-prompts) below)
- **`output/`** - Generated analysis files (.md) are written here

Both directories are created automatically when you run the scripts.

---

## Quick Setup

From the repo root:

### 1. Create virtual environment and install dependencies

```powershell
.\setup_cuda130_uv.ps1
.\venv_uv\Scripts\Activate.ps1
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
# Uses default model llama3.2:latest
python .\rlm_ollama_demo.py --no-cloud

# Or specify a different model
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
# Relative path (looks in input/ directory)
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file "my_log.txt"
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file "maintenance_report.pdf"

# Or absolute path
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file ".\input\my_log.txt"
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

| Option                     | Default                                 | Description                                               |
| -------------------------- | --------------------------------------- | --------------------------------------------------------- |
| `--model`                | `llama3.2:latest`                     | Ollama model name (cloud models end with `-cloud`)      |
| `--root-model`           | `llama3.2:latest`                     | Root model (planner/controller). Defaults to --model.     |
| `--sub-model`            | `qwen3:0.6b`                         | Sub-model used for recursive chunk summarization.         |
| `--cloud`                | Auto-detect                             | Force use of Ollama Cloud                                 |
| `--no-cloud`             | Auto-detect                             | Force use of local Ollama server                          |
| `--api-key`              | From env var                            | Ollama Cloud API key (or set `OLLAMA_API_KEY` env var)  |
| `--question`             | Default question                        | Question to answer                                        |
| `--timeout`              | `900`                                 | Per-call Ollama read timeout in seconds                   |
| `--num-predict`          | `3000`                                | Max tokens to generate per call                           |
| `--max-spans`            | `8`                                   | Max retrieved spans to summarize (lower = faster)         |
| `--window`               | `600`                                 | Chars around each keyword hit to extract (lower = faster) |
| `--fallback-chunk-chars` | `2200`                                | Chunk size if keyword search returns too little           |
| `--fallback-overlap`     | `200`                                 | Chunk overlap if keyword search returns too little        |
| `--doc-file`             | None                                    | Path to a text or PDF file to analyze (.txt or .pdf)      |
| `--synthetic-file`       | `input/synthetic_maintenance_log.txt` | Path to synthetic .txt file                               |
| `--generate-synthetic`   | False                                   | Generate/overwrite the synthetic file before running      |
| `--entries`              | `250`                                 | Synthetic entries to generate (only if generating)        |
| `--seed`                 | `42`                                  | Synthetic RNG seed (only if generating)                   |
| `--asset`                | `C-201`                               | Asset tag for synthetic doc (only if generating)          |
| `--output-file`          | `output/rlm_analysis_output.md`       | Write output to markdown file (default: writes to file)   |

---

### RLM Advanced Demo (`rlm_ollama_advanced.py`)

Enhanced RLM implementation with full paper features including REPL environment, iterative refinement loop, and multi-level recursive sub-calls.

Supports both **local** and **cloud** modes.

#### Local Mode Examples

**Basic local run:**

```powershell
python .\rlm_ollama_advanced.py --no-cloud
```

**Local with custom recursion depth:**

```powershell
python .\rlm_ollama_advanced.py --no-cloud --max-recursion-depth 5
```

**Local with iterative refinement:**

```powershell
python .\rlm_ollama_advanced.py --no-cloud --max-iterations 10 --min-spans 6
```

**Local with custom document:**

```powershell
python .\rlm_ollama_advanced.py --no-cloud --doc-file "my_log.txt"
```

#### Cloud Mode Examples

**Cloud mode (auto-detected from model name):**

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rlm_ollama_advanced.py --model gpt-oss:120b-cloud
```

**Cloud mode (explicit flag):**

```powershell
$env:OLLAMA_API_KEY="your_api_key_here"
python .\rlm_ollama_advanced.py --cloud --model gpt-oss:120b-cloud
```

#### RLM Advanced Command-Line Options

| Option                  | Default                                  | Description                                               |
| ----------------------- | ---------------------------------------- | --------------------------------------------------------- |
| `--model`               | `llama3.2:latest`                        | Ollama model name (cloud models end with `-cloud`)      |
| `--root-model`          | `llama3.2:latest`                        | Root model (planner/controller). Defaults to --model.    |
| `--sub-model`           | `qwen3:0.6b`                         | Sub-model used for recursive calls.                      |
| `--cloud`               | Auto-detect                               | Force use of Ollama Cloud                                 |
| `--no-cloud`            | Auto-detect                               | Force use of local Ollama server                          |
| `--api-key`             | From env var                              | Ollama Cloud API key (or set `OLLAMA_API_KEY` env var)  |
| `--question`            | From prompt file                          | Question to answer                                        |
| `--timeout`             | `900`                                   | Per-call Ollama read timeout in seconds                   |
| `--num-predict`         | `3000`                                  | Max tokens to generate per call                           |
| `--max-iterations`      | `5`                                     | Max iterations for search refinement                      |
| `--min-spans`           | `4`                                     | Minimum spans to find before stopping iteration           |
| `--max-recursion-depth` | `3`                                     | Maximum recursion depth for chunk summarization           |
| `--doc-file`            | None                                     | Path to a text or PDF file to analyze (.txt or .pdf)    |
| `--synthetic-file`      | `input/synthetic_maintenance_log.txt`    | Path to synthetic .txt file                               |
| `--generate-synthetic`  | False                                    | Generate/overwrite the synthetic file before running      |
| `--entries`             | `250`                                   | Synthetic entries to generate (only if generating)        |
| `--seed`                | `42`                                   | Synthetic RNG seed (only if generating)                   |
| `--asset`               | `C-201`                                 | Asset tag for synthetic doc (only if generating)          |
| `--output-file`         | `output/rlm_advanced_analysis_output.md` | Write output to markdown file                             |

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

| Option                | Default                                 | Description                                              |
| --------------------- | --------------------------------------- | -------------------------------------------------------- |
| `--model`           | `llama3.2:latest`                     | Ollama model name (local model)                          |
| `--cloud`           | Auto-detect                             | Force use of Ollama Cloud                                |
| `--no-cloud`        | Auto-detect                             | Force use of local Ollama server                         |
| `--api-key`         | From env var                            | Ollama Cloud API key (or set `OLLAMA_API_KEY` env var) |
| `--question`        | Default question                        | Question to answer                                       |
| `--timeout`         | `900`                                 | Per-call Ollama read timeout in seconds                  |
| `--num-predict`     | `3000`                                | Max tokens to generate                                   |
| `--top-k`           | `15`                                  | Number of chunks to retrieve (RAG parameter)             |
| `--chunk-size`      | `500`                                 | Chunk size in words (RAG parameter)                      |
| `--chunk-overlap`   | `50`                                  | Chunk overlap in words (RAG parameter)                   |
| `--embedding-model` | `nomic-embed-text`                    | Ollama embedding model (local)                           |
| `--doc-file`        | None                                    | Path to a text or PDF file to analyze (.txt or .pdf)     |
| `--synthetic-file`  | `input/synthetic_maintenance_log.txt` | Path to synthetic .txt file                              |
| `--output-file`     | `output/rag_analysis_output.md`       | Output markdown file                                     |

---

### Compare Outputs

To compare the outputs from RLM and RAG approaches, follow these steps:

**Step 1: Run both demos** (you must run these first):

```powershell
# Run RLM demo
python .\rlm_ollama_demo.py --no-cloud

# Run RAG demo  
python .\rag_demo.py --no-cloud
```

**Step 2: Compare the outputs** (run this manually after both demos complete):

```powershell
python .\compare_rlm_vs_rag.py --compare-only
```

This creates `output/rlm_vs_rag_comparison_report.md` comparing:

- `output/rlm_analysis_output.md` (from RLM demo)
- `output/rag_analysis_output.md` (from RAG demo)

**Important Notes:**
- You must run both demos first - the comparison script does **not** run them automatically
- The comparison script compares the basic RLM demo (`rlm_ollama_demo.py`) with RAG
- The advanced RLM demo (`rlm_ollama_advanced.py`) generates separate output files and is not included in this comparison

---

## Key Scripts Overview

### `rlm_ollama_demo.py`

- **Purpose**: RLM-style analysis with query-driven scanning
- **Modes**: Local and Cloud (auto-detects from model name or use `--cloud`/`--no-cloud`)
- **Default model**: `llama3.2:latest` (local model)
- **Best for**: Finding all recurring patterns across entire document

### `rlm_ollama_advanced.py`

- **Purpose**: Enhanced RLM with full paper features (REPL, iterative refinement, multi-level recursion)
- **Modes**: Local and Cloud (auto-detects from model name or use `--cloud`/`--no-cloud`)
- **Default model**: `llama3.2:latest` (local model)
- **Best for**: Advanced use cases requiring iterative refinement and deeper recursion

### `rag_demo.py`

- **Purpose**: RAG-style analysis with pre-chunking and semantic retrieval
- **Modes**: Local and Cloud (auto-detects from model name or use `--cloud`/`--no-cloud`)
- **Default model**: `llama3.2:latest` (local model)
- **Best for**: Fast retrieval of top-K most relevant chunks

### `compare_rlm_vs_rag.py`

- **Purpose**: Compare outputs from RLM and RAG approaches
- **Who runs it**: You (the user) run this manually after running both demos
- **Usage**: 
  1. First run `rlm_ollama_demo.py` and `rag_demo.py` separately
  2. Then run: `python .\compare_rlm_vs_rag.py --compare-only`
- **What it does**: Reads existing output files and creates a side-by-side comparison report
- **Output**: `rlm_vs_rag_comparison_report.md`
- **Note**: Does NOT run the demos automatically - you must run them first

### `synthetic_data.py`

- **Purpose**: Generate synthetic maintenance log for repeatable demos

**Example:**

```powershell
# Default writes to input/synthetic_maintenance_log.txt
python .\synthetic_data.py --entries 250 --seed 42

# Or specify custom path
python .\synthetic_data.py --out .\input\my_custom_log.txt --entries 250 --seed 42
```

---

## Choosing Between RLM Scripts

Both `rlm_ollama_demo.py` and `rlm_ollama_advanced.py` implement RLM-style analysis, but with different levels of sophistication:

### `rlm_ollama_demo.py` - Basic RLM

**Workflow:** Plan → Search → Summarize → Reduce

- **Simple keyword-based search** - LLM generates keywords, then direct search
- **Fixed chunking parameters** - Uses `--window`, `--fallback-chunk-chars`, `--fallback-overlap`
- **Single-pass approach** - No iterative refinement
- **Faster execution** - Simpler algorithm, fewer LLM calls
- **Best for:** Quick analysis, predictable behavior, straightforward use cases

### `rlm_ollama_advanced.py` - Full Paper Implementation

**Workflow:** Iterative Refinement → Recursive Summarization → Variable Storage → Final Synthesis

- **REPL environment** - LLM generates and executes Python code in a safe sandbox
- **Iterative refinement** - Refines search strategy based on results (`--max-iterations`, `--min-spans`)
- **Dynamic chunking** - LLM-controlled chunking strategies
- **Multi-level recursion** - Configurable recursion depth (`--max-recursion-depth`)
- **More sophisticated** - Implements full RLM paper features
- **Best for:** Complex analysis, adaptive search, deeper reasoning, research/experimentation

### Quick Comparison

| Feature | `rlm_ollama_demo.py` | `rlm_ollama_advanced.py` |
|---------|----------------------|--------------------------|
| Code execution | No | Yes (REPL sandbox) |
| Search strategy | Fixed keyword search | Iterative refinement |
| Chunking | Fixed parameters | LLM-controlled dynamic |
| Recursion depth | Single level | Multi-level (configurable) |
| Iteration | Single pass | Multiple iterations |
| Key parameters | `--max-spans`, `--window` | `--max-iterations`, `--min-spans`, `--max-recursion-depth` |
| Speed | Faster | Slower (more LLM calls) |
| Complexity | Simpler | More sophisticated |

**Recommendation:** Start with `rlm_ollama_demo.py` for most use cases. Use `rlm_ollama_advanced.py` when you need adaptive search strategies or are exploring the full RLM capabilities.

---

## Output Files

After running the demos, you'll find these files in the `output/` directory:

- `rlm_analysis_output.md` - RLM analysis results (always generated)
- `rlm_advanced_analysis_output.md` - RLM advanced analysis results (from advanced demo)
- `rag_analysis_output.md` - RAG analysis results (always generated)
- `rlm_vs_rag_comparison_report.md` - Side-by-side comparison (from compare script)

**Note**: To disable file output for RLM demo:

```powershell
python .\rlm_ollama_demo.py --no-cloud --output-file ""
```

---

## GPU / CUDA Setup

The setup script (`.\setup_cuda130_uv.ps1`) automatically installs PyTorch with CUDA 13.0 support from:
`https://download.pytorch.org/whl/cu130`.

To verify CUDA is working:

```powershell
python .\check_cuda.py
```

**Note**: CUDA 13.0 requires NVIDIA driver version 600+. Check with `nvidia-smi`.

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
# Place PDF in input/ directory first, then:
python .\rlm_ollama_demo.py --no-cloud --model llama3.2:3b --doc-file "report.pdf"

# RAG with PDF
python .\rag_demo.py --no-cloud --doc-file "report.pdf"
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

## Customizing Prompts

Both RLM and RAG demos load prompts from `input/proj_prompt.txt`. This allows you to customize:

- **Main question**: The analysis question used by both demos
- **RLM prompts**: System prompts for planning, chunk summarization, and final synthesis
- **RAG prompts**: System prompt for answer generation

### Editing the Prompt File

1. Open `input/proj_prompt.txt` in any text editor
2. Edit the main question (first non-comment line)
3. Edit system prompts in the `[SECTION_NAME]` sections
4. Save the file
5. Run the demos - they will automatically use your custom prompts

### Prompt File Format

```text
# Main question (first non-comment line)
What are the recurring problems, what evidence supports them, and what actions are recommended?

# RLM prompts
[RLM_SUMMARIZE_SYSTEM]
Your custom prompt here...

[RLM_REDUCE_SYSTEM]
Your custom prompt here...

[RLM_PLAN_SYSTEM]
Your custom prompt here...

# RAG prompts
[RAG_SYSTEM]
Your custom prompt here...
```

**Important**: The `input/proj_prompt.txt` file is **required**. The scripts will fail with a clear error message if it doesn't exist. This ensures prompts are always domain-appropriate and not hardcoded to a specific topic.

---

## How RLM and RAG Work (Step-by-Step)

This section explains how each approach processes documents, designed for CS students new to these concepts.

### RLM (Recursive Language Model) - How It Works

RLM treats the document as an **external file** that the LLM can programmatically access. The LLM never sees the full document at once.

#### Basic RLM (`rlm_ollama_demo.py`) - 4 Steps:

**Step 1: Plan (1 LLM call)**
- LLM receives only the **question** (not the document)
- LLM generates search keywords like `["vibration", "bearing", "fault", "alarm"]`
- Output: List of keywords to search for

**Step 2: Search (No LLM - Python code)**
- Python searches the document using those keywords
- Extracts text spans around each keyword match
- Document stays external - only relevant spans are extracted
- Output: List of text spans (chunks)

**Step 3: Summarize (N LLM calls - one per chunk)**
- Each chunk is sent to the LLM separately
- LLM extracts facts from each chunk independently
- Uses a smaller, faster model (`qwen3:0.6b`) for efficiency
- This is the "recursive" part - each chunk processed separately
- Output: List of summaries, one per chunk

**Step 4: Reduce (1 LLM call)**
- All summaries are sent to the LLM together
- LLM combines them into a final answer
- Uses the larger model (`llama3.2:latest`) for synthesis
- Output: Final answer with chunk ID references

**Visual Flow:**
```
Question → [LLM: Generate Keywords] → [Python: Search Document] → 
           (1 call)                    (no LLM)

[LLM: Summarize Chunk1] → [LLM: Summarize Chunk2] → ... → [LLM: Summarize ChunkN] →
(N calls, recursive)

[LLM: Combine All Summaries] → Final Answer
(1 call)
```

**Key Insight:** The document never enters the LLM's context window fully. Instead, the LLM controls *what* to read through code/search, then processes pieces recursively.

#### Advanced RLM (`rlm_ollama_advanced.py`) - Enhanced Features:

**Iterative Refinement Loop:**
- Instead of generating keywords once, the LLM generates **Python code** to search
- Code executes in a safe sandbox
- LLM observes results, then refines the search strategy
- Repeats until enough relevant spans are found

**Multi-Level Recursion:**
- If a chunk is too large, the LLM can recursively call itself to summarize sub-chunks
- Configurable depth (default: 3 levels)

**Variable Storage:**
- Handles very long outputs by storing intermediate results
- Prevents context window overflow

---

### RAG (Retrieval-Augmented Generation) - How It Works

RAG pre-processes the document into a searchable index, then retrieves relevant pieces when answering.

#### RAG (`rag_demo.py`) - 2 Steps:

**Step 1: Index (Before any query - no LLM)**
- Document is split into fixed-size chunks (500 words each)
- Each chunk is converted to a **vector embedding** (numerical representation)
- All embeddings are stored in an index
- This happens once, before any questions are asked

**Step 2: Retrieve & Generate (1 LLM call)**
- When a question arrives, it's converted to an embedding
- System finds the top-K most similar chunks using **cosine similarity**
- Top-K chunks are sent to the LLM in one prompt
- LLM generates answer from those chunks
- Output: Final answer with chunk references

**Visual Flow:**
```
Document → [Split into Chunks] → [Create Embeddings] → [Store in Index]
           (no LLM)              (embedding model)    (no LLM)

Question → [Create Query Embedding] → [Find Top-K Similar Chunks] → 
           (embedding model)          (cosine similarity, no LLM)

[LLM: Generate Answer from Top-K Chunks] → Final Answer
(1 call)
```

**Key Insight:** RAG uses semantic similarity (meaning-based) rather than keyword matching. It finds chunks that are "similar in meaning" to the question.

---

### How the Three Scripts Relate

```
┌─────────────────────────────────────────────────────────┐
│                    rlm_ollama_demo.py                    │
│              (Basic RLM - Simplified)                    │
│  • Simple keyword search                                │
│  • Single-pass processing                               │
│  • Faster, easier to understand                         │
└─────────────────────────────────────────────────────────┘
                    ↓ (imports utilities from)
┌─────────────────────────────────────────────────────────┐
│                rlm_ollama_advanced.py                    │
│         (Enhanced RLM - Full Paper Features)            │
│  • REPL with code execution                             │
│  • Iterative refinement                                 │
│  • Multi-level recursion                                │
│  • More sophisticated, slower                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    rag_demo.py                           │
│              (RAG - Different Approach)                 │
│  • Semantic similarity search                           │
│  • Pre-indexed embeddings                               │
│  • Single LLM call                                      │
│  • Fast for specific fact retrieval                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              compare_rlm_vs_rag.py                       │
│              (Comparison Tool)                          │
│  Compares outputs from RLM and RAG demos                │
└─────────────────────────────────────────────────────────┘
```

**Shared Utilities:**
- Both RLM scripts import utilities from `rlm_ollama_demo.py`:
  - File loading (`extract_text_from_file`, PDF/OCR support)
  - Prompt loading (`load_prompts`)
  - Ollama communication (`ollama_chat`)
  - Output formatting (`format_output`)

**Key Differences:**

| Aspect | Basic RLM | Advanced RLM | RAG |
|--------|-----------|--------------|-----|
| **Search Method** | Keyword matching | Code generation + iterative refinement | Semantic similarity |
| **When Chunking** | After query (dynamic) | After query (LLM-controlled) | Before query (fixed) |
| **LLM Calls** | 1 plan + N summaries + 1 reduce | Multiple iterations + recursive summaries | 1 call |
| **Code Execution** | No | Yes (sandbox) | No |
| **Best For** | Finding all patterns | Complex adaptive search | Finding specific facts |

---

### When to Use Which Approach

**Use Basic RLM (`rlm_ollama_demo.py`)** when:
- You want to find all recurring patterns in a document
- You need predictable, fast results
- You're learning how RLM works
- The document structure is relatively uniform

**Use Advanced RLM (`rlm_ollama_advanced.py`)** when:
- You need adaptive search strategies
- Documents have complex, variable structures
- You want to explore full RLM capabilities
- You're doing research or experimentation

**Use RAG (`rag_demo.py`)** when:
- You need to find specific facts quickly
- Questions are about isolated information (not patterns)
- You want the fastest response time
- You have many documents to search across

---

## Troubleshooting

### "Could not connect to Ollama"

- Make sure Ollama server is running: `ollama serve`
- Or open the Ollama desktop app on Windows
- Verify Ollama is accessible: `curl http://localhost:11434` (should return JSON)

### "404 Client Error: Not Found for url: http://localhost:11434/api/chat"

- **Model not found**: The 404 error often occurs when the specified model doesn't exist locally
  - Check available models: `ollama list`
  - Pull the required model: `ollama pull llama3.2:3b` (or use `llama3.2:latest` if that's what you have)
  - Use `--model` flag to specify a model you have: `python .\rlm_ollama_demo.py --no-cloud --model llama3.2:latest`
- **Ollama version issue**: Older Ollama versions (< 0.2.0) may not support `/api/chat` endpoint
  - Update Ollama: Download latest version from https://ollama.com
  - Or restart Ollama service: `ollama serve` (restart the desktop app)
- **Service not fully started**: Wait a few seconds after starting Ollama before running scripts
- **Verify endpoint**: Check if `http://localhost:11434/api/tags` works (lists available models)

### "OLLAMA_API_KEY environment variable required"

- Set it: `$env:OLLAMA_API_KEY="your_key"` (PowerShell)
- Or pass via `--api-key` flag

### Model not found / 404 errors

- **Check what models you have**: `ollama list`
- **Pull the required model**: `ollama pull llama3.2:3b` (or `ollama pull llama3.2:latest`)
- **Use a model you have**: If you only have `llama3.2:latest`, use `--model llama3.2:latest` flag
- **Default models**: 
  - `rlm_ollama_demo.py` defaults to `llama3.2:latest` - works if you have that installed
  - `rlm_ollama_advanced.py` defaults to `llama3.2:latest` - works if you have that installed
  - `rag_demo.py` defaults to `llama3.2:latest` - works if you have that installed
  - If you need `llama3.2:3b`, pull it: `ollama pull llama3.2:3b` or use `--model llama3.2:3b`
- For cloud models, ensure you have API access

### Timeout errors

- Increase `--timeout` (default: 900 seconds)
- Reduce `--max-spans` or `--top-k` for faster processing
- Use a smaller model

---

## References/Credits

This demo implements the **Recursive Language Model (RLM)** approach described in:

**Recursive Language Models**  
Alex L. Zhang, Tim Kraska, Omar Khattab  
arXiv:2512.24601, 2025  
https://arxiv.org/abs/2512.24601

> *"We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt."*

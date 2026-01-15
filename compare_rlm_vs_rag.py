"""
Comparison script to run both RLM and RAG approaches and compare outputs.
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return output."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*80)
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        print("STDERR:", result.stderr)
        return None
    
    return result.stdout

def compare_outputs(rlm_file, rag_file):
    """Compare the two output files and create a comparison report."""
    # Default to output/ directory if relative paths
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    rlm_path = Path(rlm_file) if Path(rlm_file).is_absolute() else output_dir / rlm_file
    rag_path = Path(rag_file) if Path(rag_file).is_absolute() else output_dir / rag_file
    
    if not rlm_path.exists():
        print(f"ERROR: RLM output file not found: {rlm_path}")
        return
    
    if not rag_path.exists():
        print(f"ERROR: RAG output file not found: {rag_path}")
        return
    
    rlm_content = rlm_path.read_text(encoding='utf-8')
    rag_content = rag_path.read_text(encoding='utf-8')
    
    # Create comparison report
    comparison = f"""# RLM vs RAG Comparison Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report compares the outputs from:
- **RLM approach**: `rlm_ollama_demo.py`
- **RAG approach**: `rag_demo.py`

    Both approaches analyzed the same document (`input/synthetic_maintenance_log.txt`)
with the same question: "What are the recurring problems, what evidence 
supports them, and what actions are recommended?"

---

## Key Differences in Approach

### RLM (Recursive Language Model)
- **Chunking**: Dynamic, query-dependent
- **Access**: Model writes code to search/chunk based on query
- **Retrieval**: Keyword/regex matching
- **Process**: Recursive summarization and combination
- **Chunks used**: All relevant spans found (43 chunks in this case)

### RAG (Retrieval-Augmented Generation)
- **Chunking**: Pre-chunked, fixed-size (500 words)
- **Access**: Static indexing with vector embeddings
- **Retrieval**: Semantic similarity (top-K chunks)
- **Process**: Retrieve → Generate answer
- **Chunks used**: Top-K most similar chunks (typically 10-15)

---

## Output Comparison

### RLM Output Length
{len(rlm_content)} characters

### RAG Output Length
{len(rag_content)} characters

---

## RLM Output

{rlm_content}

---

## RAG Output

{rag_content}

---

## Analysis

### Coverage
- **RLM**: Systematically scans entire document, finds all occurrences
- **RAG**: Only uses top-K retrieved chunks (may miss some occurrences)

### Evidence References
- **RLM**: References specific chunk IDs (CH-00 through CH-42)
- **RAG**: References retrieved chunk IDs (RAG_CHUNK_XXX)

### Completeness
- **RLM**: More likely to find all recurring patterns (dense task)
- **RAG**: May miss patterns if they're spread across many chunks

### Speed
- **RLM**: Slower (many LLM calls for chunk summarization)
- **RAG**: Faster (single retrieval + one LLM call)

---

## Conclusion

For **dense tasks** (like finding recurring problems across entire document):
- **RLM** is better suited because it systematically processes everything
- **RAG** may miss patterns if they're not in the top-K retrieved chunks

For **sparse tasks** (like finding specific facts):
- **RAG** is faster and more efficient
- **RLM** may be overkill

"""
    
    comparison_path = output_dir / "rlm_vs_rag_comparison_report.md"
    comparison_path.write_text(comparison, encoding='utf-8')
    print(f"\nComparison report written to: {comparison_path}")

if __name__ == "__main__":
    print("RLM vs RAG Comparison Tool")
    print("="*80)
    
    # Check if we should run both or just compare existing outputs
    if len(sys.argv) > 1 and sys.argv[1] == "--compare-only":
        print("Comparing existing output files...")
        compare_outputs(
            "rlm_analysis_output.md",  # Will look in output/ directory
            "rag_analysis_output.md"   # Will look in output/ directory
        )
    else:
        print("This script will help you run both approaches.")
        print("\nTo run RLM:")
        print("  python rlm_ollama_demo.py")
        print("\nTo run RAG:")
        print("  python rag_demo.py")
        print("\nThen run this script with --compare-only to compare:")
        print("  python compare_rlm_vs_rag.py --compare-only")

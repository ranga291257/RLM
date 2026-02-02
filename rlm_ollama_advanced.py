"""
Enhanced RLM Implementation with Full Paper Features

This implements the complete RLM paradigm from the paper:
- REPL environment with LLM-generated code execution
- Iterative refinement loop (observe → refine → execute)
- Dynamic chunking strategies (LLM-controlled)
- Multi-level recursive sub-calls
- Variable storage for unbounded output

Based on: "Recursive Language Models" (Zhang, Kraska, Khattab, arXiv:2512.24601)
"""

import json
import re
import argparse
import os
import sys
import io
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Callable
import requests

from synthetic_data import SyntheticDocConfig, write_synthetic_doc

# PDF support
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# OCR support (optional)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from ollama import Client
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False

# Import shared utilities from original file
from rlm_ollama_demo import (
    PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR,
    load_prompts, extract_text_from_file, chunk_text,
    Span, safe_json_load, ollama_chat, format_output,
    is_text_based_pdf, extract_text_from_pdf_with_ocr
)

OLLAMA_LOCAL_URL = "http://localhost:11434/api/chat"


# ============================================================================
# CODE EXECUTION SANDBOX (Restricted Python REPL)
# ============================================================================

class CodeSandbox:
    """
    Restricted Python execution environment for LLM-generated code.
    Only allows safe operations (no file system, network, imports, etc.)
    """
    
    # Allowed built-ins (whitelist approach)
    SAFE_BUILTINS = {
        'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
        'range', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'min', 'max',
        'sum', 'abs', 'round', 'any', 'all', 'print', 'repr',
        'isinstance',
        'slice', 'reversed', 'iter', 'next',
    }
    
    def __init__(self, initial_context: Dict[str, Any] = None):
        """
        Initialize sandbox with initial context (e.g., doc variable).
        """
        # Handle __builtins__ being either a dict or module
        if isinstance(__builtins__, dict):
            builtins_dict = __builtins__
        else:
            # __builtins__ is a module, get its dict
            builtins_dict = __builtins__.__dict__
        
        # Create restricted __import__ that only allows 're' and 'json'
        def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            """Only allow importing 're' and 'json' modules."""
            allowed_modules = {'re': re, 'json': json}
            if name in allowed_modules:
                return allowed_modules[name]
            raise ImportError(f"Import of '{name}' is not allowed in sandbox")
        
        # Build safe builtins with restricted import
        safe_builtins = {k: builtins_dict[k] for k in self.SAFE_BUILTINS 
                        if k in builtins_dict}
        safe_builtins['__import__'] = restricted_import
        
        self.context = {
            '__builtins__': safe_builtins,
            're': re,  # Pre-imported for convenience
            'json': json,  # Pre-imported for convenience
        }
        
        if initial_context:
            self.context.update(initial_context)
        
        self.execution_log: List[str] = []
        self.last_output: str = ""
    
    def execute(self, code: str, capture_output: bool = True) -> Tuple[Any, str]:
        """
        Execute code in sandbox and return (result, output_text).
        
        Args:
            code: Python code to execute
            capture_output: If True, capture print() statements
        
        Returns:
            Tuple of (result_value, captured_output_text)
        """
        # Capture stdout if requested
        output_buffer = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(output_buffer) if capture_output else contextlib.nullcontext():
                # Compile and execute
                _orig_sub_lm = self.context.get('sub_lm')
                compiled = compile(code, '<sandbox>', 'exec')
                exec(compiled, self.context)
                # If generated code overwrote sub_lm, restore the injected bridge
                if _orig_sub_lm is not None and self.context.get('sub_lm') is not _orig_sub_lm:
                    self.context['sub_lm'] = _orig_sub_lm
                    self.execution_log.append('WARN: sub_lm was overwritten by generated code; restored.')
            
            output_text = output_buffer.getvalue() if capture_output else ""
            self.last_output = output_text
            self.execution_log.append(f"EXECUTED: {code[:100]}...")
            
            # Try to get result from last expression (if any)
            result = self.context.get('_result', None)

            # Paper-faithful pattern: allow code to set _result_var = '<varname>'
            # to return an arbitrary variable from the sandbox (useful for long outputs).
            result_var = self.context.get('_result_var')
            if isinstance(result_var, str) and result_var in self.context:
                result = self.context.get(result_var)

            return result, output_text
            
        except Exception as e:
            error_msg = f"Sandbox execution error: {type(e).__name__}: {e}"
            self.execution_log.append(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from sandbox context."""
        return self.context.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in sandbox context."""
        self.context[name] = value
    
    def get_context_summary(self) -> str:
        """Get summary of current context (for LLM observation)."""
        summary = []
        for key, value in self.context.items():
            if not key.startswith('_') and key not in ['__builtins__', 're', 'json']:
                if isinstance(value, str):
                    preview = value[:100] + "..." if len(value) > 100 else value
                    summary.append(f"{key} = {repr(preview)}")
                elif isinstance(value, (list, dict)):
                    summary.append(f"{key} = {type(value).__name__}(len={len(value)})")
                else:
                    summary.append(f"{key} = {repr(value)}")
        return "\n".join(summary) if summary else "(empty context)"


# ============================================================================
# SUB-LM BRIDGE (Injected into the sandbox for in-REPL recursion)
# ============================================================================

class SubLMBridge:
    """A callable injected into the sandbox as `sub_lm(prompt: str) -> str`.

    This is the paper's key feature: code running *inside* the REPL can recursively
    query a (typically smaller/cheaper) sub-model.

    NOTE: Any callable injected into a Python sandbox can be introspected if the
    generated code is adversarial. This demo assumes benign code generation.
    """

    def __init__(
        self,
        *,
        sub_model: str,
        timeout_s: int,
        num_predict: int,
        use_cloud: bool,
        api_key: str | None,
    ):
        self._sub_model = sub_model
        self._timeout_s = timeout_s
        self._num_predict = num_predict
        self._use_cloud = use_cloud
        self._api_key = api_key

    def __call__(self, prompt: str) -> str:
        # Keep the interface extremely simple for codegen.
        return ollama_chat(
            self._sub_model,
            [
                {"role": "system", "content": "You are a helpful sub-model. Be concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            timeout_s=self._timeout_s,
            num_predict=self._num_predict,
            use_cloud=self._use_cloud,
            api_key=self._api_key,
        )


# ============================================================================
# LLM CODE GENERATION
# ============================================================================

def extract_code_from_llm_output(text: str) -> str:
    """
    Extract Python code from LLM output (handles markdown code blocks).
    """
    # Try to find code block
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code block, assume entire output is code
    return text.strip()


def generate_search_code(
    model: str,
    question: str,
    doc_metadata: Dict[str, Any],
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    LLM generates Python code to search the document.
    
    Returns executable Python code as string.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a Python code generation assistant. Generate ONLY valid, executable Python code. "
            "CRITICAL: Your code must be syntactically correct Python. Return code wrapped in ```python code blocks. "
            "The code runs in a sandbox with: 'doc' (document text) and 'sub_lm' (callable function). "
            "DO NOT redefine 'sub_lm'. DO NOT use undefined variables. DO NOT use incomplete syntax."
        )
    
    doc_preview = doc_metadata.get('preview', '')
    doc_length = doc_metadata.get('length', 0)
    
    user = (
        f"Question: {question}\n\n"
        f"Document metadata:\n"
        f"- Length: {doc_length:,} characters\n"
        f"- Preview (first 500 chars): {doc_preview[:500]}\n\n"
        f"Generate Python code to search this document. Available:\n"
        f"- Variable 'doc' contains full document text (string)\n"
        f"- Module 're' is pre-imported\n"
        f"- Standard Python built-ins (len, str, list, dict, range, enumerate, etc.)\n\n"
        f"REQUIREMENTS:\n"
        f"1. Code must be complete, valid Python syntax\n"
        f"2. Create a list called 'spans' (initialize as: spans = [])\n"
        f"3. Each span must be a dict: {{'chunk_id': str, 'text': str}}\n"
        f"4. Print final count: print(f'Found {{len(spans)}} spans')\n"
        f"5. DO NOT use undefined variables or incomplete code blocks\n\n"
        f"Complete working example:\n"
        f"```python\n"
        f"import re\n"
        f"keywords = ['fault', 'alarm', 'trip']\n"
        f"spans = []\n"
        f"for kw in keywords:\n"
        f"    pattern = r'\\\\b' + kw + r'\\\\b'\n"
        f"    for match in re.finditer(pattern, doc, re.IGNORECASE):\n"
        f"        start = max(0, match.start() - 450)\n"
        f"        end = min(len(doc), match.end() + 450)\n"
        f"        chunk_id = f'{{kw}}_{{match.start()}}'\n"
        f"        spans.append({{'chunk_id': chunk_id, 'text': doc[start:end]}})\n"
        f"print(f'Found {{len(spans)}} spans')\n"
        f"```\n\n"
        f"Generate similar code that searches for passages relevant to: {question}\n"
    )
    
    response = ollama_chat(
        model,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        temperature=0.2,  # Lower temperature for more consistent, valid syntax
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )
    
    return extract_code_from_llm_output(response)


def generate_chunking_code(
    model: str,
    question: str,
    doc_metadata: Dict[str, Any],
    search_results: Dict[str, Any],
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    LLM generates Python code to chunk the document dynamically.
    Used when initial search finds too few results.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a Python code generation assistant. Generate ONLY valid, executable Python code. "
            "CRITICAL: Code must be syntactically correct. Return code wrapped in ```python code blocks. "
            "DO NOT use undefined variables. DO NOT use incomplete syntax."
        )
    
    user = (
        f"Question: {question}\n\n"
        f"Document metadata:\n"
        f"- Length: {doc_metadata.get('length', 0):,} characters\n"
        f"- Preview: {doc_metadata.get('preview', '')[:500]}\n\n"
        f"Previous search results:\n"
        f"- Found {search_results.get('span_count', 0)} spans\n"
        f"- Observation: {search_results.get('observation', 'Search found too few results')}\n\n"
        f"Generate Python code to chunk the document intelligently. Available:\n"
        f"- Variable 'doc' contains full document text (string)\n"
        f"- Module 're' is pre-imported\n"
        f"- Standard Python built-ins\n\n"
        f"REQUIREMENTS:\n"
        f"1. Code must be complete, valid Python syntax\n"
        f"2. Create a list called 'chunks' (initialize as: chunks = [])\n"
        f"3. Each chunk must be a dict: {{'chunk_id': str, 'text': str}}\n"
        f"4. Print final count: print(f'Created {{len(chunks)}} chunks')\n"
        f"5. DO NOT use undefined variables or incomplete code\n\n"
        f"Complete working example:\n"
        f"```python\n"
        f"import re\n"
        f"chunks = []\n"
        f"# Split by double newlines (log entries)\n"
        f"entries = doc.split('\\n\\n')\n"
        f"for i, entry in enumerate(entries):\n"
        f"    if entry.strip():\n"
        f"        chunks.append({{'chunk_id': f'CHUNK_{{i:03d}}', 'text': entry.strip()}})\n"
        f"print(f'Created {{len(chunks)}} chunks')\n"
        f"```\n\n"
        f"Generate similar code that chunks the document based on its structure.\n"
    )
    
    response = ollama_chat(
        model,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        temperature=0.2,  # Lower temperature for more consistent, valid syntax
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )
    
    return extract_code_from_llm_output(response)


# ---------------------------------------------------------------------------
# Span utilities: parse ranges + deduplicate spans
# ---------------------------------------------------------------------------

_SPAN_RANGE_RE = re.compile(r'@(?P<start>\d+):(?P<end>\d+)$')


def _span_range(chunk_id: str) -> tuple[int | None, int | None]:
    """Extract (start,end) from chunk_id formatted like 'ALARM@38:949'."""
    if not isinstance(chunk_id, str):
        return None, None
    m = _SPAN_RANGE_RE.search(chunk_id)
    if not m:
        return None, None
    try:
        return int(m.group('start')), int(m.group('end'))
    except Exception:
        return None, None


def dedupe_spans(spans: list[Span]) -> list[Span]:
    """Remove duplicate spans (same chunk_id or same (start,end)).

    This fixes the duplicate summaries you saw in your console log.
    """
    if not spans:
        return []

    seen_ids: set[str] = set()
    seen_ranges: set[tuple[int, int]] = set()
    out: list[Span] = []

    for sp in spans:
        cid = sp.chunk_id or 'UNKNOWN'
        s, e = _span_range(cid)

        if cid in seen_ids:
            continue
        if s is not None and e is not None and (s, e) in seen_ranges:
            continue

        out.append(sp)
        seen_ids.add(cid)
        if s is not None and e is not None:
            seen_ranges.add((s, e))

    # Sort by start index if available (nice-to-have)
    def _key(sp: Span):
        s, _ = _span_range(sp.chunk_id or '')
        return (s if s is not None else 10**18, sp.chunk_id)

    out.sort(key=_key)
    return out


# ============================================================================
# ITERATIVE REFINEMENT LOOP
# ============================================================================

def iterative_search_refinement(
    model: str,
    question: str,
    doc: str,
    sandbox: CodeSandbox,
    *,
    max_iterations: int = 5,
    min_spans: int = 4,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> List[Span]:
    """
    Iterative refinement loop: generate code → execute → observe → refine.
    
    Returns list of Span objects.
    """
    doc_metadata = {
        'length': len(doc),
        'preview': doc[:1000],
    }
    
    sandbox.set_variable('doc', doc)
    
    spans = []
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"[ITERATION {iteration}/{max_iterations}] Generating search code...")
        
        # Generate code
        if iteration == 1:
            # First iteration: initial search strategy
            code = generate_search_code(
                model, question, doc_metadata,
                timeout_s=timeout_s, num_predict=num_predict,
                use_cloud=use_cloud, api_key=api_key,
                system_prompt=prompts.get("rlm_code_generation_system") if prompts else None,
            )
        else:
            # Subsequent iterations: refine based on observations
            observation = (
                f"Iteration {iteration-1} found {len(spans)} spans. "
                f"Last output: {sandbox.last_output[:200]}"
            )
            
            # Ask LLM to refine strategy
            refine_prompt = (
                f"Previous attempt found {len(spans)} spans (need at least {min_spans}).\n"
                f"Error observation: {sandbox.last_output[:500]}\n"
                f"Context: {sandbox.get_context_summary()}\n\n"
                f"Generate improved Python search code. CRITICAL:\n"
                f"- Code must be complete, valid Python syntax\n"
                f"- Initialize 'spans = []' at the start\n"
                f"- Each span must be dict: {{'chunk_id': str, 'text': str}}\n"
                f"- End with: print(f'Found {{len(spans)}} spans')\n"
                f"- DO NOT use incomplete code blocks or undefined variables\n\n"
                f"Improvement strategies:\n"
                f"- If syntax error: Fix the syntax completely\n"
                f"- If too few results: Use broader keywords or simpler patterns\n"
                f"- If too many results: Use more specific patterns\n"
                f"- Try different approach: regex, string.find(), or string splitting\n\n"
                f"Generate complete, working Python code:\n"
            )
            
            code = ollama_chat(
                model,
                [{"role": "system", "content": "Generate improved Python search code. CRITICAL: Code must be complete, valid Python syntax. Return ONLY executable code in ```python blocks. DO NOT redefine sub_lm. DO NOT use undefined variables."},
                 {"role": "user", "content": refine_prompt}],
                temperature=0.2,  # Lower temperature for more consistent syntax
                timeout_s=timeout_s,
                num_predict=num_predict,
                use_cloud=use_cloud,
                api_key=api_key,
            )
            code = extract_code_from_llm_output(code)
        
        print(f"[ITERATION {iteration}] Executing generated code...")
        print(f"[CODE PREVIEW] {code[:200]}...")
        
        # Execute code
        try:
            result, output = sandbox.execute(code, capture_output=True)
            print(f"[ITERATION {iteration}] Output: {output[:200]}")
            
            # Extract spans from sandbox
            spans_var = sandbox.get_variable('spans')
            if spans_var:
                if isinstance(spans_var, list):
                    # Convert dict spans to Span objects
                    spans = []
                    for s in spans_var:
                        if isinstance(s, dict):
                            spans.append(Span(
                                chunk_id=s.get('chunk_id', 'UNKNOWN'),
                                text=s.get('text', '')
                            ))
                        elif isinstance(s, Span):
                            spans.append(s)
                    
                    _before = len(spans)
                    spans = dedupe_spans(spans)
                    _after = len(spans)
                    if _after != _before:
                        print(f"[ITERATION {iteration}] Deduped spans: {_before} -> {_after}")

                    print(f"[ITERATION {iteration}] Found {len(spans)} spans")
                    
                    # Check if we have enough
                    if len(spans) >= min_spans:
                        print(f"[ITERATION {iteration}] ✓ Sufficient spans found!")
                        return spans
                    else:
                        print(f"[ITERATION {iteration}] ⚠ Only {len(spans)} spans (need {min_spans}), refining...")
                else:
                    print(f"[ITERATION {iteration}] ⚠ 'spans' variable is not a list")
            else:
                print(f"[ITERATION {iteration}] ⚠ No 'spans' variable found in output")
                
        except Exception as e:
            print(f"[ITERATION {iteration}] ✗ Execution error: {e}")
            # Continue to next iteration
    
    # If we get here, iterations exhausted - try fallback
    print(f"[FALLBACK] Iterations exhausted. Using dynamic chunking...")
    return dynamic_chunking_fallback(
        model, question, doc, sandbox,
        timeout_s=timeout_s, num_predict=num_predict,
        use_cloud=use_cloud, api_key=api_key, prompts=prompts,
    )


def dynamic_chunking_fallback(
    model: str,
    question: str,
    doc: str,
    sandbox: CodeSandbox,
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> List[Span]:
    """
    LLM-controlled dynamic chunking when search fails.
    """
    print("[DYNAMIC CHUNKING] Generating chunking strategy...")
    
    doc_metadata = {
        'length': len(doc),
        'preview': doc[:1000],
    }
    
    search_results = {
        'span_count': 0,
        'observation': 'Search found too few results, using intelligent chunking',
    }
    
    code = generate_chunking_code(
        model, question, doc_metadata, search_results,
        timeout_s=timeout_s, num_predict=num_predict,
        use_cloud=use_cloud, api_key=api_key,
        system_prompt=prompts.get("rlm_chunking_system") if prompts else None,
    )
    
    print(f"[DYNAMIC CHUNKING] Executing chunking code...")
    print(f"[CODE PREVIEW] {code[:200]}...")
    
    try:
        result, output = sandbox.execute(code, capture_output=True)
        print(f"[DYNAMIC CHUNKING] Output: {output[:200]}")
        
        # Extract chunks
        chunks_var = sandbox.get_variable('chunks')
        if chunks_var and isinstance(chunks_var, list):
            spans = []
            for i, c in enumerate(chunks_var):
                if isinstance(c, dict):
                    spans.append(Span(
                        chunk_id=c.get('chunk_id', f'CHUNK_{i:02d}'),
                        text=c.get('text', str(c))
                    ))
                elif isinstance(c, str):
                    spans.append(Span(chunk_id=f'CHUNK_{i:02d}', text=c))
            
            spans = dedupe_spans(spans)
            print(f"[DYNAMIC CHUNKING] Created {len(spans)} chunks")
            return spans
    except Exception as e:
        print(f"[DYNAMIC CHUNKING] Error: {e}")
    
    # Ultimate fallback: simple chunking
    print("[FALLBACK] Using simple character-based chunking")
    raw_chunks = chunk_text(doc, max_chars=2200, overlap=200)
    return [Span(chunk_id=f"CHUNK_{i:02d}", text=c) for i, c in enumerate(raw_chunks)]


# ============================================================================
# RECURSIVE SUMMARIZATION WITH DEEPER RECURSION
# ============================================================================

def recursive_summarize(
    model: str,
    question: str,
    chunk: Span,
    *,
    depth: int = 0,
    max_depth: int = 3,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Recursive summarization with configurable depth.
    
    If chunk is too complex, can recursively decompose it further.
    """
    if depth >= max_depth:
        # Base case: summarize directly
        return summarize_chunk_simple(
            model, question, chunk,
            timeout_s=timeout_s, num_predict=num_predict,
            use_cloud=use_cloud, api_key=api_key,
            system_prompt=system_prompt,
        )
    
    # Check if chunk needs further decomposition
    if len(chunk.text) > 5000:  # Large chunk - might need recursion
        print(f"[RECURSIVE {depth}] Large chunk ({len(chunk.text)} chars), checking if decomposition needed...")
        
        # Ask LLM if this chunk should be decomposed
        should_decompose = ask_should_decompose(
            model, question, chunk, depth,
            timeout_s=timeout_s, num_predict=num_predict,
            use_cloud=use_cloud, api_key=api_key,
        )
        
        if should_decompose:
            print(f"[RECURSIVE {depth}] Decomposing chunk into sub-chunks...")
            # Split chunk into sub-chunks
            sub_chunks = split_chunk_intelligently(chunk)
            
            # Recursively summarize each sub-chunk
            sub_summaries = []
            for i, sub_chunk in enumerate(sub_chunks):
                print(f"[RECURSIVE {depth+1}] Summarizing sub-chunk {i+1}/{len(sub_chunks)}...")
                sub_summary = recursive_summarize(
                    model, question, sub_chunk,
                    depth=depth+1, max_depth=max_depth,
                    timeout_s=timeout_s, num_predict=num_predict,
                    use_cloud=use_cloud, api_key=api_key,
                    system_prompt=system_prompt,
                )
                sub_summaries.append((sub_chunk.chunk_id, sub_summary))
            
            # Merge sub-summaries
            return merge_sub_summaries(
                model, question, chunk.chunk_id, sub_summaries,
                timeout_s=timeout_s, num_predict=num_predict,
                use_cloud=use_cloud, api_key=api_key,
            )
    
    # Normal summarization
    return summarize_chunk_simple(
        model, question, chunk,
        timeout_s=timeout_s, num_predict=num_predict,
        use_cloud=use_cloud, api_key=api_key,
        system_prompt=system_prompt,
    )


def summarize_chunk_simple(
    model: str,
    question: str,
    chunk: Span,
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Simple chunk summarization (base case)."""
    if system_prompt is None:
        system_prompt = (
            "You are a careful analyst. Extract only what is supported by the provided chunk. "
            "Return a compact bullet list of facts relevant to the question. "
            "If something is unknown, say 'Not in chunk'."
        )
    
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CHUNK_ID: {chunk.chunk_id}\n"
        f"CHUNK_TEXT:\n{chunk.text}\n\n"
        "OUTPUT FORMAT:\n"
        "- Fact: ... (evidence: short quote)\n"
    )
    
    return ollama_chat(
        model,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        temperature=0.1,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )


def ask_should_decompose(
    model: str,
    question: str,
    chunk: Span,
    depth: int,
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> bool:
    """Ask LLM if chunk should be decomposed further."""
    user = (
        f"Question: {question}\n\n"
        f"Chunk (depth {depth}):\n"
        f"ID: {chunk.chunk_id}\n"
        f"Length: {len(chunk.text)} characters\n"
        f"Preview: {chunk.text[:500]}...\n\n"
        f"Should this chunk be decomposed into smaller sub-chunks for better analysis?\n"
        f"Answer with JSON: {{\"decompose\": true/false, \"reason\": \"...\"}}"
    )
    
    response = ollama_chat(
        model,
        [{"role": "system", "content": "Answer with JSON only."}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )
    
    try:
        decision = safe_json_load(response)
        return decision.get("decompose", False)
    except (ValueError, KeyError, TypeError):
        return False  # Default: don't decompose if unclear


def split_chunk_intelligently(chunk: Span) -> List[Span]:
    """Split a large chunk into smaller sub-chunks intelligently."""
    text = chunk.text
    
    # Try to split by paragraphs first
    if '\n\n' in text:
        parts = text.split('\n\n')
    elif '\n' in text:
        parts = text.split('\n')
    else:
        # Fallback: character-based
        parts = [text[i:i+2000] for i in range(0, len(text), 2000)]
    
    sub_chunks = []
    for i, part in enumerate(parts):
        if part.strip():
            sub_chunks.append(Span(
                chunk_id=f"{chunk.chunk_id}.SUB_{i:02d}",
                text=part.strip()
            ))
    
    return sub_chunks if sub_chunks else [chunk]


def merge_sub_summaries(
    model: str,
    question: str,
    parent_chunk_id: str,
    sub_summaries: List[Tuple[str, str]],
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """Merge sub-chunk summaries into parent summary."""
    system = "You are merging sub-chunk summaries into a coherent parent summary."
    
    joined = "\n\n".join(f"=== {cid} ===\n{summ}" for cid, summ in sub_summaries)
    
    user = (
        f"QUESTION: {question}\n\n"
        f"PARENT CHUNK: {parent_chunk_id}\n\n"
        f"SUB-CHUNK SUMMARIES:\n{joined}\n\n"
        f"Merge these into one summary for the parent chunk."
    )
    
    return ollama_chat(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )


# ============================================================================
# LONG OUTPUT VIA VARIABLES
# ============================================================================

class VariableStorage:
    """Store long outputs in variables to bypass token limits."""
    
    def __init__(self):
        self.variables: Dict[str, str] = {}
        self.counter = 0
    
    def store(self, content: str, prefix: str = "output") -> str:
        """Store content and return variable name."""
        var_name = f"{prefix}_{self.counter}"
        self.counter += 1
        self.variables[var_name] = content
        return var_name
    
    def get(self, var_name: str) -> str:
        """Retrieve stored content."""
        return self.variables.get(var_name, "")
    
    def get_all(self) -> Dict[str, str]:
        """Get all stored variables."""
        return self.variables.copy()
    
    def combine(self, var_names: List[str], separator: str = "\n\n") -> str:
        """Combine multiple variables into one string."""
        parts = [self.get(v) for v in var_names if v in self.variables]
        return separator.join(parts)


def reduce_summaries_with_variables(
    model: str,
    question: str,
    summaries: List[Tuple[str, str]],
    storage: VariableStorage,
    *,
    max_chunk_size: int = 8000,  # Tokens per chunk
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Reduce summaries using variable storage for unbounded output.
    Splits large summary sets into chunks, processes each, stores in variables,
    then combines.
    """
    if system_prompt is None:
        system_prompt = (
            "You are synthesizing multiple chunk-level findings into one final answer. "
            "Do not invent facts. When you state a claim, include supporting CHUNK_ID(s)."
        )
    
    # If summaries fit in one call, use simple reduction
    total_size = sum(len(f"=== {cid} ===\n{summ}") for cid, summ in summaries)
    if total_size < max_chunk_size:
        return reduce_summaries_simple(
            model, question, summaries,
            timeout_s=timeout_s, num_predict=num_predict,
            use_cloud=use_cloud, api_key=api_key,
            system_prompt=system_prompt,
        )
    
    # Split into chunks
    print(f"[VARIABLE STORAGE] Large output ({total_size} chars), splitting into chunks...")
    chunks = []
    current_chunk = []
    current_size = 0
    
    for cid, summ in summaries:
        item_size = len(f"=== {cid} ===\n{summ}\n\n")
        if current_size + item_size > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [(cid, summ)]
            current_size = item_size
        else:
            current_chunk.append((cid, summ))
            current_size += item_size
    
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"[VARIABLE STORAGE] Split into {len(chunks)} chunks")
    
    # Process each chunk and store in variables
    var_names = []
    for i, chunk_summaries in enumerate(chunks):
        print(f"[VARIABLE STORAGE] Processing chunk {i+1}/{len(chunks)}...")
        partial_result = reduce_summaries_simple(
            model, question, chunk_summaries,
            timeout_s=timeout_s, num_predict=num_predict,
            use_cloud=use_cloud, api_key=api_key,
            system_prompt=system_prompt,
        )
        var_name = storage.store(partial_result, prefix="partial_result")
        var_names.append(var_name)
    
    # Combine partial results
    print(f"[VARIABLE STORAGE] Combining {len(var_names)} partial results...")
    combined = storage.combine(var_names, separator="\n\n---\n\n")
    
    # Final synthesis
    print(f"[VARIABLE STORAGE] Final synthesis...")
    final = reduce_summaries_simple(
        model, question, [("COMBINED", combined)],
        timeout_s=timeout_s, num_predict=num_predict,
        use_cloud=use_cloud, api_key=api_key,
        system_prompt=system_prompt,
    )
    
    return final


def reduce_summaries_simple(
    model: str,
    question: str,
    summaries: List[Tuple[str, str]],
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Simple reduction (original implementation)."""
    if system_prompt is None:
        system_prompt = (
            "You are synthesizing multiple chunk-level findings into one final answer. "
            "Do not invent facts. When you state a claim, include supporting CHUNK_ID(s). "
            "Use clear, concise formatting."
        )
    
    joined = "\n\n".join(f"=== {cid} ===\n{summ}" for cid, summ in summaries)
    
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CHUNK SUMMARIES:\n{joined}\n\n"
        f"FINAL OUTPUT FORMAT:\n"
        f"Use clear, readable formatting with bullet points.\n"
        f"Include CHUNK_ID references inline, e.g., (CH-01, CH-02)\n"
    )
    
    return ollama_chat(
        model,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )


# ============================================================================
# ENHANCED RLM PIPELINE
# ============================================================================

def rlm_answer_question_advanced(
    root_model: str,
    sub_model: str,
    doc: str,
    question: str,
    *,
    timeout_s: int = 600,
    num_predict: int = 512,
    max_iterations: int = 5,
    min_spans: int = 4,
    max_recursion_depth: int = 3,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> str:
    """
    Enhanced RLM pipeline with full paper features:
    - REPL environment with code execution
    - Iterative refinement
    - Dynamic chunking
    - Multi-level recursion
    - Variable storage for long outputs
    """
    # Load prompts
    if prompts is None:
        prompts = load_prompts()
    
    # Initialize sandbox and storage
    sandbox = CodeSandbox({'doc': doc})
    sandbox.set_variable('sub_lm', SubLMBridge(sub_model=sub_model, timeout_s=timeout_s, num_predict=max(256, num_predict//6), use_cloud=use_cloud, api_key=api_key))
    storage = VariableStorage()
    
    print("\n" + "="*70)
    print("ENHANCED RLM PIPELINE (Full Paper Implementation)")
    print("="*70)
    
    # STEP 1: Iterative search refinement
    print(f"\n[STEP 1] Iterative search refinement (max {max_iterations} iterations)...")
    spans = iterative_search_refinement(
        root_model, question, doc, sandbox,
        max_iterations=max_iterations, min_spans=min_spans,
        timeout_s=timeout_s, num_predict=num_predict,
        use_cloud=use_cloud, api_key=api_key, prompts=prompts,
    )
    print(f"[STEP 1] ✓ Found {len(spans)} spans")
    
    # STEP 2: Recursive summarization
    print(f"\n[STEP 2] Recursive summarization (max depth {max_recursion_depth})...")
    chunk_summaries: List[Tuple[str, str]] = []
    for i, sp in enumerate(spans):
        print(f"[STEP 2] Summarizing {i+1}/{len(spans)}: {sp.chunk_id}")
        summ = recursive_summarize(
            sub_model, question, sp,
            depth=0, max_depth=max_recursion_depth,
            timeout_s=timeout_s, num_predict=num_predict,
            use_cloud=use_cloud, api_key=api_key,
            system_prompt=prompts.get("rlm_summarize_system"),
        )
        chunk_summaries.append((sp.chunk_id, summ))
    print(f"[STEP 2] ✓ Generated {len(chunk_summaries)} summaries")
    
    # STEP 3: Reduction with variable storage
    print(f"\n[STEP 3] Final reduction (with variable storage for long outputs)...")
    final = reduce_summaries_with_variables(
        root_model, question, chunk_summaries, storage,
        timeout_s=timeout_s, num_predict=num_predict,
        use_cloud=use_cloud, api_key=api_key,
        system_prompt=prompts.get("rlm_reduce_system"),
    )
    print(f"[STEP 3] ✓ Final answer generated ({len(final)} chars)")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70 + "\n")
    
    return final


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced RLM demo with full paper features (REPL, iteration, recursion)."
    )
    parser.add_argument("--model", default="llama3.2:latest", 
                       help="Ollama model name.")
    parser.add_argument("--root-model", default="llama3.2:latest", help="Root model (planner/controller). Defaults to --model.")
    parser.add_argument("--sub-model", default="qwen3:0.6b", help="Sub-model used for recursive calls. Defaults to root model.")
    parser.add_argument("--cloud", action="store_true", default=None,
                       help="Force use of Ollama Cloud.")
    parser.add_argument("--no-cloud", dest="cloud", action="store_false",
                       help="Force use of local Ollama server.")
    parser.add_argument("--api-key", default=None,
                       help="Ollama Cloud API key (or set OLLAMA_API_KEY env var).")
    
    # Try to load default question
    try:
        _default_prompts = load_prompts()
        _default_question = _default_prompts.get("question", "")
    except (FileNotFoundError, ValueError):
        _default_question = ""
    
    parser.add_argument(
        "--question",
        default=_default_question if _default_question else None,
        required=not _default_question,
        help="Question to answer."
    )
    parser.add_argument("--timeout", type=int, default=900,
                       help="Per-call Ollama read timeout in seconds.")
    parser.add_argument("--num-predict", type=int, default=3000,
                       help="Max tokens to generate per call.")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Max iterations for search refinement.")
    parser.add_argument("--min-spans", type=int, default=4,
                       help="Minimum spans to find before stopping iteration.")
    parser.add_argument("--max-recursion-depth", type=int, default=3,
                       help="Maximum recursion depth for chunk summarization.")
    
    # Document input
    parser.add_argument(
        "--doc-file",
        default=None,
        help="Path to a text or PDF file to analyze (.txt or .pdf).",
    )
    parser.add_argument(
        "--synthetic-file",
        default=str(INPUT_DIR / "synthetic_maintenance_log.txt"),
        help="Path to synthetic .txt file.",
    )
    parser.add_argument("--generate-synthetic", action="store_true",
                       help="Generate/overwrite the synthetic file before running.")
    parser.add_argument("--entries", type=int, default=250,
                       help="Synthetic entries to generate.")
    parser.add_argument("--seed", type=int, default=42,
                       help="Synthetic RNG seed.")
    parser.add_argument("--asset", default="C-201",
                       help="Asset tag for synthetic doc.")
    
    # Output
    parser.add_argument(
        "--output-file",
        default=str(OUTPUT_DIR / "rlm_advanced_analysis_output.md"),
        help="Write output to markdown file.",
    )
    
    args = parser.parse_args()

    # Resolve root/sub models
    root_model = args.root_model or args.model
    sub_model = args.sub_model or root_model
    
    # Determine cloud/local mode
    USE_CLOUD = args.cloud
    if USE_CLOUD is None:
        USE_CLOUD = root_model.endswith("-cloud")
    
    API_KEY = args.api_key or os.getenv("OLLAMA_API_KEY")
    if USE_CLOUD and not API_KEY:
        print("ERROR: OLLAMA_API_KEY required for cloud mode.")
        sys.exit(1)
    
    # Load document
    if args.doc_file:
        doc_path = Path(args.doc_file)
        if not doc_path.is_absolute():
            doc_path = INPUT_DIR / doc_path
        if not doc_path.exists():
            print(f"ERROR: File not found: {doc_path}")
            sys.exit(1)
        doc = extract_text_from_file(doc_path)
    else:
        syn_path = Path(args.synthetic_file)
        if args.generate_synthetic or not syn_path.exists():
            cfg = SyntheticDocConfig(asset=args.asset, num_entries=args.entries, seed=args.seed)
            write_synthetic_doc(str(syn_path), cfg)
        doc = extract_text_from_file(syn_path)
    
    # Load prompts
    try:
        prompts = load_prompts()
    except (FileNotFoundError, ValueError) as e:
        print(f"[WARNING] {e}")
        print(f"[WARNING] Continuing with function default prompts.")
        prompts = None
    
    # Run enhanced RLM
    print("\n--- Enhanced RLM-style answer (Ollama) ---\n")
    try:
        answer = rlm_answer_question_advanced(
            root_model,
            sub_model,
            doc,
            args.question,
            timeout_s=args.timeout,
            num_predict=args.num_predict,
            max_iterations=args.max_iterations,
            min_spans=args.min_spans,
            max_recursion_depth=args.max_recursion_depth,
            use_cloud=USE_CLOUD,
            api_key=API_KEY,
            prompts=prompts,
        )
        
        # Format and display
        formatted = format_output(answer)
        print(formatted)
        
        # Write to file
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(formatted, encoding="utf-8")
            print(f"\n[INFO] Output written to: {output_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

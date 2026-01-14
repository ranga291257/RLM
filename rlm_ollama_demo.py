import json
import re
import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import requests

from synthetic_data import SyntheticDocConfig, write_synthetic_doc

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from ollama import Client
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False


# -----------------------------
# Ollama client (supports both local and cloud)
# -----------------------------
OLLAMA_LOCAL_URL = "http://localhost:11434/api/chat"

# Cache cloud client per API key to avoid recreating
_cloud_clients: Dict[str, Client] = {}

def _get_cloud_client(api_key: str) -> Client:
    """Get or create a cached cloud client for the given API key."""
    if api_key not in _cloud_clients:
        _cloud_clients[api_key] = Client(
            host='https://ollama.com',
            headers={'Authorization': f'Bearer {api_key}'}
        )
    return _cloud_clients[api_key]

def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    timeout_s: int = 600,
    num_predict: int = 512,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Ollama chat client supporting both local and cloud modes.
    """
    if use_cloud:
        if not OLLAMA_CLIENT_AVAILABLE:
            raise ImportError(
                "ollama package required for cloud mode. Install with: pip install ollama"
            )
        if not api_key:
            api_key = os.environ.get('OLLAMA_API_KEY')
            if not api_key:
                raise ValueError(
                    "OLLAMA_API_KEY environment variable required for cloud mode. "
                    "Set it or pass via --api-key"
                )
        
        client = _get_cloud_client(api_key)
        
        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
                stream=False,
            )
            # Response structure: ChatResponse object with message attribute
            # Some models (like gpt-oss) may put reasoning in "thinking" field
            # Handle both dict and object responses
            if isinstance(response, dict):
                message = response.get("message", {})
                content = message.get("content", "")
                # Fallback to thinking if content is empty (some models use thinking field)
                if not content and "thinking" in message:
                    content = message.get("thinking", "")
                done_reason = response.get("done_reason", "")
            else:
                # ChatResponse object (Pydantic model)
                # Access message attribute which contains content and thinking
                if hasattr(response, 'message'):
                    message_obj = response.message
                    content = getattr(message_obj, 'content', '') or ''
                    # Fallback to thinking if content is empty (models like gpt-oss use thinking)
                    if not content:
                        thinking = getattr(message_obj, 'thinking', None)
                        if thinking:
                            content = thinking
                    done_reason = getattr(response, 'done_reason', '')
                else:
                    content = ''
                    done_reason = ''
            
            # Check if response was truncated due to length limit
            if not content:
                error_msg = f"Empty response from Ollama Cloud"
                if done_reason == 'length':
                    error_msg += f" (hit token limit of {num_predict}). Try increasing --num-predict."
                error_msg += f" Full response: {response}"
                raise ValueError(error_msg)
            
            # Warn if truncated but we still got content
            if done_reason == 'length' and content:
                print(f"[WARNING] Response truncated at {num_predict} tokens. Consider increasing --num-predict for complete output.")
            
            return content
        except Exception as e:
            raise RuntimeError(f"Ollama Cloud API error: {e}")
    else:
        # Local mode using HTTP requests
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": num_predict},
            "stream": False,
        }
        try:
            r = requests.post(OLLAMA_LOCAL_URL, json=payload, timeout=(10, timeout_s))
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                raise ValueError(f"Empty response from Ollama. Full response: {data}")
            return content
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {OLLAMA_LOCAL_URL}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.ReadTimeout:
            raise TimeoutError(
                f"Ollama call timed out after {timeout_s}s. "
                "Try increasing --timeout, reducing --max-spans/--max-chars, or using a smaller model."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")


# -----------------------------
# RLM-ish utilities
# -----------------------------
@dataclass
class Span:
    chunk_id: str
    text: str

def safe_json_load(s: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM output robustly.
    """
    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to find a JSON object inside text by matching balanced braces
    start_idx = s.find('{')
    if start_idx == -1:
        raise ValueError("Could not find JSON object in model output.")
    
    # Count braces to find matching closing brace
    brace_count = 0
    for i in range(start_idx, len(s)):
        if s[i] == '{':
            brace_count += 1
        elif s[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = s[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except Exception:
                    break
    
    # Fallback: try regex match (greedy)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    
    raise ValueError("Could not find valid JSON object in model output.")


def format_output(text: str, max_line_length: int = 100) -> str:
    """
    Clean up and format output text for better readability.
    """
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            formatted_lines.append('')
            continue
        
        # Don't break markdown table separators or headers
        if line.strip().startswith('|') and '---' in line:
            formatted_lines.append(line)
            continue
        
        # For very long lines, try to break them intelligently
        if len(line) > max_line_length:
            # Try to break at sentence boundaries first
            if '. ' in line:
                parts = line.split('. ')
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        part += '.'
                    if len(part) > max_line_length:
                        # Break at commas or other punctuation
                        if ', ' in part:
                            subparts = part.split(', ')
                            for j, subpart in enumerate(subparts):
                                if j < len(subparts) - 1:
                                    subpart += ','
                                formatted_lines.append(subpart.strip())
                        else:
                            formatted_lines.append(part.strip())
                    else:
                        formatted_lines.append(part.strip())
            else:
                # Just break at spaces if no sentence boundaries
                words = line.split()
                current_line = []
                current_length = 0
                for word in words:
                    if current_length + len(word) + 1 > max_line_length and current_line:
                        formatted_lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1
                if current_line:
                    formatted_lines.append(' '.join(current_line))
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def chunk_text(text: str, max_chars: int = 2500, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking (good enough for the demo).
    """
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        i += (max_chars - overlap)
    return chunks


def find_spans_by_keywords(doc: str, keywords: List[str], window: int = 900, max_spans: int = 30) -> List[Span]:
    """
    Search the doc for keywords and extract surrounding windows.
    """
    spans: List[Span] = []
    seen: set = set()

    for kw in keywords:
        if not kw.strip():
            continue
        # word-ish search
        pattern = re.compile(re.escape(kw), flags=re.IGNORECASE)
        for match in pattern.finditer(doc):
            start = max(match.start() - window // 2, 0)
            end = min(match.end() + window // 2, len(doc))
            snippet = doc[start:end].strip()

            # dedupe by snippet hash
            key = (kw.lower(), start, end)
            if key in seen:
                continue
            seen.add(key)

            spans.append(Span(chunk_id=f"{kw.upper()}@{start}:{end}", text=snippet))
            if len(spans) >= max_spans:
                return spans

    return spans


def summarize_chunk(
    model: str,
    question: str,
    chunk: Span,
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Ask the LLM to extract structured facts from a chunk.
    """
    system = (
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
        "- Fact: ... (evidence: short quote)\n"
    )
    return ollama_chat(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )


def reduce_summaries(
    model: str,
    question: str,
    summaries: List[Tuple[str, str]],
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """
    Combine chunk summaries into final answer with references.
    summaries: list of (chunk_id, summary_text)
    """
    system = (
        "You are synthesizing multiple chunk-level findings into one final answer. "
        "Do not invent facts. When you state a claim, include supporting CHUNK_ID(s). "
        "If evidence conflicts, call it out. "
        "Use clear, concise formatting. Avoid overly long lines or complex markdown tables."
    )

    joined = "\n\n".join(
        f"=== {cid} ===\n{summ}" for cid, summ in summaries
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CHUNK SUMMARIES:\n{joined}\n\n"
        "FINAL OUTPUT FORMAT:\n"
        "Use clear, readable formatting:\n"
        "- Use bullet points, not complex tables\n"
        "- Keep lines under 100 characters when possible\n"
        "- Group related information\n"
        "- Include CHUNK_ID references inline, e.g., (CH-01, CH-02)\n"
        "\n"
        "Structure:\n"
        "1) Direct answer (2-6 concise bullets)\n"
        "2) Recommended next checks (if applicable, 2-4 bullets)\n"
        "3) Evidence map: key claims with CHUNK_ID(s) references\n"
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


def plan_search(
    model: str,
    question: str,
    *,
    timeout_s: int,
    num_predict: int,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    LLM proposes a search plan (keywords + optional regex hints).
    This is the 'RLM-style planning step'.
    """
    system = (
        "You propose a search plan for finding relevant passages in a large document stored externally. "
        "Return STRICT JSON only. No commentary."
    )
    user = (
        f"Question: {question}\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        '  "keywords": ["..."],\n'
        '  "notes": "short reason for keywords"\n'
        "}\n\n"
        "Rules:\n"
        "- 6 to 12 keywords max\n"
        "- Prefer concrete terms (fault codes, components, symptoms, actions)\n"
    )
    raw = ollama_chat(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )
    return safe_json_load(raw)


# -----------------------------
# RLM pipeline
# -----------------------------
def rlm_answer_question(
    model: str,
    doc: str,
    question: str,
    *,
    timeout_s: int = 600,
    num_predict: int = 512,
    window: int = 900,
    max_spans: int = 18,
    fallback_chunk_chars: int = 2200,
    fallback_overlap: int = 200,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> str:
    # 1) Plan search (LLM proposes keywords)
    print(f"[DEBUG] Step 1: Planning search with model {model} ({'cloud' if use_cloud else 'local'})...")
    plan = plan_search(model, question, timeout_s=timeout_s, num_predict=num_predict, use_cloud=use_cloud, api_key=api_key)
    keywords = plan.get("keywords", [])
    if not keywords:
        keywords = ["alarm", "trip", "fault", "recommendation"]
    print(f"[DEBUG] Keywords: {keywords}")

    # 2) Programmatic search over external doc (no huge prompt)
    print(f"[DEBUG] Step 2: Searching document ({len(doc)} chars)...")
    spans = find_spans_by_keywords(doc, keywords=keywords, window=window, max_spans=max_spans)
    print(f"[DEBUG] Found {len(spans)} spans")

    # If search yields too little, fallback to chunking whole doc
    if len(spans) < 4:
        print(f"[DEBUG] Too few spans, falling back to chunking...")
        raw_chunks = chunk_text(doc, max_chars=fallback_chunk_chars, overlap=fallback_overlap)
        spans = [Span(chunk_id=f"CHUNK_{i:02d}", text=c) for i, c in enumerate(raw_chunks)]
        print(f"[DEBUG] Created {len(spans)} chunks")

    # 3) Recursive chunk summarization (subcalls)
    print(f"[DEBUG] Step 3: Summarizing {len(spans)} chunks...")
    chunk_summaries: List[Tuple[str, str]] = []
    for i, sp in enumerate(spans):
        print(f"[DEBUG] Summarizing chunk {i+1}/{len(spans)}: {sp.chunk_id}")
        summ = summarize_chunk(model, question, sp, timeout_s=timeout_s, num_predict=num_predict, use_cloud=use_cloud, api_key=api_key)
        chunk_summaries.append((sp.chunk_id, summ))

    # 4) Reduce step: stitch into final answer with evidence map
    print(f"[DEBUG] Step 4: Reducing {len(chunk_summaries)} summaries into final answer...")
    if not chunk_summaries:
        return "ERROR: No chunk summaries were generated. Check if spans were found and summarized successfully."
    # Use higher token limit for final synthesis step
    final_num_predict = max(num_predict, 3000)  # Ensure at least 3000 for final step
    final = reduce_summaries(model, question, chunk_summaries, timeout_s=timeout_s, num_predict=final_num_predict, use_cloud=use_cloud, api_key=api_key)
    if not final or not final.strip():
        return "ERROR: Final reduction step returned empty answer. Check Ollama model responses."
    print(f"[DEBUG] Final answer length: {len(final)} chars")
    return final


if __name__ == "__main__":
    # Treat the folder containing this file as the project root.
    # (This repo keeps `rlm_ollama_demo.py` and `synthetic_maintenance_log.txt` side-by-side.)
    PROJECT_ROOT = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="RLM-style Ollama demo (loads a doc file).")
    parser.add_argument("--model", default="gpt-oss:120b-cloud", help="Ollama model name (cloud models end with '-cloud').")
    parser.add_argument("--cloud", action="store_true", default=None, help="Force use of Ollama Cloud.")
    parser.add_argument("--no-cloud", dest="cloud", action="store_false", help="Force use of local Ollama server.")
    parser.add_argument("--api-key", default=None, help="Ollama Cloud API key (or set OLLAMA_API_KEY env var).")
    parser.add_argument("--question", default="What are the recurring problems, what evidence supports them, and what actions are recommended?")
    parser.add_argument("--timeout", type=int, default=900, help="Per-call Ollama read timeout in seconds.")
    parser.add_argument("--num-predict", type=int, default=3000, help="Max tokens to generate per call (increased for models with thinking/reasoning and final synthesis).")
    parser.add_argument("--max-spans", type=int, default=8, help="Max retrieved spans to summarize (lower = faster).")
    parser.add_argument("--window", type=int, default=600, help="Chars around each keyword hit to extract (lower = faster).")
    parser.add_argument("--fallback-chunk-chars", type=int, default=2200, help="Chunk size if keyword search returns too little.")
    parser.add_argument("--fallback-overlap", type=int, default=200, help="Chunk overlap if keyword search returns too little.")

    # You must provide either a real doc file, or a synthetic file.
    parser.add_argument("--doc-file", default=None, help="Path to a text file to analyze.")

    # Synthetic doc is stored as a separate .txt file
    parser.add_argument(
        "--synthetic-file",
        default=str(PROJECT_ROOT / "synthetic_maintenance_log.txt"),
        help="Path to synthetic .txt file (default: repo's synthetic_maintenance_log.txt).",
    )
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate/overwrite the synthetic file before running.")
    parser.add_argument("--entries", type=int, default=250, help="Synthetic entries to generate (only if generating).")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic RNG seed (only if generating).")
    parser.add_argument("--asset", default="C-201", help="Asset tag for synthetic doc (only if generating).")
    args = parser.parse_args()

    MODEL = args.model
    QUESTION = args.question
    # Auto-detect cloud mode from model name if not explicitly set
    # args.cloud will be None if neither --cloud nor --no-cloud was used
    USE_CLOUD = args.cloud if args.cloud is not None else MODEL.endswith('-cloud')
    API_KEY = args.api_key

    if USE_CLOUD:
        print(f"[INFO] Using Ollama Cloud with model: {MODEL}")
        if not API_KEY:
            API_KEY = os.environ.get('OLLAMA_API_KEY')
            if not API_KEY:
                print("ERROR: OLLAMA_API_KEY environment variable not set and --api-key not provided.")
                print("Set it with: set OLLAMA_API_KEY=your_key (Windows) or export OLLAMA_API_KEY=your_key (Linux/Mac)")
                sys.exit(1)
    else:
        print(f"[INFO] Using local Ollama server with model: {MODEL}")

    if args.doc_file:
        doc_path = Path(args.doc_file)
        doc = doc_path.read_text(encoding="utf-8")
    else:
        syn_path = Path(args.synthetic_file)
        if args.generate_synthetic or not syn_path.exists():
            cfg = SyntheticDocConfig(asset=args.asset, num_entries=args.entries, seed=args.seed)
            write_synthetic_doc(str(syn_path), cfg)
        doc = syn_path.read_text(encoding="utf-8")

    print("\n--- RLM-style answer (Ollama) ---\n")
    try:
        answer = rlm_answer_question(
            MODEL,
            doc,
            QUESTION,
            timeout_s=args.timeout,
            num_predict=args.num_predict,
            window=args.window,
            max_spans=args.max_spans,
            fallback_chunk_chars=args.fallback_chunk_chars,
            fallback_overlap=args.fallback_overlap,
            use_cloud=USE_CLOUD,
            api_key=API_KEY,
        )
        if answer:
            # Clean up and print answer with better formatting
            cleaned = format_output(answer.strip())
            print(cleaned)
            print()  # Extra blank line for readability
        else:
            print("ERROR: Function returned empty answer.")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Notes ---")
    print("1) The large document never gets fully stuffed into the model.")
    print("2) The model proposes a plan (keywords), Python searches, then the model summarizes spans recursively.")
    print("3) Final step merges summaries with chunk IDs as evidence.")

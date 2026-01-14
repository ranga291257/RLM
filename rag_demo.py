from __future__ import annotations

# Suppress warnings BEFORE any imports that might load NumPy
import sys
import warnings
import os
print("SCRIPT STARTED", flush=True)
# Only suppress specific NumPy warnings, not all warnings
warnings.filterwarnings("ignore", message=".*MINGW-W64.*")
warnings.filterwarnings("ignore", message=".*CRASHES ARE TO BE EXPECTED.*")

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

# Fix Windows console encoding early
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    # Force flush after reconfigure
    sys.stdout.flush()
    sys.stderr.flush()

# Now import everything else
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np

RAG_AVAILABLE = True

try:
    from ollama import Client
    OLLAMA_CLIENT_AVAILABLE = True
except ImportError:
    OLLAMA_CLIENT_AVAILABLE = False
    Client = Any  # type: ignore[misc,assignment]


# -----------------------------
# RAG Components
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding: Optional[np.ndarray] = None


def embed_with_ollama(
    texts: List[str],
    model: str,
    *,
    timeout_s: int = 300,
    host: str = "http://localhost:11434/api/embeddings",
) -> np.ndarray:
    """
    Create embeddings for a list of texts using Ollama's embeddings endpoint.
    """
    embeddings = []
    for i, txt in enumerate(texts):
        try:
            resp = requests.post(
                host,
                json={"model": model, "prompt": txt},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            emb = data.get("embedding")
            if emb is None:
                raise ValueError(f"No embedding returned for text index {i}")
            embeddings.append(emb)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to get embedding from Ollama for chunk {i}: {exc}"
            ) from exc
    return np.array(embeddings, dtype=np.float32)


class RAGSystem:
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """Initialize RAG system with embedding model (via Ollama embeddings)."""
        if not RAG_AVAILABLE:
            raise ImportError("RAG dependencies not installed.")
        self.embedding_model = embedding_model
        print(f"[RAG] Using Ollama embeddings with model: {embedding_model}")
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def chunk_document(
        self, doc: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[Chunk]:
        """
        Pre-chunk document into fixed-size pieces (RAG approach).
        This happens BEFORE any query.
        """
        if overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        print(f"[RAG] Chunking document (size={chunk_size}, overlap={overlap})...")
        chunks = []
        words = doc.split()
        i = 0
        chunk_num = 0
        step = max(1, chunk_size - overlap)

        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(Chunk(chunk_id=f"RAG_CHUNK_{chunk_num:03d}", text=chunk_text))
            chunk_num += 1
            i += step

        print(f"[RAG] Created {len(chunks)} chunks")
        return chunks

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """
        Create embeddings for all chunks and store them using Ollama embeddings.
        This is the 'indexing' phase in RAG.
        """
        print(f"[RAG] Creating embeddings for {len(chunks)} chunks via Ollama...")
        self.chunks = chunks

        texts = [chunk.text for chunk in chunks]
        embeddings = embed_with_ollama(texts, self.embedding_model)
        self.embeddings = embeddings

        # Store embeddings in chunks
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]

        print(
            "[RAG] Indexing complete. "
            f"Embedding dimension: {self.embeddings.shape[1]}"
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-K most similar chunks using semantic similarity.
        This is the 'retrieval' phase in RAG.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            raise ValueError("Chunks not indexed. Call index_chunks() first.")

        print(f"[RAG] Retrieving top-{top_k} chunks for query...")

        # Embed the query via Ollama
        query_embedding = embed_with_ollama([query], self.embedding_model)[0]

        # Compute cosine similarity manually
        emb_mat = self.embeddings
        query_vec = query_embedding
        emb_norms = np.linalg.norm(emb_mat, axis=1) + 1e-12
        q_norm = np.linalg.norm(query_vec) + 1e-12
        similarities = (emb_mat @ query_vec) / (emb_norms * q_norm)

        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return chunks with similarity scores
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        print(
            "[RAG] Retrieved {count} chunks (similarity range: {min_sim:.3f} - "
            "{max_sim:.3f})".format(
                count=len(results),
                min_sim=float(similarities.min()),
                max_sim=float(similarities.max()),
            )
        )
        return results


# -----------------------------
# Ollama client (for answer generation)
# -----------------------------
OLLAMA_LOCAL_URL = "http://localhost:11434/api/chat"

_cloud_clients: Dict[str, Client] = {}


def _get_cloud_client(api_key: str) -> Client:
    """Get or create a cached cloud client for the given API key."""
    if api_key not in _cloud_clients:
        _cloud_clients[api_key] = Client(
            host="https://ollama.com", headers={"Authorization": f"Bearer {api_key}"}
        )
    return _cloud_clients[api_key]


def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    timeout_s: int = 600,
    num_predict: int = 3000,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> str:
    """Ollama chat client supporting both local and cloud modes."""
    if use_cloud:
        if not OLLAMA_CLIENT_AVAILABLE:
            raise ImportError("ollama package required for cloud mode.")
        if not api_key:
            api_key = os.environ.get("OLLAMA_API_KEY")
            if not api_key:
                raise ValueError("OLLAMA_API_KEY environment variable required.")

        client = _get_cloud_client(api_key)

        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "num_predict": num_predict},
                stream=False,
            )
            if isinstance(response, dict):
                message = response.get("message", {})
                content = message.get("content", "")
                if not content and "thinking" in message:
                    content = message.get("thinking", "")
                done_reason = response.get("done_reason", "")
            else:
                if hasattr(response, "message"):
                    message_obj = response.message
                    content = getattr(message_obj, "content", "") or ""
                    if not content:
                        thinking = getattr(message_obj, "thinking", None)
                        if thinking:
                            content = thinking
                    done_reason = getattr(response, "done_reason", "")
                else:
                    content = ""
                    done_reason = ""

            if not content:
                raise ValueError("Empty response from Ollama Cloud.")
            if done_reason == "length" and content:
                print(f"[WARNING] Response truncated at {num_predict} tokens.")
            return content
        except Exception as exc:
            raise RuntimeError(f"Ollama Cloud API error: {exc}") from exc
    else:
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": num_predict},
            "stream": False,
        }
        try:
            response = requests.post(
                OLLAMA_LOCAL_URL, json=payload, timeout=(10, timeout_s)
            )
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response from Ollama.")
            return content
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"Could not connect to Ollama at {OLLAMA_LOCAL_URL}."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama API error: {exc}") from exc


# -----------------------------
# RAG Pipeline
# -----------------------------
def rag_answer_question(
    rag: RAGSystem,
    question: str,
    model: str,
    *,
    top_k: int = 10,
    timeout_s: int = 600,
    num_predict: int = 3000,
    use_cloud: bool = False,
    api_key: Optional[str] = None,
) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    RAG-style answer: retrieve relevant chunks, then generate answer.
    Returns: (answer, retrieved_chunks_with_scores)
    """
    print(f"\n[RAG] Step 1: Retrieving top-{top_k} relevant chunks...")
    retrieved = rag.retrieve(question, top_k=top_k)

    print(
        f"[RAG] Step 2: Generating answer from {len(retrieved)} retrieved chunks..."
    )

    # Build context from retrieved chunks
    context_parts = []
    for chunk, score in retrieved:
        context_parts.append(
            f"=== {chunk.chunk_id} (similarity: {score:.3f}) ===\n{chunk.text}"
        )

    context = "\n\n".join(context_parts)

    system = (
        "You are analyzing maintenance log data. Answer the question based ONLY on "
        "the provided context. Do not invent facts. When you state a claim, include "
        "the CHUNK_ID(s) that support it. Use clear, concise formatting with bullet "
        "points."
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT (Retrieved chunks):\n{context}\n\n"
        "OUTPUT FORMAT:\n"
        "Use clear sections with bullet points:\n"
        "- Group recurring problems by type\n"
        "- List evidence for each problem\n"
        "- List recommended actions\n"
        "- Include CHUNK_ID references (e.g., RAG_CHUNK_001, RAG_CHUNK_005)\n"
    )

    answer = ollama_chat(
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        timeout_s=timeout_s,
        num_predict=num_predict,
        use_cloud=use_cloud,
        api_key=api_key,
    )

    return answer, retrieved


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG-style demo for comparison with RLM."
    )
    parser.add_argument(
        "--model", default="llama3.2:latest", help="Ollama model name (local model)."
    )
    parser.add_argument(
        "--cloud", action="store_true", default=None, help="Force use of Ollama Cloud."
    )
    parser.add_argument(
        "--no-cloud",
        dest="cloud",
        action="store_false",
        help="Force use of local Ollama server.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Ollama Cloud API key (or set OLLAMA_API_KEY env var).",
    )
    parser.add_argument(
        "--question",
        default=(
            "What are the recurring problems, what evidence supports them, and what "
            "actions are recommended?"
        ),
        help="Question to answer.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-call Ollama read timeout in seconds.",
    )
    parser.add_argument(
        "--num-predict", type=int, default=3000, help="Max tokens to generate."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of chunks to retrieve (RAG parameter).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in words (RAG parameter).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in words (RAG parameter).",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Ollama embedding model (local).",
    )
    parser.add_argument(
        "--doc-file", default=None, help="Path to a text file to analyze."
    )
    parser.add_argument(
        "--synthetic-file",
        default=r"d:\poc\bespoke\synthetic_maintenance_log.txt",
        help="Path to synthetic .txt file.",
    )
    parser.add_argument(
        "--output-file",
        default=r"d:\poc\bespoke\rag_analysis_output.md",
        help="Output markdown file.",
    )

    args = parser.parse_args()

    if not RAG_AVAILABLE:
        print("ERROR: RAG dependencies not installed.")
        print("Install with: pip install sentence-transformers scikit-learn")
        sys.exit(1)

    model_name = args.model
    question = args.question
    # Default to local unless explicitly set to cloud or model name ends with '-cloud'
    if args.cloud is not None:
        use_cloud = args.cloud
    else:
        use_cloud = model_name.endswith("-cloud")  # Auto-detect from model name
    api_key = args.api_key

    if use_cloud:
        print(f"[INFO] Using Ollama Cloud with model: {model_name}")
        if not api_key:
            api_key = os.environ.get("OLLAMA_API_KEY")
            if not api_key:
                print("ERROR: OLLAMA_API_KEY environment variable not set.")
                sys.exit(1)
    else:
        print(f"[INFO] Using local Ollama server with model: {model_name}")

    # Load document
    if args.doc_file:
        doc_path = Path(args.doc_file)
        doc = doc_path.read_text(encoding="utf-8")
    else:
        syn_path = Path(args.synthetic_file)
        if not syn_path.exists():
            print(f"ERROR: File not found: {syn_path}")
            sys.exit(1)
        doc = syn_path.read_text(encoding="utf-8")

    print(f"\n[RAG] Document loaded: {len(doc)} characters, {len(doc.split())} words")

    # Initialize RAG system
    print(
        f"\n[RAG] Initializing RAG system with embedding model: {args.embedding_model}"
    )
    try:
        rag = RAGSystem(embedding_model=args.embedding_model)
    except Exception as exc:
        print(f"ERROR: Failed to initialize RAG system: {exc}")
        sys.exit(1)

    # Pre-process: chunk and index (this is the RAG way)
    chunks = rag.chunk_document(
        doc, chunk_size=args.chunk_size, overlap=args.chunk_overlap
    )
    rag.index_chunks(chunks)

    # Query: retrieve and generate
    print("\n--- RAG-style answer ---\n")
    try:
        answer, retrieved = rag_answer_question(
            rag,
            question,
            model_name,
            top_k=args.top_k,
            timeout_s=args.timeout,
            num_predict=args.num_predict,
            use_cloud=use_cloud,
            api_key=api_key,
        )

        # Create markdown output
        output_lines = [
            "# RAG Analysis Output: Recurring Problems in Maintenance Log",
            "",
            "## Analysis Results",
            "",
            f"**Question:** {question}",
            "",
            "---",
            "",
            "## RAG Approach Details",
            "",
            f"- **Total chunks created:** {len(chunks)}",
            f"- **Chunks retrieved:** {len(retrieved)}",
            f"- **Chunk size:** {args.chunk_size} words",
            f"- **Chunk overlap:** {args.chunk_overlap} words",
            f"- **Embedding model:** {args.embedding_model}",
            "",
            "### Retrieved Chunks (with similarity scores):",
            "",
        ]

        for chunk, score in retrieved:
            output_lines.append(
                f"- **{chunk.chunk_id}** (similarity: {score:.3f})"
            )

        output_lines.extend(
            [
                "",
                "---",
                "",
                "## Answer",
                "",
                answer,
                "",
                "---",
                "",
                "## Notes",
                "",
                "1. RAG pre-chunks the document into fixed-size pieces.",
                "2. Creates vector embeddings for all chunks.",
                "3. Retrieves top-K most similar chunks using semantic search.",
                "4. Generates answer from retrieved chunks only.",
                "",
                "*Generated by RAG-style demo*",
            ]
        )

        output_content = "\n".join(output_lines)

        # Write to file
        output_path = Path(args.output_file)
        output_path.write_text(output_content, encoding="utf-8")
        print(f"\n[RAG] Output written to: {output_path}")

        # Also print to console
        print("\n" + "=" * 80)
        print(answer)
        print("=" * 80)

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
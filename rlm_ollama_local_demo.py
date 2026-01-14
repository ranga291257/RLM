"""
Local-only RLM demo (no cloud calls).

This script reuses the RLM pipeline implemented in `rlm_ollama_demo.py`, but:
- defaults to a local Ollama model (`llama3.2:3b`)
- removes all cloud-related flags and logic
- reads the repo's `synthetic_maintenance_log.txt` by default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from synthetic_data import SyntheticDocConfig, write_synthetic_doc

# Reuse the actual RLM pipeline implementation.
from rlm_ollama_demo import format_output, rlm_answer_question


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RLM-style Ollama demo (LOCAL only)."
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Local Ollama model name (e.g. llama3.2:3b).",
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
        "--num-predict",
        type=int,
        default=3000,
        help="Max tokens to generate per call.",
    )
    parser.add_argument(
        "--max-spans",
        type=int,
        default=8,
        help="Max retrieved spans to summarize (lower = faster).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=600,
        help="Chars around each keyword hit to extract (lower = faster).",
    )
    parser.add_argument(
        "--fallback-chunk-chars",
        type=int,
        default=2200,
        help="Chunk size if keyword search returns too little.",
    )
    parser.add_argument(
        "--fallback-overlap",
        type=int,
        default=200,
        help="Chunk overlap if keyword search returns too little.",
    )

    # You must provide either a real doc file, or a synthetic file.
    parser.add_argument(
        "--doc-file",
        default=None,
        help="Path to a text file to analyze.",
    )
    parser.add_argument(
        "--synthetic-file",
        default=str(PROJECT_ROOT / "synthetic_maintenance_log.txt"),
        help="Path to synthetic .txt file (default: repo's synthetic_maintenance_log.txt).",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate/overwrite the synthetic file before running.",
    )
    parser.add_argument(
        "--entries",
        type=int,
        default=250,
        help="Synthetic entries to generate (only if generating).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Synthetic RNG seed (only if generating).",
    )
    parser.add_argument(
        "--asset",
        default="C-201",
        help="Asset tag for synthetic doc (only if generating).",
    )

    args = parser.parse_args()

    model = args.model
    question = args.question

    # Load document text
    if args.doc_file:
        doc_path = Path(args.doc_file)
        doc = doc_path.read_text(encoding="utf-8")
    else:
        syn_path = Path(args.synthetic_file)
        if args.generate_synthetic or not syn_path.exists():
            cfg = SyntheticDocConfig(asset=args.asset, num_entries=args.entries, seed=args.seed)
            write_synthetic_doc(str(syn_path), cfg)
        doc = syn_path.read_text(encoding="utf-8")

    print(f"[INFO] Using local Ollama server with model: {model}")
    print("\n--- RLM-style answer (Ollama LOCAL) ---\n")

    try:
        answer = rlm_answer_question(
            model,
            doc,
            question,
            timeout_s=args.timeout,
            num_predict=args.num_predict,
            window=args.window,
            max_spans=args.max_spans,
            fallback_chunk_chars=args.fallback_chunk_chars,
            fallback_overlap=args.fallback_overlap,
            use_cloud=False,  # hard lock: local only
            api_key=None,
        )
        if answer:
            print(format_output(answer.strip()))
            print()
            return 0

        print("ERROR: Function returned empty answer.")
        return 2
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


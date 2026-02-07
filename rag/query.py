from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a local RAG index.")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.index / "index.faiss"
    metadata_path = args.index / "metadata.jsonl"

    if not index_path.exists() or not metadata_path.exists():
        raise SystemExit("Index files not found. Run rag/ingest.py first.")

    index = faiss.read_index(str(index_path))
    metadata = load_metadata(metadata_path)

    model = SentenceTransformer(args.model)
    query_embedding = model.encode([args.query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, args.top_k)

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        doc = metadata[idx]
        snippet = doc["text"][:300].strip()
        print("=" * 80)
        print(f"Score: {score:.4f}")
        print(f"Title: {doc['title']}")
        if doc.get("source_url"):
            print(f"Source: {doc['source_url']}")
        print(f"PDF: {doc['pdf_url']}")
        print(f"Chunk: {doc['chunk_id']}")
        print(f"Snippet: {snippet}...")


if __name__ == "__main__":
    main()

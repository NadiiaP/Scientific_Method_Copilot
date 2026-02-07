from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import requests
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.utils import ensure_dir, sanitize_filename, stable_id


def read_papers(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("title") or not row.get("pdf_url"):
                continue
            yield {
                "title": row["title"].strip(),
                "pdf_url": row["pdf_url"].strip(),
                "source_url": row.get("source_url", "").strip(),
            }


def download_pdf(pdf_url: str, dest: Path) -> None:
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(cleaned):
            break
    return [chunk for chunk in chunks if chunk]


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest open-access papers into a RAG index.")
    parser.add_argument("--input", type=Path, default=Path("data/papers/papers.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/index"))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-papers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)
    pdf_dir = args.output / "pdfs"
    ensure_dir(pdf_dir)

    model = SentenceTransformer(args.model)

    documents: list[dict[str, str]] = []
    embeddings: list[np.ndarray] = []

    papers = list(read_papers(args.input))
    if args.max_papers:
        papers = papers[: args.max_papers]

    for paper in tqdm(papers, desc="Ingesting papers"):
        paper_id = stable_id(paper["pdf_url"])
        filename = sanitize_filename(paper["title"])
        pdf_path = pdf_dir / f"{filename}-{paper_id}.pdf"
        if not pdf_path.exists():
            download_pdf(paper["pdf_url"], pdf_path)
        text = extract_text(str(pdf_path))
        if not text.strip():
            continue
        chunks = chunk_text(text, args.chunk_size, args.overlap)
        if not chunks:
            continue
        chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        for idx, chunk in enumerate(chunks):
            documents.append(
                {
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "pdf_url": paper["pdf_url"],
                    "source_url": paper["source_url"],
                    "chunk_id": str(idx),
                    "text": chunk,
                }
            )
        embeddings.append(chunk_embeddings)

    if not embeddings:
        raise SystemExit("No embeddings generated. Check your input CSV or PDFs.")

    all_embeddings = np.vstack(embeddings)
    index = build_index(all_embeddings)

    faiss.write_index(index, str(args.output / "index.faiss"))
    with (args.output / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for doc in documents:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with (args.output / "index_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"model": args.model, "total_chunks": len(documents)}, handle, indent=2)


if __name__ == "__main__":
    main()

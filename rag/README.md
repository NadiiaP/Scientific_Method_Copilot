# Open-Access Paper RAG

This folder contains a minimal Retrieval-Augmented Generation (RAG) workflow that indexes
open-access papers (PDFs) and lets you search them locally.

## 1) Prepare your paper list

Create a CSV at `data/papers/papers.csv` with these headers:

```csv
title,pdf_url,source_url
"The Structure of Scientific Revolutions",https://arxiv.org/pdf/2301.00001.pdf,https://arxiv.org/abs/2301.00001
```

`pdf_url` should be a direct PDF link. `source_url` is optional.

## 2) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3) Ingest papers into a vector index

```bash
python rag/ingest.py --input data/papers/papers.csv --output data/index
```

Artifacts are stored under `data/index`:

- `index.faiss` — vector index
- `metadata.jsonl` — chunk metadata
- `index_meta.json` — model + stats

## 4) Query the index

```bash
python rag/query.py "how do scientists test hypotheses" --index data/index
```

The script prints the top matching chunks, including paper metadata and PDF links.

## Notes

- The pipeline uses a local `sentence-transformers` embedding model.
- For larger corpora, consider swapping to a persistent vector database or a
  chunking strategy with section-aware segmentation.

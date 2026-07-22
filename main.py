import json
from pathlib import Path

from pdf_loader import PDFLoader
from chunker import RecursiveChunker
from embedder import Embedder
from searcher import Searcher
from reranker import Reranker

DATA_DIR = Path("Dataset")
CACHE_DIR = Path("cache")
META_PATH = CACHE_DIR / "meta.json"


def dataset_signature(pdf_paths):
    return [
        {
            "name": pdf_path.name,
            "size": pdf_path.stat().st_size,
            "mtime_ns": pdf_path.stat().st_mtime_ns,
        }
        for pdf_path in pdf_paths
    ]


def cache_is_valid(pdf_paths):
    if not (CACHE_DIR / "index.faiss").exists():
        return False
    if not (CACHE_DIR / "chunks.json").exists():
        return False
    if not META_PATH.exists():
        return False

    with open(META_PATH, encoding="utf-8") as f:
        saved = json.load(f)

    return saved.get("files") == dataset_signature(pdf_paths)


def build_index(pdf_paths):
    loader = PDFLoader()
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)

    chunks = []
    for pdf_path in pdf_paths:
        text = loader.load(str(pdf_path))
        doc_chunks = chunker.chunk(text, source=pdf_path.name)
        chunks.extend(doc_chunks)

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i

    print(f"Loaded {len(pdf_paths)} PDFs → {len(chunks)} chunks")

    embedder = Embedder()
    chunk_embeddings = embedder.embed([chunk["text"] for chunk in chunks])
    searcher = Searcher(chunk_embeddings, chunks)

    searcher.save(CACHE_DIR)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"files": dataset_signature(pdf_paths)}, f)

    print(f"Saved index to {CACHE_DIR}/")
    return searcher, embedder


pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
if not pdf_paths:
    raise FileNotFoundError(f"No PDFs found in {DATA_DIR.resolve()}")

if cache_is_valid(pdf_paths):
    print(f"Loading cached index from {CACHE_DIR}/")
    searcher = Searcher.load(CACHE_DIR)
    embedder = Embedder()
    print(f"Loaded {len(searcher.chunks)} chunks from cache")
else:
    print("Cache missing or Dataset changed — building index...")
    searcher, embedder = build_index(pdf_paths)

reranker = Reranker()

query = input("Enter your query: ")
query_embedding = embedder.embed(query)

candidates = searcher.search(query_embedding, top_k=20)
results = reranker.rerank(query, candidates, top_k=3)

print("\nTop Results\n")

for i, result in enumerate(results, 1):
    chunk = result["chunk"]
    print("=" * 50)
    print(f"#{i}")
    print("Chunk ID:", chunk["chunk_id"])
    print("Source:", chunk.get("source"))
    print("Score:", round(result["score"], 4))
    print(chunk["text"])

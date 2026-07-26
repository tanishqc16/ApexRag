import json
from pathlib import Path

from pdf_loader import PDFLoader
from chunker import RecursiveChunker
from embedder import Embedder
from searcher import Searcher
from bm25_searcher import BM25Searcher
from hybrid import reciprocal_rank_fusion
from query_expansion import QueryExpander
from parent_child import map_to_parents
from graph_builder import build_graph, KnowledgeGraph
from graph_retriever import GraphRetriever
from reranker import Reranker

DATA_DIR = Path("Dataset")
CACHE_DIR = Path("cache")
META_PATH = CACHE_DIR / "meta.json"
GRAPH_PATH = CACHE_DIR / "graph.json"
CACHE_SCHEMA = "parent_child_graph_v1"


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
    if not GRAPH_PATH.exists():
        return False
    if not META_PATH.exists():
        return False

    with open(META_PATH, encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("schema") != CACHE_SCHEMA:
        return False

    return saved.get("files") == dataset_signature(pdf_paths)


def build_index(pdf_paths):
    loader = PDFLoader()
    chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50, parent_size=1500)

    chunks = []
    for pdf_path in pdf_paths:
        text = loader.load(str(pdf_path))
        doc_chunks = chunker.chunk_parent_child(text, source=pdf_path.name)
        chunks.extend(doc_chunks)

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = i

    n_parents = len({(c.get("source"), c.get("parent_id")) for c in chunks})
    print(f"Loaded {len(pdf_paths)} PDFs -> {n_parents} parents -> {len(chunks)} children")

    embedder = Embedder()
    chunk_embeddings = embedder.embed([chunk["text"] for chunk in chunks])
    searcher = Searcher(chunk_embeddings, chunks)
    searcher.save(CACHE_DIR)

    graph = build_graph(chunks)
    graph.save(GRAPH_PATH)
    print(f"Built graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"schema": CACHE_SCHEMA, "files": dataset_signature(pdf_paths)},
            f,
        )

    print(f"Saved index + graph to {CACHE_DIR}/")
    return searcher, embedder, graph


def run():
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR.resolve()}")

    if cache_is_valid(pdf_paths):
        print(f"Loading cached index from {CACHE_DIR}/")
        searcher = Searcher.load(CACHE_DIR)
        embedder = Embedder()
        graph = KnowledgeGraph.load(GRAPH_PATH)
        print(
            f"Loaded {len(searcher.chunks)} child chunks, "
            f"{len(graph.nodes)} graph nodes"
        )
    else:
        print("Cache missing, outdated, or Dataset changed — building index...")
        searcher, embedder, graph = build_index(pdf_paths)

    bm25 = BM25Searcher(searcher.chunks)
    graph_retriever = GraphRetriever(graph, searcher.chunks, max_hops=2)
    expander = QueryExpander()
    reranker = Reranker()

    query = input("Enter your query: ")
    expanded = expander.expand(query)

    print(f"\nExpansion ({expanded['method']})")
    print(f"  Rewritten: {expanded['rewritten']}")
    print(f"  Keywords:  {', '.join(expanded['keywords'])}")

    seeds = graph_retriever.find_seed_entities(query)
    print(f"  Graph seeds: {len(seeds)}")

    query_embedding = embedder.embed(expanded["rewritten"])
    dense_hits = searcher.search(query_embedding, top_k=20)
    sparse_hits = bm25.search(expanded["sparse_query"], top_k=20)
    graph_hits = graph_retriever.search(query, top_k=20)

    candidates = reciprocal_rank_fusion(
        [dense_hits, sparse_hits, graph_hits], top_k=20
    )
    child_results = reranker.rerank(query, candidates, top_k=10)
    results = map_to_parents(child_results, top_k=3)

    print("\nTop Results (parent sections)\n")

    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        print("=" * 50)
        print(f"#{i}")
        print("Parent ID:", chunk.get("parent_id"))
        print("Child ID:", chunk.get("chunk_id"))
        print("Source:", chunk.get("source"))
        print("Score:", round(result["score"], 4))
        print(chunk["text"])


if __name__ == "__main__":
    run()

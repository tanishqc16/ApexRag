import csv
from pathlib import Path

from ranx import Qrels, Run, evaluate

from embedder import Embedder
from searcher import Searcher
from bm25_searcher import BM25Searcher
from hybrid import reciprocal_rank_fusion
from query_expansion import QueryExpander
from parent_child import map_to_parents
from graph_builder import KnowledgeGraph
from graph_retriever import GraphRetriever
from reranker import Reranker

CACHE_DIR = Path("cache")
GRAPH_PATH = CACHE_DIR / "graph.json"
EVAL_PATH = Path("eval_questions.csv")
CANDIDATE_K = 20
TOP_K = 3

METRICS = [
    "hit_rate@1",
    "hit_rate@3",
    "mrr",
    "precision@3",
    "recall@3",
    "ndcg@3",
]


def load_questions(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def results_to_run_scores(results):
    """Map retrieved chunks to doc-level scores (PDF source)."""
    scores = {}
    for result in results:
        source = result["chunk"].get("source")
        if not source:
            continue
        score = float(result["score"])
        if source not in scores or score > scores[source]:
            scores[source] = score
    return scores


def is_pct_metric(metric):
    return (
        metric.startswith("hit_rate")
        or metric.startswith("precision")
        or metric.startswith("recall")
    )


def format_val(metric, value):
    if is_pct_metric(metric):
        return f"{value:>9.2%}"
    return f"{value:>10.3f}"


def format_delta(metric, value):
    if is_pct_metric(metric):
        return f"{value:>+9.2%}"
    return f"{value:>+10.3f}"


def print_report(faiss_scores, rerank_scores, hybrid_scores, expand_scores, parent_scores, graph_scores):
    print("\n=== ranx Evaluation Summary ===\n")
    print("Delta = GraphRAG - FAISS (best vs baseline).\n")
    header = (
        f"{'Metric':<14} "
        f"{'FAISS':>10} "
        f"{'Reranking':>10} "
        f"{'Hybrid':>10} "
        f"{'QueryExp':>10} "
        f"{'ParentChild':>12} "
        f"{'GraphRAG':>10} "
        f"{'Delta':>10}"
    )
    print(header)
    print("-" * len(header))

    for metric in METRICS:
        faiss_val = float(faiss_scores[metric])
        rerank_val = float(rerank_scores[metric])
        hybrid_val = float(hybrid_scores[metric])
        expand_val = float(expand_scores[metric])
        parent_val = float(parent_scores[metric])
        graph_val = float(graph_scores[metric])
        print(
            f"{metric:<14} "
            f"{format_val(metric, faiss_val)} "
            f"{format_val(metric, rerank_val)} "
            f"{format_val(metric, hybrid_val)} "
            f"{format_val(metric, expand_val)} "
            f"{format_val(metric, parent_val)} "
            f"{format_val(metric, graph_val)} "
            f"{format_delta(metric, graph_val - faiss_val)}"
        )


def main():
    if not (CACHE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "No cache found. Run python main.py once to build the index."
        )
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            "No graph found. Re-run python main.py to build cache + graph."
        )

    questions = load_questions(EVAL_PATH)
    if not questions:
        raise ValueError(f"No questions found in {EVAL_PATH}")

    print(f"Loading cache from {CACHE_DIR}/")
    searcher = Searcher.load(CACHE_DIR)
    if searcher.chunks and "parent_id" not in searcher.chunks[0]:
        raise ValueError(
            "Cache has no parent-child metadata. Re-run python main.py to rebuild."
        )

    graph = KnowledgeGraph.load(GRAPH_PATH)
    graph_retriever = GraphRetriever(graph, searcher.chunks, max_hops=2)
    bm25 = BM25Searcher(searcher.chunks)
    embedder = Embedder()
    expander = QueryExpander()
    reranker = Reranker()

    print(
        f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
    )
    print(f"Evaluating {len(questions)} questions with ranx...\n")

    qrels_dict = {}
    faiss_run_dict = {}
    rerank_run_dict = {}
    hybrid_run_dict = {}
    expand_run_dict = {}
    parent_run_dict = {}
    graph_run_dict = {}

    for i, row in enumerate(questions):
        qid = f"q{i}"
        query = row["query"].strip()
        expected = row["expected_source"].strip()
        qtype = (row.get("type") or "single_hop").strip()

        query_embedding = embedder.embed(query)
        dense_hits = searcher.search(query_embedding, top_k=CANDIDATE_K)
        sparse_hits = bm25.search(query, top_k=CANDIDATE_K)

        faiss_top = dense_hits[:TOP_K]
        reranked = reranker.rerank(query, dense_hits, top_k=TOP_K)

        hybrid_candidates = reciprocal_rank_fusion(
            [dense_hits, sparse_hits], top_k=CANDIDATE_K
        )
        hybrid_reranked = reranker.rerank(query, hybrid_candidates, top_k=TOP_K)

        expanded = expander.expand(query)
        expand_dense = searcher.search(
            embedder.embed(expanded["rewritten"]), top_k=CANDIDATE_K
        )
        expand_sparse = bm25.search(expanded["sparse_query"], top_k=CANDIDATE_K)
        expand_candidates = reciprocal_rank_fusion(
            [expand_dense, expand_sparse], top_k=CANDIDATE_K
        )
        expand_reranked = reranker.rerank(query, expand_candidates, top_k=TOP_K)

        parent_child_hits = reranker.rerank(query, expand_candidates, top_k=10)
        parent_results = map_to_parents(parent_child_hits, top_k=TOP_K)

        graph_hits = graph_retriever.search(query, top_k=CANDIDATE_K)
        graph_candidates = reciprocal_rank_fusion(
            [expand_dense, expand_sparse, graph_hits], top_k=CANDIDATE_K
        )
        graph_reranked_children = reranker.rerank(query, graph_candidates, top_k=10)
        graph_results = map_to_parents(graph_reranked_children, top_k=TOP_K)

        qrels_dict[qid] = {expected: 1}
        faiss_run_dict[qid] = results_to_run_scores(faiss_top)
        rerank_run_dict[qid] = results_to_run_scores(reranked)
        hybrid_run_dict[qid] = results_to_run_scores(hybrid_reranked)
        expand_run_dict[qid] = results_to_run_scores(expand_reranked)
        parent_run_dict[qid] = results_to_run_scores(parent_results)
        graph_run_dict[qid] = results_to_run_scores(graph_results)

        print(f"Q: {query}")
        print(f"   Type:         {qtype}")
        print(f"   Expanded:     {expanded['rewritten']}")
        print(f"   Expected:     {expected}")
        print(f"   FAISS:        {list(faiss_run_dict[qid].keys())}")
        print(f"   Reranking:    {list(rerank_run_dict[qid].keys())}")
        print(f"   Hybrid:       {list(hybrid_run_dict[qid].keys())}")
        print(f"   QueryExp:     {list(expand_run_dict[qid].keys())}")
        print(f"   ParentChild:  {list(parent_run_dict[qid].keys())}")
        print(f"   GraphRAG:     {list(graph_run_dict[qid].keys())}")
        print()

    qrels = Qrels(qrels_dict)
    faiss_run = Run(faiss_run_dict, name="FAISS")
    rerank_run = Run(rerank_run_dict, name="Reranking")
    hybrid_run = Run(hybrid_run_dict, name="Hybrid")
    expand_run = Run(expand_run_dict, name="QueryExpansion")
    parent_run = Run(parent_run_dict, name="ParentChild")
    graph_run = Run(graph_run_dict, name="GraphRAG")

    faiss_scores = evaluate(qrels, faiss_run, METRICS)
    rerank_scores = evaluate(qrels, rerank_run, METRICS)
    hybrid_scores = evaluate(qrels, hybrid_run, METRICS)
    expand_scores = evaluate(qrels, expand_run, METRICS)
    parent_scores = evaluate(qrels, parent_run, METRICS)
    graph_scores = evaluate(qrels, graph_run, METRICS)

    print_report(
        faiss_scores,
        rerank_scores,
        hybrid_scores,
        expand_scores,
        parent_scores,
        graph_scores,
    )


if __name__ == "__main__":
    main()

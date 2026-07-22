import csv
from pathlib import Path

from ranx import Qrels, Run, evaluate

from embedder import Embedder
from searcher import Searcher
from reranker import Reranker

CACHE_DIR = Path("cache")
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


def print_report(faiss_scores, rerank_scores):
    print("\n=== ranx Evaluation Summary ===\n")
    header = f"{'Metric':<14} {'FAISS':>10} {'Rerank':>10} {'Delta':>10}"
    print(header)
    print("-" * len(header))

    for metric in METRICS:
        faiss_val = float(faiss_scores[metric])
        rerank_val = float(rerank_scores[metric])
        delta = rerank_val - faiss_val
        if metric.startswith("hit_rate") or metric.startswith("precision") or metric.startswith("recall"):
            print(
                f"{metric:<14} {faiss_val:>9.2%} {rerank_val:>9.2%} {delta:>+9.2%}"
            )
        else:
            print(
                f"{metric:<14} {faiss_val:>10.3f} {rerank_val:>10.3f} {delta:>+10.3f}"
            )


def main():
    if not (CACHE_DIR / "index.faiss").exists():
        raise FileNotFoundError(
            "No cache found. Run python main.py once to build the index."
        )

    questions = load_questions(EVAL_PATH)
    if not questions:
        raise ValueError(f"No questions found in {EVAL_PATH}")

    print(f"Loading cache from {CACHE_DIR}/")
    searcher = Searcher.load(CACHE_DIR)
    embedder = Embedder()
    reranker = Reranker()

    print(f"Evaluating {len(questions)} questions with ranx...\n")

    qrels_dict = {}
    faiss_run_dict = {}
    rerank_run_dict = {}

    for i, row in enumerate(questions):
        qid = f"q{i}"
        query = row["query"].strip()
        expected = row["expected_source"].strip()

        query_embedding = embedder.embed(query)
        candidates = searcher.search(query_embedding, top_k=CANDIDATE_K)
        faiss_top = candidates[:TOP_K]
        reranked = reranker.rerank(query, candidates, top_k=TOP_K)

        qrels_dict[qid] = {expected: 1}
        faiss_run_dict[qid] = results_to_run_scores(faiss_top)
        rerank_run_dict[qid] = results_to_run_scores(reranked)

        print(f"Q: {query}")
        print(f"   Expected: {expected}")
        print(f"   FAISS:    {list(faiss_run_dict[qid].keys())}")
        print(f"   Rerank:   {list(rerank_run_dict[qid].keys())}")
        print()

    qrels = Qrels(qrels_dict)
    faiss_run = Run(faiss_run_dict, name="FAISS")
    rerank_run = Run(rerank_run_dict, name="FAISS+Rerank")

    faiss_scores = evaluate(qrels, faiss_run, METRICS)
    rerank_scores = evaluate(qrels, rerank_run, METRICS)

    print_report(faiss_scores, rerank_scores)


if __name__ == "__main__":
    main()

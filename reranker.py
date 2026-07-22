from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, results, top_k=3):
        if not results:
            return []

        pairs = [(query, result["chunk"]["text"]) for result in results]
        scores = self.model.predict(pairs)

        reranked = []
        for result, score in zip(results, scores):
            item = dict(result)
            item["retrieval_score"] = result["score"]
            item["score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

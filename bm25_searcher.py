import re

from rank_bm25 import BM25Okapi


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Searcher:

    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_k=20):
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results = []
        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue
            results.append({
                "rank": int(index),
                "score": float(scores[index]),
                "chunk": self.chunks[index],
            })

        return results

import json
from pathlib import Path

import faiss
import numpy as np


class Searcher:

    def __init__(self, chunk_embeddings, chunks):
        self.chunks = chunks
        vectors = self._to_numpy(chunk_embeddings)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        vectors = self._normalize(vectors)

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)

    @classmethod
    def from_index(cls, index, chunks):
        searcher = cls.__new__(cls)
        searcher.index = index
        searcher.chunks = chunks
        return searcher

    def search(self, query_embedding, top_k=3):
        query = self._to_numpy(query_embedding)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = self._normalize(query)

        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append({
                "rank": int(index),
                "score": float(score),
                "chunk": self.chunks[int(index)],
            })

        return results

    def save(self, cache_dir):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(cache_dir / "index.faiss"))
        with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)

    @classmethod
    def load(cls, cache_dir):
        cache_dir = Path(cache_dir)
        index = faiss.read_index(str(cache_dir / "index.faiss"))
        with open(cache_dir / "chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        return cls.from_index(index, chunks)

    def _to_numpy(self, embeddings):
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        return np.asarray(embeddings, dtype=np.float32)

    def _normalize(self, vectors):
        faiss.normalize_L2(vectors)
        return vectors

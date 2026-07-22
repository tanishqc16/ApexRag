from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def embed(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)

        return embeddings
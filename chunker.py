import re


class RecursiveChunker:

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text, source=None):
        raw_chunks = []

        for paragraph in self._split_paragraphs(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) <= self.chunk_size:
                raw_chunks.append(paragraph)
            else:
                raw_chunks.extend(self._split_by_sentences(paragraph))

        overlapped = self._add_word_overlap(raw_chunks)
        return self._with_metadata(overlapped, source=source)

    def _split_paragraphs(self, text):
        return re.split(r"\n\s*\n", text)

    def _split_by_sentences(self, paragraph):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        return self._pack(sentences, on_overflow=self._split_by_words)

    def _split_by_words(self, sentence):
        words = sentence.split()
        return self._pack(words, on_overflow=self._split_by_chars)

    def _split_by_chars(self, word):
        return [
            word[i:i + self.chunk_size]
            for i in range(0, len(word), self.chunk_size)
        ]

    def _pack(self, pieces, on_overflow):
        chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if len(piece) > self.chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(on_overflow(piece))
                continue

            candidate = f"{current} {piece}".strip() if current else piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece

        if current:
            chunks.append(current)

        return chunks

    def _add_word_overlap(self, chunks):
        if not chunks or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            previous_words = chunks[i - 1].split()
            overlap_words = previous_words[-self.chunk_overlap:]
            overlap_text = " ".join(overlap_words)
            overlapped.append(f"{overlap_text} {chunks[i]}".strip())

        return overlapped

    def _with_metadata(self, chunks, source=None):
        result = []

        for i, text in enumerate(chunks):
            item = {
                "chunk_id": i,
                "text": text,
                "char_count": len(text),
            }
            if source is not None:
                item["source"] = source
            result.append(item)

        return result

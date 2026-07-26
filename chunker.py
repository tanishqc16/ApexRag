import re


class RecursiveChunker:

    def __init__(self, chunk_size=500, chunk_overlap=50, parent_size=1500):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.parent_size = parent_size

    def chunk(self, text, source=None):
        """Flat recursive chunks (no parent links)."""
        raw_chunks = self._split_to_size(text, self.chunk_size)
        overlapped = self._add_word_overlap(raw_chunks)
        return self._with_metadata(overlapped, source=source)

    def chunk_parent_child(self, text, source=None):
        """
        Build large parents, then small children linked to each parent.
        Children are what we embed/search; parents are what we return.
        """
        parent_texts = self._split_to_size(text, self.parent_size)
        children = []

        for parent_id, parent_text in enumerate(parent_texts):
            child_texts = self._split_to_size(parent_text, self.chunk_size)
            child_texts = self._add_word_overlap(child_texts)

            for child_text in child_texts:
                item = {
                    "text": child_text,
                    "char_count": len(child_text),
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "parent_char_count": len(parent_text),
                }
                if source is not None:
                    item["source"] = source
                children.append(item)

        return children

    def _split_to_size(self, text, size):
        raw_chunks = []
        for paragraph in self._split_paragraphs(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= size:
                raw_chunks.append(paragraph)
            else:
                raw_chunks.extend(self._split_by_sentences(paragraph, size))
        return raw_chunks

    def _split_paragraphs(self, text):
        return re.split(r"\n\s*\n", text)

    def _split_by_sentences(self, paragraph, size):
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        return self._pack(
            sentences,
            size=size,
            on_overflow=lambda s: self._split_by_words(s, size),
        )

    def _split_by_words(self, sentence, size):
        words = sentence.split()
        return self._pack(
            words,
            size=size,
            on_overflow=lambda w: self._split_by_chars(w, size),
        )

    def _split_by_chars(self, word, size):
        return [word[i:i + size] for i in range(0, len(word), size)]

    def _pack(self, pieces, size, on_overflow):
        chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if len(piece) > size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(on_overflow(piece))
                continue

            candidate = f"{current} {piece}".strip() if current else piece

            if len(candidate) <= size:
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

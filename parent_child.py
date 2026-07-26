def map_to_parents(results, top_k=3):
    """
    Map ranked child hits to unique parent sections.
    Keeps first (best) child score for each parent.
    """
    seen = set()
    mapped = []

    for result in results:
        chunk = result["chunk"]
        parent_id = chunk.get("parent_id")
        source = chunk.get("source")
        key = (source, parent_id)

        if key in seen:
            continue
        seen.add(key)

        parent_text = chunk.get("parent_text") or chunk["text"]
        parent_chunk = {
            "chunk_id": chunk.get("chunk_id"),
            "parent_id": parent_id,
            "text": parent_text,
            "char_count": len(parent_text),
            "child_text": chunk["text"],
            "source": source,
        }

        item = dict(result)
        item["chunk"] = parent_chunk
        mapped.append(item)

        if len(mapped) >= top_k:
            break

    return mapped

def reciprocal_rank_fusion(result_lists, top_k=20, k=60):
    """Merge ranked lists with Reciprocal Rank Fusion (RRF)."""
    fused = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk"]["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "rank": chunk_id,
                    "score": 0.0,
                    "chunk": item["chunk"],
                }
            fused[chunk_id]["score"] += 1.0 / (k + rank)

    merged = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k]

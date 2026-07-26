from collections import defaultdict, deque

from graph_builder import PRODUCTS, PROMOS, PLACES, _node_id, adjacency


class GraphRetriever:
    """Retrieve chunks by walking the knowledge graph from query entities."""

    def __init__(self, graph, chunks, max_hops=2):
        self.graph = graph
        self.chunks = chunks
        self.chunk_by_id = {c["chunk_id"]: c for c in chunks}
        self.max_hops = max_hops
        self.adj = adjacency(graph)

    def find_seed_entities(self, query):
        seeds = []
        lower = query.lower()

        for name in PRODUCTS:
            if name.lower() in lower:
                seeds.append(_node_id("product", name))
        for name in PROMOS:
            if name.lower() in lower:
                seeds.append(_node_id("promo", name))
        for name in PLACES:
            if name.lower() in lower:
                seeds.append(_node_id("place", name))

        # Policy keywords in the query.
        policy_keywords = {
            "warranty": "Warranty Policy",
            "upgrade": "Software Upgrade Policy",
            "software": "Software Upgrade Policy",
            "terms": "Terms and Conditions",
            "environmental": "Environmental Policy",
            "service": "Servicing Details",
            "servicing": "Servicing Details",
            "store": "Store Locations",
            "promo": "Promotional Offers",
            "promotional": "Promotional Offers",
            "offer": "Promotional Offers",
            "corporate": "Corporate Booking",
            "bundle": "Promo Product Eligibility",
            "kit": "Promo Product Eligibility",
        }
        for keyword, policy_name in policy_keywords.items():
            if keyword in lower:
                seeds.append(_node_id("policy", policy_name))

        # Dedupe while preserving order.
        seen = set()
        unique = []
        for seed in seeds:
            if seed in self.graph.nodes and seed not in seen:
                seen.add(seed)
                unique.append(seed)
        return unique

    def search(self, query, top_k=20):
        seeds = self.find_seed_entities(query)
        if not seeds:
            return []

        chunk_scores = defaultdict(float)
        visited_nodes = {}
        queue = deque()

        for seed in seeds:
            queue.append((seed, 0))
            visited_nodes[seed] = 0

        while queue:
            node_id, hops = queue.popleft()
            node = self.graph.nodes.get(node_id)
            if not node:
                continue

            # Closer hops get higher weight.
            weight = 1.0 / (hops + 1)
            for chunk_id in node.get("chunk_ids", []):
                chunk_scores[chunk_id] += weight

            if hops >= self.max_hops:
                continue

            for edge in self.adj.get(node_id, []):
                for chunk_id in edge.get("chunk_ids", []):
                    chunk_scores[chunk_id] += weight * 0.8

                neighbor = edge["target"]
                next_hops = hops + 1
                if neighbor not in visited_nodes or next_hops < visited_nodes[neighbor]:
                    visited_nodes[neighbor] = next_hops
                    queue.append((neighbor, next_hops))

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for chunk_id, score in ranked[:top_k]:
            chunk = self.chunk_by_id.get(chunk_id)
            if not chunk:
                continue
            results.append(
                {
                    "rank": chunk_id,
                    "score": float(score),
                    "chunk": chunk,
                }
            )
        return results

import json
import re
from collections import defaultdict
from pathlib import Path


# Longer names first so "Velvera One Pro" matches before "Velvera One".
PRODUCTS = [
    "Velvera One Ultra",
    "Velvera One Pro",
    "Velvera One",
    "Velvera Edge Pro",
    "Velvera Edge",
    "Velvera Nexus 1 Pro",
    "Velvera Nexus 1",
    "Velvera Nexus",
    "Velvera Core Max",
    "Velvera Core",
    "Velvera Start",
    "Velvera Pad Pro",
    "Velvera Pad",
    "Velvera Watch Pro",
    "Velvera Watch SE",
    "Velvera Watch",
    "Velvera Buds Premium",
    "Velvera Buds Deluxe",
    "Velvera Buds",
    "Velvera Headphones Premium",
    "Velvera Headphones Deluxe",
    "Velvera Headphones",
]

PROMOS = [
    "Creator Pro Pack",
    "Ultimate Power Kit",
    "AI Productivity Suite",
    "Family Connect Pack",
    "Student Essential Kit",
    "Young Pro Starter",
]

PLACES = [
    "Bengaluru",
    "Mumbai",
    "Hyderabad",
    "Delhi",
    "Kochi",
    "Ahmedabad",
    "Kolkata",
    "Gurugram",
    "Chandigarh",
    "Andheri",
    "Koramangala",
    "Connaught Place",
    "Jubilee Hills",
]

POLICY_DOCS = {
    "warranty_and_service.pdf": "Warranty Policy",
    "software_upgrade_policy.pdf": "Software Upgrade Policy",
    "terms_and_conditions.pdf": "Terms and Conditions",
    "environmental_policy.pdf": "Environmental Policy",
    "servicing_details.pdf": "Servicing Details",
    "store_locations.pdf": "Store Locations",
    "promotional_offers.pdf": "Promotional Offers",
    "corporate_booking.pdf": "Corporate Booking",
    "product_policy_links.pdf": "Product Policy Links",
    "product_store_service_map.pdf": "Product Store Service Map",
    "promo_product_eligibility.pdf": "Promo Product Eligibility",
}


def _node_id(entity_type, name):
    return f"{entity_type}:{name.lower()}"


def _find_matches(text, names):
    lower = text.lower()
    found = []
    for name in names:
        if name.lower() in lower:
            found.append(name)
    return found


def _infer_relation(text):
    lower = text.lower()
    if any(w in lower for w in ("warranty", "covered", "care+", "guarantee")):
        return "covered_by"
    if any(w in lower for w in ("store", "sold", "demo", "available at", "purchase")):
        return "sold_at"
    if any(w in lower for w in ("service", "repair", "servicing", "pickup")):
        return "serviced_at"
    if any(w in lower for w in ("bundle", "pack", "includes", "kit", "promo")):
        return "includes"
    if any(w in lower for w in ("upgrade", "software", "os update")):
        return "upgraded_by"
    return "related_to"


class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._edge_keys = set()

    def add_node(self, entity_type, name, chunk_id=None, source=None):
        nid = _node_id(entity_type, name)
        if nid not in self.nodes:
            self.nodes[nid] = {
                "id": nid,
                "name": name,
                "type": entity_type,
                "chunk_ids": [],
                "sources": [],
            }
        node = self.nodes[nid]
        if chunk_id is not None and chunk_id not in node["chunk_ids"]:
            node["chunk_ids"].append(chunk_id)
        if source and source not in node["sources"]:
            node["sources"].append(source)
        return nid

    def add_edge(self, source_id, target_id, relation, chunk_id=None):
        if source_id == target_id:
            return
        key = (source_id, target_id, relation)
        if key in self._edge_keys:
            for edge in self.edges:
                if (
                    edge["source"] == source_id
                    and edge["target"] == target_id
                    and edge["relation"] == relation
                ):
                    if chunk_id is not None and chunk_id not in edge["chunk_ids"]:
                        edge["chunk_ids"].append(chunk_id)
                    return
        self._edge_keys.add(key)
        self.edges.append(
            {
                "source": source_id,
                "target": target_id,
                "relation": relation,
                "chunk_ids": [chunk_id] if chunk_id is not None else [],
            }
        )

    def to_dict(self):
        return {"nodes": self.nodes, "edges": self.edges}

    @classmethod
    def from_dict(cls, data):
        graph = cls()
        graph.nodes = data.get("nodes", {})
        graph.edges = data.get("edges", [])
        graph._edge_keys = {
            (e["source"], e["target"], e["relation"]) for e in graph.edges
        }
        return graph

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def build_graph(chunks):
    """Rule-based entity/relation extraction over child chunks."""
    graph = KnowledgeGraph()

    for chunk in chunks:
        text = chunk.get("text", "")
        chunk_id = chunk.get("chunk_id")
        source = chunk.get("source")

        product_hits = _find_matches(text, PRODUCTS)
        promo_hits = _find_matches(text, PROMOS)
        place_hits = _find_matches(text, PLACES)

        entity_ids = []
        for name in product_hits:
            entity_ids.append(graph.add_node("product", name, chunk_id, source))
        for name in promo_hits:
            entity_ids.append(graph.add_node("promo", name, chunk_id, source))
        for name in place_hits:
            entity_ids.append(graph.add_node("place", name, chunk_id, source))

        if source in POLICY_DOCS:
            policy_id = graph.add_node(
                "policy", POLICY_DOCS[source], chunk_id, source
            )
            entity_ids.append(policy_id)

        # Link products mentioned in a policy/bridge doc to that policy node.
        if source in POLICY_DOCS:
            policy_id = _node_id("policy", POLICY_DOCS[source])
            for name in product_hits:
                graph.add_edge(
                    _node_id("product", name),
                    policy_id,
                    _infer_relation(text),
                    chunk_id,
                )
            for name in promo_hits:
                graph.add_edge(
                    _node_id("promo", name),
                    policy_id,
                    "documented_in",
                    chunk_id,
                )

        relation = _infer_relation(text)

        # Promo includes products when they co-occur.
        for promo in promo_hits:
            promo_id = _node_id("promo", promo)
            for product in product_hits:
                graph.add_edge(
                    promo_id,
                    _node_id("product", product),
                    "includes",
                    chunk_id,
                )

        # Products related to places.
        for product in product_hits:
            product_id = _node_id("product", product)
            for place in place_hits:
                graph.add_edge(
                    product_id,
                    _node_id("place", place),
                    relation if relation in ("sold_at", "serviced_at") else "sold_at",
                    chunk_id,
                )

        # Co-occurring products get a soft related_to link.
        for i, left in enumerate(product_hits):
            for right in product_hits[i + 1 :]:
                graph.add_edge(
                    _node_id("product", left),
                    _node_id("product", right),
                    "related_to",
                    chunk_id,
                )

        # If warranty language appears with products, link to Warranty Policy.
        if re.search(r"warranty|care\+|24 months|12 months", text, re.I):
            warranty_id = graph.add_node(
                "policy", "Warranty Policy", chunk_id, source
            )
            for product in product_hits:
                graph.add_edge(
                    _node_id("product", product),
                    warranty_id,
                    "covered_by",
                    chunk_id,
                )

    return graph


def adjacency(graph):
    """Undirected adjacency for traversal."""
    adj = defaultdict(list)
    for edge in graph.edges:
        adj[edge["source"]].append(edge)
        adj[edge["target"]].append(
            {
                "source": edge["target"],
                "target": edge["source"],
                "relation": edge["relation"],
                "chunk_ids": edge["chunk_ids"],
            }
        )
    return adj

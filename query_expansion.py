import os
import re

# Domain-oriented synonyms for Velvera-style docs (offline expansion).
SYNONYMS = {
    "serviced": ["service", "repair", "servicing", "warranty"],
    "service": ["servicing", "repair", "support", "fix"],
    "servicing": ["service", "repair", "warranty"],
    "warranty": ["guarantee", "coverage", "service", "repair"],
    "charge": ["charging", "charger", "battery", "power"],
    "charging": ["charge", "charger", "battery"],
    "charger": ["charging", "charge", "power adapter"],
    "phone": ["mobile", "smartphone", "handset"],
    "phones": ["mobile", "smartphone", "handset", "phone"],
    "tablet": ["pad", "tablets"],
    "tablets": ["pad", "tablet"],
    "store": ["stores", "location", "locations", "shop", "outlet"],
    "locations": ["stores", "address", "outlets", "branches"],
    "promotional": ["promo", "offers", "deals", "discount"],
    "offers": ["promotions", "deals", "discounts", "promo"],
    "corporate": ["business", "enterprise", "company"],
    "booking": ["reservation", "book", "schedule"],
    "environmental": ["environment", "sustainability", "eco", "green"],
    "policy": ["policies", "guidelines", "rules"],
    "terms": ["conditions", "agreement", "policies"],
    "conditions": ["terms", "agreement"],
    "upgrade": ["update", "software update", "os"],
    "software": ["os", "firmware", "update"],
    "watch": ["smartwatch", "wearable"],
    "catalog": ["catalogue", "products", "models", "lineup"],
    "device": ["product", "gadget", "hardware"],
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
    "how", "what", "where", "when", "why", "which", "who",
    "i", "my", "me", "we", "our", "you", "your",
    "in", "on", "at", "to", "for", "of", "and", "or", "with",
    "can", "could", "should", "would", "will", "be", "been",
    "get", "got", "getting", "make", "made",
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class QueryExpander:
    """Expand a query with synonyms (offline) or an optional LLM rewrite."""

    def expand(self, query):
        query = query.strip()
        if os.getenv("OPENAI_API_KEY"):
            try:
                return self._expand_llm(query)
            except Exception:
                pass
        return self._expand_synonyms(query)

    def _expand_synonyms(self, query):
        tokens = tokenize(query)
        base_keywords = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        extra = []
        seen = set(tokens)
        for token in tokens:
            for syn in SYNONYMS.get(token, []):
                if syn not in seen:
                    extra.append(syn)
                    seen.add(syn)

        keywords = (base_keywords + extra)[:12]
        sparse_query = f"{query} {' '.join(extra)}".strip() if extra else query
        rewritten = sparse_query

        return {
            "original": query,
            "rewritten": rewritten,
            "keywords": keywords,
            "sparse_query": sparse_query,
            "method": "synonyms",
        }

    def _expand_llm(self, query):
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "You expand search queries for a product/company document retrieval system.\n"
            "Return exactly two lines:\n"
            "REWRITE: <one clearer retrieval-oriented rewrite of the query>\n"
            "KEYWORDS: <comma-separated keywords and synonyms, max 8>\n\n"
            f"Query: {query}"
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content or ""

        rewritten = query
        keywords = []
        for line in text.splitlines():
            line = line.strip()
            if line.upper().startswith("REWRITE:"):
                rewritten = line.split(":", 1)[1].strip() or query
            elif line.upper().startswith("KEYWORDS:"):
                raw = line.split(":", 1)[1]
                keywords = [k.strip() for k in raw.split(",") if k.strip()][:8]

        if not keywords:
            return self._expand_synonyms(query)

        sparse_query = f"{query} {' '.join(keywords)}"
        return {
            "original": query,
            "rewritten": rewritten,
            "keywords": keywords,
            "sparse_query": sparse_query,
            "method": "llm",
        }

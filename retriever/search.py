import numpy as np

from retriever import config
from retriever.ingest import embed_texts, get_collection


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def fetch_candidates(query: str, n: int, collection) -> list[dict]:
    """Return the n nearest chunks (with embeddings), best first, with cosine similarity."""
    res = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    return [
        {
            "id": id_,
            "text": text,
            "metadata": meta,
            "score": 1 - dist,  # Chroma returns cosine distance
            "embedding": np.asarray(emb),
        }
        for id_, text, meta, dist, emb in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], res["embeddings"][0]
        )
    ]


def select_mmr(candidates: list[dict], k: int, lambda_mult: float) -> list[dict]:
    """Maximal marginal relevance: balance similarity to the query against similarity to chunks already picked."""
    vecs = np.stack([c["embedding"] / np.linalg.norm(c["embedding"]) for c in candidates])
    selected = [0]  # candidates arrive best-first
    while len(selected) < min(k, len(candidates)):
        rest = [i for i in range(len(candidates)) if i not in selected]
        redundancy = vecs[rest] @ vecs[selected].T
        mmr = [
            lambda_mult * candidates[i]["score"] - (1 - lambda_mult) * redundancy[j].max()
            for j, i in enumerate(rest)
        ]
        selected.append(rest[int(np.argmax(mmr))])
    return [candidates[i] for i in selected]


def select_doc_cap(candidates: list[dict], k: int, max_per_doc: int) -> list[dict]:
    """Take chunks in similarity order, allowing at most max_per_doc per document."""
    counts: dict[str, int] = {}
    picked = []
    for c in candidates:
        sid = c["metadata"]["doc_id"]
        if counts.get(sid, 0) < max_per_doc:
            counts[sid] = counts.get(sid, 0) + 1
            picked.append(c)
        if len(picked) == k:
            break
    return picked


def search(query: str, k: int = 5, collection=None, strategy: str | None = None) -> list[dict]:
    """Return the top-k chunks for a query using the configured strategy."""
    collection = collection or get_collection()
    strategy = strategy or config.SEARCH_STRATEGY
    if strategy == "similarity":
        results = fetch_candidates(query, k, collection)
    else:
        candidates = fetch_candidates(query, max(config.FETCH_K, k), collection)
        if strategy == "mmr":
            results = select_mmr(candidates, k, config.MMR_LAMBDA)
        elif strategy in ("doc_cap", "section_cap"):  # section_cap: old name
            results = select_doc_cap(candidates, k, config.MAX_PER_DOC)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    for r in results:
        del r["embedding"]
    return results


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the difference between embeddings and a retriever?"
    for r in search(q):
        print(f"{r['score']:.3f}  {r['metadata']['doc_id'][-40:]}  {r['text'][:80]!r}")

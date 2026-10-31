import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from retriever import config
from retriever.ingest import get_collection
from retriever.rerank import rerank
from retriever.search import fetch_candidates


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index over every chunk currently in the vector store."""
    data = get_collection().get(include=["documents", "metadatas"])
    chunks = [
        {"id": id_, "text": text, "metadata": meta}
        for id_, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    return BM25Okapi([tokenize(c["text"]) for c in chunks]), chunks


def lexical_search(query: str, n: int) -> list[dict]:
    """Top-n chunks by BM25 keyword score, best first."""
    bm25, chunks = _bm25_index()
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:n]
    return [{**chunks[i], "score": float(scores[i])} for i in order]


def dense_search(query: str, n: int, collection=None) -> list[dict]:
    """Top-n chunks by embedding similarity, best first."""
    results = fetch_candidates(query, n, collection or get_collection())
    for r in results:
        del r["embedding"]
    return results


def rrf_fuse(rankings: list[list[dict]], c: int = config.RRF_C) -> list[dict]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (c + rank)."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            entry = fused.setdefault(chunk["id"], {**chunk, "score": 0.0})
            entry["score"] += 1 / (c + rank)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


def hybrid_pool(query: str, pool_k: int = config.POOL_K) -> list[dict]:
    """Dense and BM25 results fused with RRF, truncated to pool_k."""
    n = config.CANDIDATES_PER_RETRIEVER
    return rrf_fuse([dense_search(query, n), lexical_search(query, n)])[:pool_k]


def hybrid_search(
    query: str, pool_k: int = config.POOL_K, final_k: int = config.FINAL_K
) -> tuple[list[dict], list[dict]]:
    """Return (pool, final): the fused top pool_k, and the reranked top final_k of that pool."""
    pool = hybrid_pool(query, pool_k)
    return pool, rerank(query, pool, final_k)

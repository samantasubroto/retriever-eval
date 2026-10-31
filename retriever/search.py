from retriever.ingest import embed_texts, get_collection


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def search(query: str, k: int = 5, collection=None) -> list[dict]:
    """Return the top-k chunks for a query, best first, with cosine similarity."""
    collection = collection or get_collection()
    res = collection.query(query_embeddings=[embed_query(query)], n_results=k)
    return [
        {
            "id": id_,
            "text": text,
            "metadata": meta,
            "score": 1 - dist,  # Chroma returns cosine distance
        }
        for id_, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the difference between embeddings and a retriever?"
    for r in search(q):
        print(f"{r['score']:.3f}  {r['metadata']['section_id']}  {r['text'][:80]!r}")

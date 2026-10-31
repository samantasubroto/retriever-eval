def dedupe(section_ids: list[str]) -> list[str]:
    """Unique section ids, keeping first-seen (rank) order."""
    return list(dict.fromkeys(section_ids))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the relevant sections found in the top-k chunks."""
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1.0 if any relevant section is in the top-k chunks."""
    return float(bool(set(retrieved[:k]) & set(relevant)))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of the top-k chunks that come from a relevant section."""
    top = retrieved[:k]
    return sum(s in set(relevant) for s in top) / len(top) if top else 0.0


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1 / rank of the first chunk from a relevant section (0 if none)."""
    for rank, s in enumerate(retrieved, start=1):
        if s in set(relevant):
            return 1 / rank
    return 0.0

import json

from openai import OpenAI

from retriever import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

PROMPT = """Rate how useful each numbered CHUNK is for answering the QUESTION, from 0 (irrelevant) to 10 (directly answers it).
Judge only the chunk text. Return JSON: {{"scores": [int, ...]}} with exactly {n} entries, in chunk order.

QUESTION: {question}

CHUNKS:
{chunks}"""


def rerank(query: str, chunks: list[dict], top_n: int) -> list[dict]:
    """Score every chunk with an LLM and keep the top_n. Ties keep their incoming order."""
    numbered = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    resp = client.chat.completions.create(
        model=config.RERANK_MODEL,
        messages=[{"role": "user", "content": PROMPT.format(question=query, chunks=numbered, n=len(chunks))}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    scores = json.loads(resp.choices[0].message.content)["scores"]
    if len(scores) != len(chunks):  # malformed reply: fall back to the fused order
        return chunks[:top_n]
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))[:top_n]
    return [{**chunks[i], "score": float(scores[i])} for i in order]

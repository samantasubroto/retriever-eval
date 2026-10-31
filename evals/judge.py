import json

from openai import OpenAI

from retriever import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

RECALL_PROMPT = """You are grading a retriever.
Split the REFERENCE ANSWER into its distinct key claims. For each claim, decide whether it is supported by the RETRIEVED CONTEXT (not by your own knowledge).
Return JSON: {{"claims": [{{"claim": str, "supported": bool}}]}}

QUESTION: {question}

REFERENCE ANSWER: {ground_truth}

RETRIEVED CONTEXT:
{context}"""

PRECISION_PROMPT = """You are grading a retriever.
For each numbered CHUNK, decide whether it contains information useful for answering the QUESTION, given the REFERENCE ANSWER.
Return JSON: {{"relevant": [bool, ...]}} with exactly {n} entries, in chunk order.

QUESTION: {question}

REFERENCE ANSWER: {ground_truth}

CHUNKS:
{chunks}"""


def _ask(prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def contextual_recall(question: str, ground_truth: str, chunks: list[str]) -> float:
    """Fraction of ground-truth claims supported by the retrieved chunks."""
    context = "\n\n---\n\n".join(chunks)
    claims = _ask(RECALL_PROMPT.format(question=question, ground_truth=ground_truth, context=context))["claims"]
    return sum(c["supported"] for c in claims) / len(claims) if claims else 0.0


def contextual_precision(question: str, ground_truth: str, chunks: list[str]) -> float:
    """Fraction of retrieved chunks the judge marks as useful."""
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, start=1))
    verdicts = _ask(
        PRECISION_PROMPT.format(question=question, ground_truth=ground_truth, chunks=numbered, n=len(chunks))
    )["relevant"]
    return sum(verdicts) / len(verdicts) if verdicts else 0.0

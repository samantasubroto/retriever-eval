"""Audit the golden dataset: are the reference answers actually supported by the sections they cite?"""
import json
from pathlib import Path

from evals.judge import _ask
from evals.run import load_golden
from retriever import config
from retriever.ingest import load_pdf

PROMPT = """You are auditing a test dataset.
Below is a corpus split into SECTIONS. Split the REFERENCE ANSWER into its distinct key claims. For each claim, list the ids of every section whose text supports it. Use an empty list if no section supports the claim. Use only the corpus text, not your own knowledge.
Return JSON: {{"claims": [{{"claim": str, "sections": [str]}}]}}

REFERENCE ANSWER: {ground_truth}

CORPUS:
{corpus}"""


def corpus_text() -> str:
    pdf = next(Path(config.DATA_DIR).glob("*.pdf"))
    return "\n\n".join(f"=== {'RAG-%03d' % p['page']} ===\n{p['text']}" for p in load_pdf(pdf))


def audit_item(item: dict, corpus: str) -> dict:
    claims = _ask(PROMPT.format(ground_truth=item["ground_truth"], corpus=corpus))["claims"]
    listed = set(item["source_section_ids"])
    used = {s for c in claims for s in c["sections"]}
    return {
        "id": item["id"],
        "n_claims": len(claims),
        "claim_support": sum(bool(c["sections"]) for c in claims) / len(claims),
        "listed_used": len(listed & used) / len(listed),
        "unlisted_used": sorted(used - listed),
        "listed_unused": sorted(listed - used),
        "unsupported_claims": [c["claim"] for c in claims if not c["sections"]],
    }


if __name__ == "__main__":
    corpus = corpus_text()
    rows = [audit_item(item, corpus) for item in load_golden()]
    for r in rows:
        print(f"{r['id']}  claims={r['n_claims']:>2}  supported={r['claim_support']:.2f}  listed_used={r['listed_used']:.2f}  "
              f"unlisted_used={len(r['unlisted_used'])}  listed_unused={len(r['listed_unused'])}")
    n = len(rows)
    print(f"\nMEAN  supported={sum(r['claim_support'] for r in rows)/n:.2f}  listed_used={sum(r['listed_used'] for r in rows)/n:.2f}  "
          f"unlisted_used={sum(len(r['unlisted_used']) for r in rows)/n:.1f}  listed_unused={sum(len(r['listed_unused']) for r in rows)/n:.1f}")
    with open("evals/results_audit.json", "w") as f:
        json.dump(rows, f, indent=2)

"""Build a small, reproducible subset of MultiHop-RAG (https://github.com/yixuantt/MultiHop-RAG, ODC-BY).

Picks n non-null queries (stratified by question type) whose evidence articles are all in the corpus and whose
`fact` snippets appear verbatim in the article body, then keeps only those articles plus random distractors.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

TYPES = ("inference_query", "comparison_query", "temporal_query")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="folder containing MultiHopRAG.json and corpus.json")
    ap.add_argument("--out-dir", default="data/multihop")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--distractors", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    queries = json.loads((raw / "MultiHopRAG.json").read_text())
    corpus = json.loads((raw / "corpus.json").read_text())
    by_url = {a["url"]: a for a in corpus}

    # keep only queries whose labels can be checked against the corpus text
    usable = defaultdict(list)
    for q in queries:
        ev = q["evidence_list"]
        if q["question_type"] not in TYPES or not ev:
            continue
        arts = [by_url.get(e["url"]) for e in ev]
        if all(a and norm(e["fact"]) in norm(a["body"]) for a, e in zip(arts, ev)):
            usable[q["question_type"]].append(q)
    print({t: len(v) for t, v in usable.items()}, "usable queries by type")

    rng = random.Random(args.seed)
    per_type = [args.n // len(TYPES) + (i < args.n % len(TYPES)) for i in range(len(TYPES))]
    picked = [q for t, k in zip(TYPES, per_type) for q in rng.sample(usable[t], k)]
    rng.shuffle(picked)

    golden = []
    for i, q in enumerate(picked, start=1):
        urls = list(dict.fromkeys(e["url"] for e in q["evidence_list"]))
        golden.append(
            {
                "id": f"MH-{i:03d}",
                "question": q["query"],
                "ground_truth": q["answer"],
                "relevant_docs": urls,
                "facts": [e["fact"] for e in q["evidence_list"]],
                "type": q["question_type"],
            }
        )

    needed = {u for g in golden for u in g["relevant_docs"]}
    others = [a for a in corpus if a["url"] not in needed]
    subset = [by_url[u] for u in sorted(needed)] + rng.sample(others, args.distractors)
    rng.shuffle(subset)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden.json").write_text(json.dumps(golden, indent=2))
    (out / "corpus.json").write_text(json.dumps(subset, indent=2))
    chars = sum(len(a["body"]) for a in subset)
    print(f"{len(golden)} questions, {len(needed)} evidence articles + {args.distractors} distractors = {len(subset)} articles")
    print(f"corpus size: {chars:,} chars (~{chars // 4:,} tokens to embed)")
    print(f"evidence docs per question: {sorted(len(g['relevant_docs']) for g in golden)}")


if __name__ == "__main__":
    main()

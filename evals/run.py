import argparse
import json
from statistics import mean

from evals import metrics
from retriever import config
from retriever.hybrid import hybrid_search
from retriever.search import search


def load_golden(path: str = config.GOLDEN_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def score_chunks(chunks: list[dict], item: dict, ks: tuple[int, ...], prefix: str = "") -> dict:
    """Document-level recall, hit rate and precision at each k; plus evidence-fact recall when the item has facts."""
    relevant = item.get("relevant_docs") or item["source_section_ids"]
    docs = [c["metadata"]["doc_id"] for c in chunks]
    row = {}
    for k in ks:
        row[f"{prefix}recall@{k}"] = metrics.recall_at_k(docs, relevant, k)
        row[f"{prefix}hit@{k}"] = metrics.hit_at_k(docs, relevant, k)
        row[f"{prefix}precision@{k}"] = metrics.precision_at_k(docs, relevant, k)
        if "facts" in item:
            row[f"{prefix}fact_recall@{k}"] = metrics.fact_recall_at_k([c["text"] for c in chunks], item["facts"], k)
    return row


def evaluate_item(item: dict, use_judge: bool) -> dict:
    """Retrieve for one golden question and compute all metrics."""
    relevant = item.get("relevant_docs") or item["source_section_ids"]
    row = {"id": item["id"], "difficulty": item.get("difficulty") or item.get("type")}

    if config.SEARCH_STRATEGY == "hybrid_rerank":
        # two stages: the fused pool (rrf_*, scored at the final k and the pool k) and the reranked final list
        pool, results = hybrid_search(item["question"])
        row.update(score_chunks(pool, item, (config.FINAL_K, config.POOL_K), prefix="rrf_"))
        ks, judge_k = (config.FINAL_K,), config.FINAL_K
    else:
        results = search(item["question"], k=max(config.K_VALUES))
        ks, judge_k = config.K_VALUES, config.JUDGE_K

    docs = [r["metadata"]["doc_id"] for r in results]
    row["retrieved"] = metrics.dedupe(docs)
    row.update(score_chunks(results, item, ks))
    row["mrr"] = metrics.reciprocal_rank(docs, relevant)

    if use_judge:
        from evals import judge

        texts = [r["text"] for r in results[:judge_k]]
        reference = item["ground_truth"]
        if "facts" in item:  # short answers carry no claims: judge precision against the answer plus its evidence
            reference = f"Answer: {item['ground_truth']}. Supporting evidence: " + " ".join(item["facts"])
        else:
            row[f"ctx_recall@{judge_k}"] = judge.contextual_recall(item["question"], reference, texts)
        row[f"ctx_precision@{judge_k}"] = judge.contextual_precision(item["question"], reference, texts)
    return row


def print_report(rows: list[dict]) -> None:
    keys = [k for k in rows[0] if k not in ("id", "difficulty", "retrieved")]
    print(f"\n{'id':<10}" + "".join(f"{k:>16}" for k in keys))
    for r in rows:
        print(f"{r['id']:<10}" + "".join(f"{r[k]:>16.2f}" for k in keys))
    print(f"\n{'MEAN':<10}" + "".join(f"{mean(r[k] for r in rows):>16.2f}" for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the retriever on the golden dataset.")
    parser.add_argument("--judge", action="store_true", help="also run LLM-judged contextual recall/precision")
    parser.add_argument("--out", default="evals/results.json", help="where to save per-question results")
    args = parser.parse_args()

    rows = [evaluate_item(item, args.judge) for item in load_golden()]
    print_report(rows)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved per-question results to {args.out}")


if __name__ == "__main__":
    main()

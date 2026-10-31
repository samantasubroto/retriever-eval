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


def score_sections(sections: list[str], relevant: list[str], ks: tuple[int, ...], prefix: str = "") -> dict:
    """Section-ID recall, hit rate and precision at each k."""
    row = {}
    for k in ks:
        row[f"{prefix}recall@{k}"] = metrics.recall_at_k(sections, relevant, k)
        row[f"{prefix}hit@{k}"] = metrics.hit_at_k(sections, relevant, k)
        row[f"{prefix}precision@{k}"] = metrics.precision_at_k(sections, relevant, k)
    return row


def evaluate_item(item: dict, use_judge: bool) -> dict:
    """Retrieve for one golden question and compute all metrics."""
    relevant = item["source_section_ids"]
    section_of = lambda chunks: [c["metadata"]["section_id"] for c in chunks]

    if config.SEARCH_STRATEGY == "hybrid_rerank":
        # two stages: the fused pool (rrf_*, scored at the final k and the pool k) and the reranked final list
        pool, results = hybrid_search(item["question"])
        row = {"id": item["id"], "difficulty": item["difficulty"], "retrieved": metrics.dedupe(section_of(results))}
        row.update(score_sections(section_of(pool), relevant, (config.FINAL_K, config.POOL_K), prefix="rrf_"))
        sections, ks, judge_k = section_of(results), (config.FINAL_K,), config.FINAL_K
    else:
        results = search(item["question"], k=max(config.K_VALUES))
        sections, ks, judge_k = section_of(results), config.K_VALUES, config.JUDGE_K
        row = {"id": item["id"], "difficulty": item["difficulty"], "retrieved": metrics.dedupe(sections)}

    row.update(score_sections(sections, relevant, ks))
    row["mrr"] = metrics.reciprocal_rank(sections, relevant)

    if use_judge:
        from evals import judge

        texts = [r["text"] for r in results[:judge_k]]
        row[f"ctx_recall@{judge_k}"] = judge.contextual_recall(item["question"], item["ground_truth"], texts)
        row[f"ctx_precision@{judge_k}"] = judge.contextual_precision(item["question"], item["ground_truth"], texts)
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

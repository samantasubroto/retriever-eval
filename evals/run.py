import argparse
import json
from statistics import mean

from evals import metrics
from retriever import config
from retriever.search import search


def load_golden(path: str = config.GOLDEN_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def evaluate_item(item: dict, use_judge: bool) -> dict:
    """Retrieve for one golden question and compute all metrics."""
    relevant = item["source_section_ids"]
    results = search(item["question"], k=max(config.K_VALUES))
    sections = [r["metadata"]["section_id"] for r in results]

    row = {"id": item["id"], "difficulty": item["difficulty"], "retrieved": metrics.dedupe(sections)}
    for k in config.K_VALUES:
        row[f"recall@{k}"] = metrics.recall_at_k(sections, relevant, k)
        row[f"hit@{k}"] = metrics.hit_at_k(sections, relevant, k)
        row[f"precision@{k}"] = metrics.precision_at_k(sections, relevant, k)
    row["mrr"] = metrics.reciprocal_rank(sections, relevant)

    if use_judge:
        from evals import judge

        texts = [r["text"] for r in results[: config.JUDGE_K]]
        row[f"ctx_recall@{config.JUDGE_K}"] = judge.contextual_recall(item["question"], item["ground_truth"], texts)
        row[f"ctx_precision@{config.JUDGE_K}"] = judge.contextual_precision(item["question"], item["ground_truth"], texts)
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

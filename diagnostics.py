"""Select a retrieval action from labeled task-level ranking probes."""

import argparse
import json

from metrics import macro_recall


def prescribe(gain, separation, ceiling, large_r10):
    """Map fractional recall probes to dense, large, or dense_rerank.

    Gain and separation are recall changes relative to dense retrieval; ceiling
    and large_r10 are absolute recall values. Threshold comparisons are strict.
    """
    if separation < -0.02 and ceiling - large_r10 > 0.10:
        return "dense_rerank"
    return "large" if gain > 0.02 else "dense"


def diagnose(qrels, dense, large, cross_encoder):
    """Return recall probes and an action for matching query-ID mappings.

    Qrels supplies relevant IDs. Dense needs 100 unique leading candidates,
    large needs 10, and cross_encoder must reorder exactly the dense top 50.
    Gain compares large with dense R@10; separation compares the cross-encoder
    with dense R@10; ceiling is dense R@100. All queries have equal weight.
    """
    if not qrels or any(set(rankings) != set(qrels) for rankings in (dense, large, cross_encoder)):
        raise ValueError("All probe query IDs must match nonempty qrels")
    for qid in qrels:
        if len(dense[qid]) < 100 or len(set(dense[qid][:100])) != 100:
            raise ValueError("Recall ceiling requires 100 unique dense candidates")
        if len(large[qid]) < 10 or len(set(large[qid][:10])) != 10:
            raise ValueError("The larger retriever must provide 10 unique candidates")
        if len(cross_encoder[qid]) != 50 or set(cross_encoder[qid]) != set(dense[qid][:50]):
            raise ValueError("The cross-encoder must rank the same dense top-50 pool")
    dense_r10 = macro_recall(qrels, dense)
    large_r10 = macro_recall(qrels, large)
    gain = large_r10 - dense_r10
    separation = macro_recall(qrels, cross_encoder) - dense_r10
    ceiling = macro_recall(qrels, dense, 100)
    return {
        "gain": gain,
        "separation": separation,
        "ceiling": ceiling,
        "dense_r10": dense_r10,
        "large_r10": large_r10,
        "action": prescribe(gain, separation, ceiling, large_r10),
    }


def main():
    """Load precomputed ranking probes from JSON and print a diagnosis as JSON."""
    parser = argparse.ArgumentParser(
        description="Compute task-level retrieval probes and select an action.",
        epilog='Input JSON: {"qrels": {"query_id": ["relevant_id"]}, "pools": '
               '{"dense": {"query_id": ["document_id", "..."]}, "large": {"query_id": '
               '["document_id", "..."]}}, "cross_encoder": {"query_id": ["document_id", "..."]}}. '
               "Dense/large pools use Qwen3-Embedding-0.6B/4B; cross_encoder contains "
               "BAAI/bge-reranker-v2-m3 rankings of each dense top-50 pool. "
               "Thresholds: separation < -0.02 and ceiling - large R@10 > 0.10 select dense_rerank; "
               "otherwise gain > 0.02 selects large, and the default action is dense. "
               "Example: python diagnostics.py /path/to/probes.json")
    parser.add_argument("input", help="JSON containing qrels, retrieval pools, and cross-encoder rankings")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    print(json.dumps(diagnose(data["qrels"], data["pools"]["dense"],
                              data["pools"]["large"], data["cross_encoder"]), indent=2))


if __name__ == "__main__":
    main()

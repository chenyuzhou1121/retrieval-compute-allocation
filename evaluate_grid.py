"""Evaluate retrieval pools and Qwen reranking actions as task and query recall."""

import argparse
import csv
import json
import logging
from pathlib import Path

from allocation import FIELDS, PRICES
from metrics import recall
from reranker import MODEL_IDS, QwenReranker


def validate(data):
    """Check a task object's required IDs and candidate pools; return None.

    Queries and qrels must share nonempty IDs. Each dense/large pool needs 100
    unique leading IDs, with document text available for the leading 50.
    """
    if not isinstance(data["task"], str) or not data["task"]:
        raise ValueError("Each task needs a nonempty name")
    qids = set(data["qrels"])
    if not qids or set(data["queries"]) != qids:
        raise ValueError("Query IDs must match nonempty qrels")
    for gold in data["qrels"].values():
        if not gold:
            raise ValueError("Each query needs at least one relevant document")
    for pool in ("dense", "large"):
        if set(data["pools"][pool]) != qids:
            raise ValueError("Pool query IDs must match qrels")
        for candidates in data["pools"][pool].values():
            if len(candidates) < 100 or len(set(candidates[:100])) != 100:
                raise ValueError("Provide 100 unique candidates per retrieval pool")
            if any(cid not in data["documents"] for cid in candidates[:50]):
                raise ValueError("Candidate document text is missing")


def evaluate(datasets, args):
    """Return grid rows and task/query/action R@10 for uniquely named tasks.

    Datasets contain queries, qrels, document text, and dense/large ranked pools.
    Args supplies device and batch_size. Each reranker scores the same top-50
    candidates for a pool; task utilities average queries with equal weight.
    """
    for data in datasets:
        validate(data)
    names = [data["task"] for data in datasets]
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide nonempty, uniquely named tasks")
    rows = {data["task"]: {"task": data["task"], "n": len(data["qrels"])} for data in datasets}
    utilities = {data["task"]: {qid: {} for qid in data["queries"]} for data in datasets}

    def record(data, action, rankings):
        """Store per-query recall and its task mean for one action."""
        task = data["task"]
        for qid, gold in data["qrels"].items():
            utilities[task][qid][action] = recall(gold, rankings[qid])
        rows[task][action] = sum(u[action] for u in utilities[task].values()) / rows[task]["n"]

    for data in datasets:
        for pool in ("dense", "large"):
            record(data, pool, data["pools"][pool])
    for scale, suffix in (("0.6B", "rerank_small"), ("4B", "rerank"), ("8B", "rerank_large")):
        # Reuse one model across tasks and pools before loading the next size.
        logging.info("Loading reranker=%s", MODEL_IDS[scale])
        scorer = QwenReranker(MODEL_IDS[scale], args.device, args.batch_size)
        try:
            for data in datasets:
                task = data["task"]
                for pool in ("dense", "large"):
                    action = pool + "_" + suffix
                    logging.info("Starting task=%s action=%s queries=%d", task, action, rows[task]["n"])
                    rankings = {}
                    for i, (qid, query) in enumerate(data["queries"].items(), 1):
                        rankings[qid] = scorer.rank(query, data["pools"][pool][qid][:50], data["documents"])
                        if i % 25 == 0:
                            logging.info("task=%s action=%s queries=%d/%d", task, action, i, rows[task]["n"])
                    record(data, action, rankings)
                    logging.info("task=%s action=%s R@10=%.8f", task, action, rows[task][action])
        finally:
            scorer.close()
    return list(rows.values()), utilities


def parse_args():
    """Return CLI settings merged with optional JSON configuration and defaults.

    Explicit CLI values take precedence; positional inputs replace the config
    list. Paths remain relative to the working directory, including config paths.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate eight retrieval/reranker actions on user-supplied task pools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Each input is JSON: {"task": "task_name", "queries": {"query_id": "query text"}, '
               '"qrels": {"query_id": ["relevant_id"]}, "documents": {"document_id": "document text"}, '
               '"pools": {"dense": {"query_id": ["document_id", "..."]}, '
               '"large": {"query_id": ["document_id", "..."]}}}. '
               "Supply 100 unique IDs per query from Qwen3-Embedding-0.6B (dense) and -4B (large). "
               "Reranking uses top-50, a generic retrieval instruction, and a 2048-token limit. "
               "For concept candidates, supply plain names plus available definitions as document text. "
               "Outputs are one CSV row per task and nested task/query/action R@10 JSON. "
               "All paths are relative to the working directory unless absolute. "
               "Examples: python evaluate_grid.py /path/to/task.json /path/to/another_task.json; "
               "python evaluate_grid.py --config example_config.json. CLI values override config values.")
    parser.add_argument("inputs", nargs="*", help="One task JSON file per task; replaces config inputs when supplied")
    parser.add_argument("--config", help="JSON configuration with inputs, output, utilities, log_file, device, batch_size")
    parser.add_argument("--output", help="Grid CSV path (default: outputs/grid.csv)")
    parser.add_argument("--utilities", help="Per-query R@10 JSON path (default: outputs/utilities.json)")
    parser.add_argument("--log-file", help="Progress log path (default: outputs/evaluate.log)")
    parser.add_argument("--device", help="PyTorch device (default: cuda:0; CUDA uses FP16)")
    parser.add_argument("--batch-size", type=int, help="Candidate pairs per batch (default: 32)")
    args = parser.parse_args()
    defaults = dict(inputs=[], output="outputs/grid.csv", utilities="outputs/utilities.json",
                    log_file="outputs/evaluate.log", device="cuda:0", batch_size=32)
    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict) or set(config) - set(defaults):
            parser.error("Configuration must be an object containing only the supported keys")
    for name, default in defaults.items():
        value = getattr(args, name)
        if value is None or (name == "inputs" and not value):
            setattr(args, name, config.get(name, default))
    if not isinstance(args.inputs, list) or not args.inputs or any(not isinstance(p, str) for p in args.inputs):
        parser.error("Provide input paths as positional arguments or a nonempty config inputs list")
    if not isinstance(args.batch_size, int) or args.batch_size < 1:
        parser.error("batch_size must be a positive integer")
    return args


def main():
    """Write the task grid, per-query utilities, and evaluation progress log."""
    args = parse_args()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("START device=%s batch_size=%d max_length=2048 prices=%s", args.device, args.batch_size, PRICES)
    try:
        datasets = []
        for path in args.inputs:
            with open(path, encoding="utf-8") as handle:
                datasets.append(json.load(handle))
        rows, utilities = evaluate(datasets, args)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        Path(args.utilities).parent.mkdir(parents=True, exist_ok=True)
        with open(args.utilities, "w", encoding="utf-8") as handle:
            json.dump(utilities, handle, indent=2)
        logging.info("COMPLETE tasks=%d queries=%d", len(rows), sum(row["n"] for row in rows))
    except Exception:
        logging.exception("Evaluation failed")
        raise


if __name__ == "__main__":
    main()

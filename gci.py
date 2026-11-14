"""Gate concept injection into LLM reranking by dense/BM25 top-10 overlap."""

import argparse
import json
import logging
from pathlib import Path

from metrics import macro_recall
from reranker import MODEL_IDS, QwenReranker


CONCEPT_INSTRUCTION = (
    "Given a query, judge whether the candidate concept is required to answer the query "
    "or is directly assessed by it"
)


def should_invoke(dense, bm25):
    """Return whether the two ordered ID lists share at most three top-10 IDs."""
    return len(set(dense[:10]).intersection(bm25[:10])) <= 3


def concept_document(name, entry=None):
    """Return name, optional definition, and examples as labeled document text.

    Entry is an optional mapping. Its examples may be a string or a list of
    strings; absent entries produce name-only text.
    """
    lines = [f"Concept: {name}"]
    if entry:
        if entry.get("definition"):
            lines.append("Definition: " + entry["definition"])
        examples = entry.get("examples", [])
        if isinstance(examples, str):
            examples = [examples]
        if examples:
            lines.append("Examples: " + "; ".join(examples))
    return "\n".join(lines)


def gated_rank(query, dense, bm25, names, entries, reranker, instruction=CONCEPT_INSTRUCTION):
    """Return dense order or rerank its top 50 when retrieval overlap is low.

    Names and entries map candidate IDs to names and optional concept details.
    Reranker must expose rank(query, candidates, documents, instruction).
    Skipped lists retain their full length; routed lists contain only the top 50.
    """
    if not should_invoke(dense, bm25):
        return list(dense)
    candidates = list(dense[:50])
    documents = {cid: concept_document(names[cid], entries.get(cid)) for cid in candidates}
    return reranker.rank(query, candidates, documents, instruction)


def main():
    """Write gated rankings, routed query IDs, and optional labeled recall."""
    parser = argparse.ArgumentParser(
        description="Rerank concept candidates when BM25 and dense top-10 overlap is at most three.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Input JSON: {"queries": {"query_id": "query text"}, "pools": {"dense": '
               '{"query_id": ["concept_id", "..."]}}, "bm25": {"query_id": ["concept_id", "..."]}, '
               '"names": {"concept_id": "concept name"}, "entries": {"concept_id": '
               '{"definition": "definition text", "examples": ["example text"]}}}. '
               "Dense pools need 50 unique IDs; BM25 lists need 10 unique IDs. "
               "Missing entries use name-only text. Optional qrels maps query IDs to relevant IDs "
               "for reporting R@10; the gate uses no labels. Skipped queries retain dense order. "
               "Example: python gci.py /path/to/concepts.json --scale 4B")
    parser.add_argument("input", help="Query, retrieval-list, and concept-entry JSON")
    parser.add_argument("--scale", choices=tuple(MODEL_IDS), default="4B", help="Qwen3-Reranker size")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device; CUDA uses FP16")
    parser.add_argument("--batch-size", type=int, default=32, help="Candidate pairs per batch")
    parser.add_argument("--instruction", default=CONCEPT_INSTRUCTION, help="Concept relevance instruction")
    parser.add_argument("--output", default="outputs/gci.json", help="Rankings, routed query IDs, and optional R@10")
    parser.add_argument("--log-file", default="outputs/gci.log", help="Progress and truncation log")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format="%(asctime)s %(message)s")
    logging.info("START model=%s device=%s batch_size=%d", MODEL_IDS[args.scale], args.device, args.batch_size)
    try:
        with open(args.input, encoding="utf-8") as handle:
            data = json.load(handle)
        qids = set(data["queries"])
        dense, bm25 = data["pools"]["dense"], data["bm25"]
        if not qids or set(dense) != qids or set(bm25) != qids:
            raise ValueError("Dense and BM25 query IDs must match nonempty queries")
        for qid in qids:
            if len(dense[qid]) < 50 or len(set(dense[qid][:50])) != 50:
                raise ValueError("Dense pools require 50 unique candidates")
            if len(bm25[qid]) < 10 or len(set(bm25[qid][:10])) != 10:
                raise ValueError("BM25 lists require 10 unique candidates")
            if any(cid not in data["names"] for cid in dense[qid][:50]):
                raise ValueError("A candidate concept name is missing")
        routed = [qid for qid in data["queries"] if should_invoke(dense[qid], bm25[qid])]
        rankings = {qid: list(dense[qid]) for qid in data["queries"]}
        if routed:
            scorer = QwenReranker(MODEL_IDS[args.scale], args.device, args.batch_size)
            try:
                for i, qid in enumerate(routed, 1):
                    rankings[qid] = gated_rank(data["queries"][qid], dense[qid], bm25[qid],
                                               data["names"], data.get("entries", {}), scorer, args.instruction)
                    if i % 25 == 0:
                        logging.info("Reranked queries=%d/%d", i, len(routed))
            finally:
                scorer.close()
        result = dict(rankings=rankings, routed_queries=routed, invocation_rate=len(routed) / len(qids))
        if "qrels" in data:
            result["r10"] = macro_recall(data["qrels"], rankings)
            result["dense_r10"] = macro_recall(data["qrels"], dense)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logging.info("COMPLETE queries=%d routed=%d", len(qids), len(routed))
    except Exception:
        logging.exception("Concept reranking failed")
        raise


if __name__ == "__main__":
    main()

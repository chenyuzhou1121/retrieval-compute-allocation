"""Compute per-query recall and equally weighted mean recall over queries."""

from statistics import mean


def recall(gold, ranked, k=10):
    """Return the fraction of distinct gold IDs present in the first k results.

    Gold must be nonempty, ranked must support slicing, and k is a positive
    cutoff. Duplicate relevant IDs or retrieved IDs do not add extra credit.
    """
    relevant = set(gold)
    if not relevant:
        raise ValueError("Each query needs at least one relevant document")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def macro_recall(qrels, rankings, k=10):
    """Average recall over matching, nonempty query-ID mappings.

    Qrels maps each query ID to relevant IDs; rankings maps it to ordered IDs.
    Each query contributes equally regardless of its number of relevant IDs.
    """
    if not qrels or set(qrels) != set(rankings):
        raise ValueError("Nonempty qrels and rankings must have identical query IDs")
    return mean(recall(gold, rankings[qid], k) for qid, gold in qrels.items())

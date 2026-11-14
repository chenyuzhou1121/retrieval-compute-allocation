"""Compare task and query action selectors using shared labeled-query draws."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from allocation import MAIN_MENU, PRICES


# Action order also breaks prediction ties in favor of the cheaper action.
MENU = MAIN_MENU[:3]
COSTS = np.array([PRICES[a] for a in MENU])
# These are input column names; feature values are supplied by the caller.
RIDGE_FEATURES = (
    "margin nqc overlap qlen top1 dense_top1 dense_margin dense_nqc dense_bm25_ratio "
    "qpp_wig qpp_smv qpp_mean10 dense_4b_overlap10 dense_4b_overlap100 "
    "ce_margin ce_nqc ce_mean10"
).split()
BASE_FEATURES = (
    "margin nqc overlap qlen top1 dense_top1 dense_margin dense_nqc dense_bm25_ratio "
    "qpp_margin qpp_nqc qpp_wig qpp_smv qpp_top1 qpp_mean10 "
    "dense_4b_overlap10 dense_4b_overlap100 pct_margin pct_nqc pct_overlap pct_qlen "
    "pct_top1 pct_dense_top1 pct_dense_margin pct_dense_nqc pct_dense_bm25_ratio "
    "pct_conf pct_uncertain pct_disagree pct_dense_conf pct_dense_low pct_qpp_margin "
    "pct_qpp_nqc pct_qpp_wig pct_qpp_smv pct_qpp_conf pct_qpp_uncertain "
    "z_margin z_nqc z_overlap z_qlen z_top1 z_dense_top1 z_dense_margin z_dense_nqc "
    "z_dense_bm25_ratio z_qpp_margin z_qpp_nqc z_qpp_top1"
).split()
RANK_FEATURES = (
    "rerank_dense_overlap10 rerank_dense_shift10 rerank_dense_promote10 "
    "rerank_dense_top1_base_rank rerank_dense_missing10 rerank_4b_overlap10 "
    "rerank_4b_shift10 rerank_4b_promote10 rerank_4b_top1_base_rank rerank_4b_missing10 "
    "both_dense_overlap10 both_dense_shift10 both_dense_promote10 "
    "both_dense_top1_base_rank both_dense_missing10 both_4b_overlap10 both_4b_shift10 "
    "both_4b_promote10 both_4b_top1_base_rank both_4b_missing10 "
    "ce_top1 ce_margin ce_nqc ce_mean10 ce_4b_top1 ce_4b_margin ce_4b_nqc ce_4b_mean10 "
    "has_ce has_ce_4b"
).split()
FEATURES = BASE_FEATURES + [p + f for p in ("", "pct_", "z_") for f in RANK_FEATURES]
POLY_FEATURES = (
    "dense_top1 dense_margin dense_nqc margin nqc qpp_margin qpp_nqc qpp_wig "
    "qpp_smv dense_4b_overlap10 dense_4b_overlap100 qlen rerank_dense_shift10 "
    "rerank_dense_overlap10 rerank_4b_overlap10 ce_top1 ce_margin ce_nqc "
    "pct_qpp_conf pct_uncertain"
).split()
POLY_COLUMNS = [FEATURES.index(f) for f in POLY_FEATURES]


def standardize(train, test, minimum=0.0):
    """Return feature matrices scaled with the training column means and SDs.

    Inputs have shape (queries, features). Constant columns, and columns below
    the minimum SD, use a divisor of one.
    """
    mu, sd = train.mean(0), train.std(0)
    sd[(sd == 0) | (sd < minimum)] = 1.0
    return (train - mu) / sd, (test - mu) / sd


def design(train, test, poly):
    """Return scaled matrices, optionally expanding selected quadratic features.

    Input columns follow FEATURES. Poly mode selects POLY_FEATURES and appends
    squares and every distinct pairwise product after training-only scaling.
    """
    if poly:
        train, test = train[:, POLY_COLUMNS], test[:, POLY_COLUMNS]
    train, test = standardize(train, test)
    if poly:
        i, j = np.triu_indices(train.shape[1], 1)
        train = np.hstack((train, train * train, train[:, i] * train[:, j]))
        test = np.hstack((test, test * test, test[:, i] * test[:, j]))
    return train, test


def ridge(train, y, test, alpha=1.0):
    """Predict test action utilities with centered multi-output ridge regression.

    Train and test are query-by-feature matrices; y is query-by-action utility.
    The intercept is unpenalized, and alpha is the positive ridge penalty.
    """
    xm, ym = train.mean(0), y.mean(0)
    z = train - xm
    # Solve in sample space so quadratic expansion need not enlarge the solve.
    return ym + (test - xm) @ z.T @ np.linalg.solve(
        z @ z.T + alpha * np.eye(len(z)), y - ym)


def curve_ridge(train, y, test):
    """Return action predictions from unit-penalty ridge with a free intercept.

    Inputs are pre-scaled feature matrices and query-by-action training targets.
    Unlike ridge(), this routine solves in feature space.
    """
    design = np.column_stack((np.ones(len(train)), train))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    target = np.column_stack((np.ones(len(test)), test))
    return np.column_stack([target @ np.linalg.solve(
        design.T @ design + penalty, design.T @ y[:, i]) for i in range(y.shape[1])])


def predict(train, y, test, kind, parameter):
    """Return test utility predictions for ridge or nearest-neighbor routing.

    Kind is poly/linear for ridge or uniform/distance for neighbor averaging.
    Parameter is a positive ridge penalty or a positive integer neighbor count.
    Neighbor counts are capped at the available training rows.
    """
    train, test = design(train, test, kind == "poly")
    if kind in ("poly", "linear"):
        return ridge(train, y, test, parameter)
    k = min(parameter, len(train))
    distances = ((test[:, None, :] - train[None, :, :]) ** 2).sum(2)
    indices = np.argpartition(distances, k - 1, axis=1)[:, :k]
    values = y[indices]
    if kind == "uniform":
        return values.mean(1)
    weights = 1 / (np.sqrt(np.take_along_axis(distances, indices, axis=1)) + 1e-9)
    return (values * weights[:, :, None]).sum(1) / weights.sum(1)[:, None]


def strong(qids, x, y, train, test):
    """Return action indices from a router tuned on the labeled training subset.

    X follows FEATURES, y follows MENU, and train/test index rows of both.
    Up to four folds, assigned by sorted query ID, select ridge or kNN settings
    by held-out utility, then lower action cost, then the trial name. A single
    training query supplies one constant action without inner validation.
    """
    if len(train) == 1:
        return np.repeat(np.argmax(y[train][0]), len(test))
    folds = np.empty(len(train), dtype=int)
    order = sorted(range(len(train)), key=lambda i: qids[train[i]])
    folds[order] = np.arange(len(train)) % min(4, len(train))
    trials = [(f"poly2_ridge_alpha={a:g}", "poly", a) for a in (.01, .1, 1., 10.)]
    trials += [(f"linear_ridge_tuned_alpha={a:g}", "linear", a) for a in (.1, 1., 10.)]
    trials += [(f"knn_k={k}_{kind}", kind, k) for k in (3, 5, 11, 21, 41, 81)
               if k <= len(train) for kind in ("distance", "uniform")]
    scores = []
    for name, kind, parameter in trials:
        values, costs = [], []
        for fold in range(folds.max() + 1):
            tr, va = train[folds != fold], train[folds == fold]
            actions = predict(x[tr], y[tr], x[va], kind, parameter).argmax(1)
            values.extend(y[va, actions])
            costs.extend(COSTS[actions])
        scores.append((float(np.mean(values)), -float(np.mean(costs)), name))
    _, kind, parameter = trials[max(range(len(trials)), key=lambda i: scores[i])]
    return predict(x[train], y[train], x[test], kind, parameter).argmax(1)


def prepare(utilities, features):
    """Return per-task query IDs, targets, ridge features, and router features.

    Both inputs are task/query mappings. Utilities map actions to R@10 values;
    features map names to finite numbers, with absent features filled by zero.
    Ridge features are scaled over the full task without using utility labels;
    router features remain raw for scaling inside training folds.
    """
    if not utilities or set(utilities) != set(features):
        raise ValueError("Utilities and features must have the same nonempty tasks")
    data = []
    for task, rows in utilities.items():
        qids = list(rows)
        if len(qids) < 2 or set(qids) != set(features[task]):
            raise ValueError("Each task needs matching query IDs and at least two queries")
        y = np.array([[rows[q][a] for a in MENU] for q in qids], dtype=float)
        small = np.array([[features[task][q].get(f, 0.) for f in RIDGE_FEATURES]
                          for q in qids], dtype=float)
        large = np.array([[features[task][q].get(f, 0.) for f in FEATURES]
                          for q in qids], dtype=float)
        if not all(np.isfinite(v).all() for v in (y, small, large)) or (y < 0).any() or (y > 1).any():
            raise ValueError("Features must be finite and R@10 utilities must lie in [0, 1]")
        small, _ = standardize(small, small, 1e-12)
        data.append((qids, y, small, large))
    return data


def label_curve(utilities, features, labels, seeds, resamples):
    """Return task-macro recall curves from shared random training subsets.

    Labels and seeds are lists of positive budgets and nonnegative seeds.
    Each draw leaves at least one query per task for testing. Selectors share
    training/test rows, and tasks have equal weight regardless of query count.
    Reported intervals are paired 2.5/97.5 percentiles across draw differences.
    """
    if not labels or min(labels) < 1 or not seeds or min(seeds) < 0 or resamples < 1:
        raise ValueError("Use positive labels/resamples and nonnegative seeds")
    data = prepare(utilities, features)
    curve = []
    for n in labels:
        draws = []
        for seed in seeds:
            rng = np.random.default_rng(seed * 1000 + n)
            for _ in range(resamples):
                tasks = []
                for qids, y, small, large in data:
                    # Preserve a held-out set even when the budget exceeds task size.
                    train = rng.choice(len(y), min(n, len(y) - 1), replace=False)
                    test = np.flatnonzero(~np.isin(np.arange(len(y)), train))
                    task_action = np.array([y[train, i].mean() for i in range(len(MENU))]).argmax()
                    ridge_actions = curve_ridge(small[train], y[train], small[test]).argmax(1)
                    strong_actions = strong(qids, large, y, train, test)
                    tasks.append([y[test, task_action].mean(), y[test, ridge_actions].mean(),
                                  y[test, strong_actions].mean()])
                draws.append(np.mean(tasks, axis=0))
            logging.info("labels=%d seed=%d resamples=%d", n, seed, resamples)
        values = np.array(draws)
        row = dict(labels_per_task=n, splits=len(draws), task_selector=float(values[:, 0].mean()),
                   ridge=float(values[:, 1].mean()), strong=float(values[:, 2].mean()))
        for index, name in ((1, "ridge"), (2, "strong")):
            difference = 100 * (values[:, 0] - values[:, index])
            row["task_minus_" + name + "_pp"] = float(difference.mean())
            row["task_minus_" + name + "_ci95_pp"] = np.quantile(difference, [.025, .975]).tolist()
        curve.append(row)
    return curve


def main():
    """Read utilities/features and write matched-label curves and a sibling log."""
    parser = argparse.ArgumentParser(
        description="Compare task selection, ridge, and quadratic-ridge/kNN with matched labels at price cap 10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Utilities JSON: {"task": {"query_id": {"dense": 0.5, "large": 0.6, '
               '"dense_rerank": 0.7}}}. Features JSON: {"task": {"query_id": {"qlen": 10}}}. '
               "Task/query IDs must match; missing feature values are zero. "
               "RIDGE_FEATURES and FEATURES define feature names; 4b denotes the larger retriever. "
               "JSON insertion order determines sampling order; inner folds sort query IDs. "
               "Ridge uses full-task feature standardization; strong routing fits it on training folds. "
               "Outputs contain task-macro means and paired 2.5/97.5 percentiles across label draws. "
               "Example: python matched_labels.py outputs/utilities.json /path/to/features.json")
    parser.add_argument("utilities", help="Per-query action R@10 JSON from evaluate_grid.py")
    parser.add_argument("features", help="Numeric query-feature JSON keyed by task and query ID")
    parser.add_argument("--labels", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                        help="Labeled queries per task; capped at task size minus one")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45],
                        help="Nonnegative random seeds for shared training draws")
    parser.add_argument("--resamples", type=int, default=200, help="Training draws per seed and label budget")
    parser.add_argument("--output", default="outputs/matched.json", help="Result JSON; a sibling .log records progress")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output.with_suffix(".log"), level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logging.info("START labels=%s seeds=%s resamples=%d", args.labels, args.seeds, args.resamples)
    try:
        with open(args.utilities, encoding="utf-8") as handle:
            utilities = json.load(handle)
        with open(args.features, encoding="utf-8") as handle:
            features = json.load(handle)
        result = dict(budget=10, menu=MENU, labels=args.labels, seeds=args.seeds,
                      resamples_per_seed=args.resamples,
                      curve=label_curve(utilities, features, args.labels, args.seeds, args.resamples))
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logging.info("COMPLETE output=%s", output)
    except Exception:
        logging.exception("Matched-label comparison failed")
        raise


if __name__ == "__main__":
    main()

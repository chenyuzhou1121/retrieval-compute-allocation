"""Compare task-specific action choices with uniform and randomized allocations."""

import argparse
import csv
import json
import math
from itertools import combinations
from statistics import mean


# Fixed action costs shared by allocation and matched-label comparisons.
PRICES = {
    "dense": 1.0,
    "dense_rerank_small": 1.5,
    "large": 4.0,
    "large_rerank_small": 5.5,
    "dense_rerank": 10.0,
    "large_rerank": 14.0,
    "dense_rerank_large": 20.0,
    "large_rerank_large": 24.0,
}
MAIN_MENU = ("dense", "large", "dense_rerank", "large_rerank")
FIELDS = ("task", "n", *PRICES)


def read_grid(path):
    """Load task-to-action R@10 mappings from a grid CSV, ignoring query counts.

    Task names must be nonempty and unique, and all eight action columns must
    contain finite values in [0, 1]. The returned mapping preserves row order.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if (not rows or any(not row.get("task") for row in rows)
            or len({row["task"] for row in rows}) != len(rows)):
        raise ValueError("Grid must contain unique task rows")
    grid = {row["task"]: {a: float(row[a]) for a in PRICES} for row in rows}
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for row in grid.values() for value in row.values()):
        raise ValueError("Grid utilities must be finite R@10 values in [0, 1]")
    return grid


def blind_envelope(utilities, prices, cost):
    """Return the best task-independent mixture utility at mean cost <= cost.

    Utilities and prices map action names to scalars. At least one action must
    cost no more than the supplied limit.
    """
    points = sorted((prices[a], value) for a, value in utilities.items())
    best = max(value for price, value in points if price <= cost)
    # With one cost constraint, an optimal mixture needs at most two actions.
    for (p1, v1), (p2, v2) in combinations(points, 2):
        if p1 < p2 and p1 <= cost <= p2:
            best = max(best, v1 + (cost - p1) * (v2 - v1) / (p2 - p1))
    return best


def cheapest_match(grid, actions, target):
    """Find the cheapest task assignment reaching a feasible mean R@10 target.

    Grid rows map actions to utilities. Return task assignments, mean relative
    cost, and mean utility, weighting tasks equally.
    """
    # Costs are multiples of 0.5, so doubled costs give exact integer DP states.
    states = {0: (0.0, ())}
    for row in grid.values():
        updated = {}
        for cost, (utility, assignment) in states.items():
            for action in actions:
                total = cost + int(2 * PRICES[action])
                value = utility + row[action]
                if total not in updated or value > updated[total][0]:
                    updated[total] = (value, assignment + (action,))
        states = updated
    cost = min(c for c, (u, _) in states.items() if u / len(grid) >= target - 1e-12)
    utility, assignment = states[cost]
    return {"assignments": dict(zip(grid, assignment)),
            "mean_cost": cost / (2 * len(grid)), "utility": utility / len(grid)}


def allocate(grid, budget=10.0, menu=MAIN_MENU):
    """Compare allocations on nonempty task-to-action utility mappings.

    Budget caps each available action, rather than the average assignment cost.
    Tasks have equal weight. Utility ties favor lower cost, then action name.
    Return assignments, uniform and mixture utilities, percentage-point gains,
    and the cheapest assignment matching the best uniform utility.
    """
    if not grid or not math.isfinite(budget):
        raise ValueError("Provide a nonempty grid and a finite budget")
    affordable = [a for a in menu if PRICES[a] <= budget]
    if not affordable:
        raise ValueError("Budget admits no action")
    uniform = {a: mean(row[a] for row in grid.values()) for a in affordable}
    def choose(row):
        """Choose the highest-utility affordable action with deterministic ties."""
        return min(affordable, key=lambda a: (-row[a], PRICES[a], a))

    uniform_action = choose(uniform)
    assignments = {task: choose(row) for task, row in grid.items()}
    utility = mean(grid[t][a] for t, a in assignments.items())
    cost = mean(PRICES[a] for a in assignments.values())
    blind = blind_envelope(uniform, PRICES, cost)
    matching = cheapest_match(grid, affordable, uniform[uniform_action])
    matching["saving_pct"] = 100 * (1 - matching["mean_cost"] / PRICES[uniform_action])
    return {
        "budget": budget,
        "assignments": assignments,
        "uniform_action": uniform_action,
        "V_U": uniform[uniform_action],
        "V_T": utility,
        "mean_cost": cost,
        "blind_envelope": blind,
        "same_price_margin_pp": 100 * (utility - blind),
        "uniform_regret_pp": 100 * (utility - uniform[uniform_action]),
        "equal_quality": matching,
        "dissenting_tasks": sum(a != uniform_action for a in assignments.values()),
    }


def main():
    """Read a grid CSV and print allocation comparisons as JSON."""
    parser = argparse.ArgumentParser(
        description="Compare task-oracle allocation with uniform and blind policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CSV columns: " + ", ".join(FIELDS) + ". Utilities are task-mean R@10. "
               "Tasks receive equal weight. Prices: " + json.dumps(PRICES) + ". "
               "Example: python allocation.py --grid outputs/grid.csv --budget 10")
    parser.add_argument("--grid", required=True, help="CSV produced by evaluate_grid.py")
    parser.add_argument("--budget", type=float, default=10.0, help="Hard price cap for every query")
    parser.add_argument("--menu", choices=("four", "eight"), default="four",
                        help="Four main actions or all eight retrieval/reranker combinations")
    args = parser.parse_args()
    menu = MAIN_MENU if args.menu == "four" else tuple(PRICES)
    print(json.dumps(allocate(read_grid(args.grid), args.budget, menu), indent=2))


if __name__ == "__main__":
    main()

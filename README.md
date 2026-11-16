# Retrieval Compute Allocation

This repository allocates retrieval compute by task between a larger first-stage retriever and LLM reranking under a hard per-query price cap. It compares task and query selectors and uses a label-free overlap gate to invoke gated concept injection (GCI), which supplies curated definitions and examples to the reranker.

Paper: Allocate by Task, Invoke by Query: The Granularity of Retrieval Compute Under a Budget (WSDM 2027 Findings).

## Installation

Use a Python environment with pip:

```sh
python3 -m pip install -r requirements.txt
```

Dependencies are NumPy, PyTorch, and Transformers. Reranking loads `Qwen/Qwen3-Reranker-0.6B`, `-4B`, or `-8B` model weights through Transformers. The default device is `cuda:0`; CUDA uses FP16 and other devices use FP32. Allocation and diagnostics use only the Python standard library.

## Evaluate and allocate

Prepare one JSON file per task. The shape below abbreviates candidate lists; supply at least 100 unique leading document IDs per query in each pool, in retrieval order.

```json
{
  "task": "example",
  "queries": {"q1": "query text"},
  "qrels": {"q1": ["d1"]},
  "documents": {"d1": "document text", "d2": "another document"},
  "pools": {
    "dense": {"q1": ["d1", "d2"]},
    "large": {"q1": ["d2", "d1"]}
  }
}
```

Task names must be unique. Query IDs must match across queries, qrels, and both pools. Each query needs at least one relevant ID. Supply document text for every top-50 candidate in either pool, using plain names plus available definitions for concept candidates. The pool names correspond to Qwen3-Embedding-0.6B (`dense`) and Qwen3-Embedding-4B (`large`); the evaluator consumes rankings rather than generating embeddings.

Replace `inputs` in `example_config.json` with your task file paths, then run:

```sh
python3 evaluate_grid.py --config example_config.json
python3 allocation.py --grid outputs/grid.csv --budget 10 --menu four
```

Alternatively, pass task paths directly: `python3 evaluate_grid.py task.json another_task.json`. Evaluation reranks each pool's top 50 with all three reranker sizes. It writes task-mean R@10 to `outputs/grid.csv`, task/query/action R@10 to `outputs/utilities.json`, and progress and truncation counts to `outputs/evaluate.log`. R@10 is the fraction of distinct relevant IDs retrieved in the first ten results.

Allocation prints JSON containing task assignments, uniform and task-oracle utilities (`V_U`, `V_T`), mean cost, the task-independent mixture envelope, percentage-point margins, and the cheapest assignment matching uniform quality. Tasks have equal weight; the CSV `n` column records query counts and does not weight allocation. `--budget` caps every action, not just mean cost. `--menu four` includes `dense`, `large`, `dense_rerank`, and `large_rerank`; `eight` includes all actions.

`results/grid.csv` contains the paper's nine-task, eight-action grid of task-mean R@10 values. K12 appears only as aggregate values. `python3 allocation.py --grid results/grid.csv --budget 10 --menu four` reproduces the four-action allocation at B=10.

Action names combine a pool name with the suffix below. Costs in `allocation.py` are normalized experimental prices with dense retrieval as one unit.

| Suffix | Qwen3 reranker size | Dense cost | Large cost |
| --- | --- | --- | --- |
| None | None | 1.0 | 4.0 |
| `_rerank_small` | 0.6B | 1.5 | 5.5 |
| `_rerank` | 4B | 10.0 | 14.0 |
| `_rerank_large` | 8B | 20.0 | 24.0 |

## Evaluation configuration

All paths are relative to the working directory unless absolute. CLI values override JSON values; positional paths replace the complete `inputs` list. Unknown configuration keys are rejected.

| JSON key | CLI option | Meaning and default |
| --- | --- | --- |
| `inputs` | Positional paths | Required nonempty list of task JSON paths; edit the example placeholder. |
| `output` | `--output` | Task grid CSV, default `outputs/grid.csv`. |
| `utilities` | `--utilities` | Per-query recall JSON, default `outputs/utilities.json`. |
| `log_file` | `--log-file` | Progress log, default `outputs/evaluate.log`. |
| `device` | `--device` | PyTorch device, default `cuda:0`. |
| `batch_size` | `--batch-size` | Positive number of candidate pairs per batch, default 32. |

## Additional entry points

```sh
python3 diagnostics.py probes.json
python3 gci.py concepts.json --scale 4B
python3 matched_labels.py outputs/utilities.json features.json
```

`diagnostics.py` takes `qrels`, `pools.dense`, and `pools.large` with the mappings shown above, plus `cross_encoder`, a query-to-ranked-ID mapping from BAAI/bge-reranker-v2-m3. Dense needs 100 unique leading candidates, large needs ten, and each cross-encoder list must reorder exactly the dense top 50. It prints recall probes and a selected action as JSON. It selects `dense_rerank` when cross-encoder minus dense R@10 is below -0.02 and dense R@100 minus large R@10 exceeds 0.10; otherwise, a large minus dense R@10 gain above 0.02 selects `large`, with `dense` as the default.

`gci.py` takes `queries`, `pools.dense`, `bm25`, and `names`. BM25 maps query IDs to ranked concept IDs; names maps concept IDs to text. Dense needs 50 unique leading IDs and BM25 needs ten, with matching query IDs. Optional `entries` maps concept IDs to objects with `definition` and `examples` (a string or string list); absent entries use the name alone. Optional `qrels` adds recall reporting. The gate invokes GCI when the dense and BM25 top-10 lists share at most three IDs; invoked queries rerank the dense top 50, and other queries keep their full dense list. The JSON output contains rankings, routed query IDs, invocation rate, and optional recall.

Gating options are `--scale` (`0.6B`, `4B`, `8B`; default `4B`), `--device` (default `cuda:0`), `--batch-size` (default 32), `--instruction` (concept relevance prompt), `--output` (default `outputs/gci.json`), and `--log-file` (default `outputs/gci.log`).

`matched_labels.py` takes utilities from evaluation and numeric features shaped as `{"example": {"q1": {"qlen": 10}}}`. Task and query IDs must match, with at least two queries per task. Accepted feature names are listed in `RIDGE_FEATURES` and `FEATURES` in `matched_labels.py`; missing values become zero. It uses `dense`, `large`, and `dense_rerank` utilities to compare a task selector, ridge regression, and a tuned quadratic-ridge/kNN router on shared training draws. Ridge scales features over the full task; the tuned router scales on training subsets.

Options are `--labels` (default `1 3 5 10 20`, capped at task size minus one), `--seeds` (default `41 42 43 44 45`), `--resamples` (default 200 per seed and label budget), and `--output` (default `outputs/matched.json`, with a sibling `.log`). JSON insertion order determines sampling order; inner folds sort query IDs. Output fields `task_selector`, `ridge`, and `strong` contain task-macro recall means; `strong` denotes the tuned router. Paired 2.5/97.5 percentile intervals summarize draw differences.

## Repository layout

```text
allocation.py       Task allocation and relative action costs
diagnostics.py      Labeled ranking probes and action selection
evaluate_grid.py    Eight-action evaluation and configuration loading
gci.py              Gated concept injection into LLM reranking
matched_labels.py   Shared-label task and query selector comparison
metrics.py          Recall helpers
reranker.py         Batched Qwen yes/no scoring
results/grid.csv    Nine-task, eight-action R@10 grid from the paper
example_config.json Evaluation configuration template
requirements.txt    Python dependency declarations
.gitignore          Generated data, outputs, caches, and local settings
README.md           Usage and input formats
LICENSE             MIT license text
```

## License

MIT; see [LICENSE](LICENSE).

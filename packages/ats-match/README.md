# ats-match

A small, dependency-light matching core: given weighted *requirements* and a
pool of atomic *claims*, pick the subset of claims that best covers the
requirements inside a cost budget.

It knows nothing about the application that feeds it — no file formats, no
document vocabulary, no rendering. Embedding and reranking models are injected
through the protocols in `providers.py`.

```python
from ats_match import BM25, Claim, Constraints, Params, Requirement, SectionRule, build_matrix, select

matrix, similarity = build_matrix(requirements, claims, embedder, alias_map)
selection = select(matrix, requirements, claims, constraints, Params(), claim_sim=similarity)
```

Modules:

| module | contents |
|---|---|
| `types.py` | `Requirement`, `Claim`, `SectionRule`, `Constraints`, `Params`, `Selection`, `Step` |
| `lexical.py` | `tokenize` (alias folding, symbol-safe) and `BM25` |
| `fusion.py` | `rrf`, `build_matrix` |
| `select.py` | budgeted lazy-greedy coverage maximisation with a trace |
| `evaluate.py` | `coverage_at_budget`, `precision_at_k`, `kendall_tau` |
| `providers.py` | `Embedder` / `Reranker` protocols |

Only dependency: numpy.

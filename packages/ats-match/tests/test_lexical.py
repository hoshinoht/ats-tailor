"""Tokenisation, alias folding, and BM25."""

import numpy as np

from ats_match import BM25, tokenize
from ats_match.fusion import ranks_from_scores, rrf

ALIASES = {
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "container orchestration": "kubernetes",
    "c++": "cpp",
    ".net": "dotnet",
    "ci/cd": "ci-cd",
    "node.js": "nodejs",
}


def test_aliases_fold_to_one_canonical_token():
    assert tokenize("K8s cluster", ALIASES) == ["kubernetes", "cluster"]
    assert tokenize("Kubernetes cluster", ALIASES) == ["kubernetes", "cluster"]
    assert tokenize("container orchestration", ALIASES) == ["kubernetes"]


def test_symbol_terms_survive_the_split():
    assert tokenize("C++ and .NET with CI/CD", ALIASES) == [
        "cpp", "and", "dotnet", "with", "ci-cd",
    ]
    assert tokenize("Node.js backend", ALIASES) == ["nodejs", "backend"]


def test_symbol_terms_are_known_without_an_alias_entry():
    assert "c#" in tokenize("Experience with C# desired", {})


def test_tokenize_is_case_and_punctuation_insensitive():
    assert tokenize("Deploy, monitor; and SCALE.", {}) == [
        "deploy", "monitor", "and", "scale",
    ]


def test_bm25_ranks_the_document_that_shares_rare_tokens():
    docs = [
        tokenize("kubernetes cluster autoscaling", ALIASES),
        tokenize("wrote documentation and tests", ALIASES),
        tokenize("k8s deployment manifests", ALIASES),
    ]
    scores = BM25(docs).score(tokenize("K8s", ALIASES))
    assert scores[0] > 0 and scores[2] > 0
    assert scores[1] == 0


def test_bm25_scores_nothing_for_an_unknown_query():
    docs = [tokenize("alpha beta", {}), tokenize("gamma", {})]
    assert BM25(docs).score(["delta"]).tolist() == [0.0, 0.0]


def test_rrf_prefers_the_item_ranked_high_by_both_lists():
    fused = rrf([[2, 0, 1], [2, 1, 0]], k=1, n=3)
    assert int(np.argmax(fused)) == 2
    # 0 and 1 swap places between the two lists, so they fuse to the same score
    assert fused[0] == fused[1] < fused[2]
    # an item both lists rank beats one only a single list saw
    one_sided = rrf([[0, 1], [0]], k=1, n=2)
    assert one_sided[0] > one_sided[1]


def test_rrf_weights_shift_the_winner():
    unweighted = rrf([[0, 1], [1, 0]], k=1, n=2)
    assert unweighted[0] == unweighted[1]
    weighted = rrf([[0, 1], [1, 0]], k=1, n=2, weights=[3.0, 1.0])
    assert weighted[0] > weighted[1]


def test_ranks_from_scores_breaks_ties_by_index():
    assert ranks_from_scores(np.array([0.5, 0.5, 0.9])).tolist() == [2, 0, 1]

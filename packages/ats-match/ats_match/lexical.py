"""Tokenisation with alias folding, and BM25 over the folded tokens.

The point of the folding is that a target asking for "K8s" and a claim saying
"Kubernetes" must land on the same token before any lexical scoring happens —
otherwise BM25 sees two unrelated words and the dense side has to carry the
match alone.

Tokenisation is deliberately not a plain ``\\W+`` split: ``C++``, ``C#``,
``.NET``, ``CI/CD`` and ``Node.js`` all lose their identity that way, and so
does every multi-word alias. Known terms are matched first, longest first, and
only what is left is split on non-alphanumerics.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

#: Terms whose punctuation is part of the name. Extend through ``aliases``:
#: any alias containing a non-alphanumeric character is protected the same way.
SYMBOL_TERMS: tuple[str, ...] = (
    "c++", "c#", "f#", ".net", "asp.net", "node.js", "next.js", "vue.js",
    "ci/cd", "objective-c", "c/c++", "tcp/ip", "i/o", "r&d", "a/b",
)

_WORD = r"[a-z0-9]+(?:\.[a-z0-9]+)*"


def _phrase_pattern(phrases: Iterable[str]) -> str:
    ordered = sorted({p for p in phrases if p}, key=lambda p: (-len(p), p))
    return "|".join(re.escape(p) for p in ordered)


def _compile(aliases: Mapping[str, str]) -> re.Pattern[str]:
    """A scanner that prefers known punctuated/multi-word terms over bare words."""
    protected = [key for key in aliases if " " in key or not key.isalnum()]
    protected += list(SYMBOL_TERMS)
    phrases = _phrase_pattern(protected)
    return re.compile(f"(?:{phrases})|{_WORD}" if phrases else _WORD)


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace — the form aliases are keyed by."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def tokenize(text: str, aliases: Mapping[str, str] | None = None) -> list[str]:
    """Lowercase, split, and map every known surface form to its canonical token.

    ``aliases`` maps a surface form (any case) to the canonical token it stands
    for. Forms with spaces or punctuation are matched before the generic word
    split, so ``"container orchestration"`` folds to one token rather than two.
    """
    folded = {normalise(k): str(v) for k, v in (aliases or {}).items() if str(k).strip()}
    scanner = _compile(folded)
    tokens: list[str] = []
    for match in scanner.finditer(normalise(text)):
        token = match.group(0)
        if not token:
            continue
        tokens.append(folded.get(token, token))
    return tokens


class BM25:
    """Okapi BM25 over pre-tokenised documents.

    ``score`` returns one number per document, in the order the documents were
    given, so callers can rank with ``argsort`` without keeping a mapping.
    """

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.docs = [list(doc) for doc in docs]
        self.n = len(self.docs)
        self.lengths = np.array([len(doc) for doc in self.docs], dtype=float)
        self.avg_len = float(self.lengths.mean()) if self.n and self.lengths.sum() else 0.0

        self.freqs: list[dict[str, int]] = []
        doc_freq: dict[str, int] = {}
        for doc in self.docs:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.freqs.append(counts)
            for token in counts:
                doc_freq[token] = doc_freq.get(token, 0) + 1
        self.doc_freq = doc_freq
        self.idf = {
            token: math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for token, df in doc_freq.items()
        }

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        """BM25 score of every document against ``query_tokens``."""
        scores = np.zeros(self.n, dtype=float)
        if not self.n or self.avg_len <= 0:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self.lengths / self.avg_len)
        for token in dict.fromkeys(query_tokens):
            idf = self.idf.get(token)
            if idf is None:
                continue
            tf = np.array([counts.get(token, 0) for counts in self.freqs], dtype=float)
            scores += idf * (tf * (self.k1 + 1.0)) / (tf + norm)
        return scores

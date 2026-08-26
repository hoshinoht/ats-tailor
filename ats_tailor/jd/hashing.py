"""One identity for a posting, shared by the store, the runs and the labels."""

from __future__ import annotations

import hashlib
import re


def jd_hash(text: str) -> str:
    """sha256 over the posting's normalised text.

    Whitespace is squashed and the ends trimmed before hashing, so a posting
    copied twice with a different amount of trailing blank lines is recognised
    as the same posting — which it is.
    """
    normalised = re.sub(r"\s+", " ", str(text)).strip()
    return hashlib.sha256(normalised.encode()).hexdigest()

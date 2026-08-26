"""One interface between the pipeline and whatever model is available.

No step of the v2 pipeline calls a model. It calls a :class:`Provider`, which is
either a local backend (``ollama``, ``lmstudio``, ``mlx``), nothing at all
(``none`` — every method returns ``None`` and the deterministic paths run), or
an agent supplying results from outside (``external``, used by the MCP server
in Phase 4, where the driving agent *is* the model).

Returning ``None`` is a first-class answer. It means "I have nothing for this",
and every caller already has a deterministic path for that case, so a laptop
with no model running produces the same resume it did in Phase 3b rather than
an error.

Selection is by name, or ``auto``: a *cheap* probe of mlx → lmstudio → ollama
with ``config.LLM_PROBE_TIMEOUT`` seconds each, falling back to ``none`` with a
single printed warning. The probe never blocks on an unreachable host, and it
treats a server that is up but serving no models as absent — which it is, for
every purpose this interface has.
"""

from __future__ import annotations

import sys
from typing import Protocol, runtime_checkable

from .. import config
from ..jd.models import JDProfile
from . import backends

#: Tokens an extraction answer may take. A posting's worth of JSON is long, and
#: v1's 512 would truncate it mid-object.
EXTRACT_MAX_TOKENS = 2048
#: Context window an extraction needs: the prompt carries the whole posting.
EXTRACT_NUM_CTX = 16384
RERANK_MAX_TOKENS = 256


@runtime_checkable
class Provider(Protocol):
    """What the pipeline may ask a model for."""

    name: str

    def extract(
        self, jd_text: str, categories: list[str], prompt_version: str
    ) -> JDProfile | None:
        """The posting as a profile, or ``None`` to fall back to the parser."""
        ...

    def rerank(self, requirement: str, claims: list[str]) -> list[float] | None:
        """One score per claim, higher is more relevant, or ``None``."""
        ...

    def propose_variant(
        self, bullet: str, archetype: str, requirements: list[str]
    ) -> str | None:
        """Phase 5. Every provider returns ``None`` today."""
        ...


class NoneProvider:
    """The provider that is not there. Every answer is ``None``."""

    name = "none"

    def healthy(self) -> bool:
        return True

    def extract(self, jd_text, categories, prompt_version):
        return None

    def rerank(self, requirement, claims):
        return None

    def propose_variant(self, bullet, archetype, requirements):
        return None


class ExternalProvider:
    """Results supplied from outside — an agent, a file, or the store.

    Never calls a model. The MCP server (Phase 4) constructs one of these with
    the profile the driving agent produced, and the pipeline cannot tell the
    difference; ``ats-tailor jd show --provider external`` uses the same class
    to replay whatever the store already holds.
    """

    name = "external"

    def __init__(self, profile: JDProfile | dict | None = None):
        self.profile = (
            JDProfile.from_json(profile) if isinstance(profile, dict) else profile
        )

    def healthy(self) -> bool:
        return True

    def extract(self, jd_text, categories, prompt_version):
        if self.profile is None:
            return None
        return self.profile.model_copy(
            update={"provider": self.name, "prompt_version": prompt_version}
        )

    def rerank(self, requirement, claims):
        return None

    def propose_variant(self, bullet, archetype, requirements):
        return None


class _ModelProvider:
    """Shared behaviour of the three local backends.

    Subclasses supply a name, a health probe and a way to generate text; the
    prompt, the JSON parsing and the validation are the same whichever process
    the weights are living in.
    """

    name = "model"

    def __init__(self, model: str | None = None, alias_map=None):
        self.model = model or config.LLM_MODEL
        #: Lowercased surface form → canonical index term, so the posting's
        #: "K8s" comes back as the vocabulary the claims are tagged with.
        self.alias_map = dict(alias_map or {})

    # ── to be provided ──
    def healthy(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, label: str, max_tokens: int, num_ctx=None) -> str:
        raise NotImplementedError

    # ── the interface ──
    def extract(self, jd_text, categories, prompt_version):
        from ..jd.extract import parse_profile, render_extract_prompt

        prompt = render_extract_prompt(jd_text, categories)
        try:
            raw = self.generate(prompt, "extract", EXTRACT_MAX_TOKENS, EXTRACT_NUM_CTX)
        except Exception as exc:  # a model that fails is a model that is absent
            print(f"  Warning: {self.name} extraction failed ({exc}); "
                  f"falling back to the deterministic parser", file=sys.stderr)
            return None
        payload = backends._parse_json_object(raw)
        if payload is None:
            print(f"  Warning: {self.name} returned no JSON object; "
                  f"falling back to the deterministic parser", file=sys.stderr)
            return None
        return parse_profile(payload, self.alias_map, provider=self.name,
                             prompt_version=prompt_version)

    def rerank(self, requirement, claims):
        claims = list(claims)
        if not claims:
            return []
        listed = "\n".join(f"{i}. {text}" for i, text in enumerate(claims, start=1))
        prompt = (
            "Score how well each numbered statement answers the requirement.\n"
            "0 means unrelated, 10 means it answers it directly.\n"
            "Answer with JSON only: {\"scores\": [n, n, ...]} — one number per "
            "statement, in order.\n\n"
            f"Requirement: {requirement}\n\nStatements:\n{listed}\n"
        )
        try:
            raw = self.generate(prompt, "rerank", RERANK_MAX_TOKENS)
        except Exception as exc:
            print(f"  Warning: {self.name} rerank failed ({exc})", file=sys.stderr)
            return None
        payload = backends._parse_json_object(raw)
        scores = payload.get("scores") if isinstance(payload, dict) else None
        if not isinstance(scores, list) or len(scores) != len(claims):
            return None
        try:
            return [float(score) for score in scores]
        except (TypeError, ValueError):
            return None

    def propose_variant(self, bullet, archetype, requirements):
        return None  # Phase 5


class OllamaProvider(_ModelProvider):
    name = "ollama"

    def healthy(self) -> bool:
        models = backends.ollama_models()
        return bool(models)

    def generate(self, prompt, label, max_tokens, num_ctx=None):
        return backends._ollama_generate(
            self.model, prompt, label=label, temperature=config.LLM_TEMPERATURE_P1,
            max_tokens=max_tokens, num_ctx=num_ctx,
        )


class LMStudioProvider(_ModelProvider):
    name = "lmstudio"

    def healthy(self) -> bool:
        models = backends.lmstudio_models()
        return bool(models)

    def generate(self, prompt, label, max_tokens, num_ctx=None):
        return backends._lmstudio_generate(
            self.model, prompt, label=label, temperature=config.LLM_TEMPERATURE_P1,
            max_tokens=max_tokens,
        )


class MLXProvider(_ModelProvider):
    name = "mlx"

    def __init__(self, model: str | None = None, alias_map=None):
        super().__init__(model or config.MLX_MODEL, alias_map)

    def healthy(self) -> bool:
        return backends.mlx_available()

    def generate(self, prompt, label, max_tokens, num_ctx=None):
        return backends._mlx_generate(
            self.model, prompt, label=label, temperature=config.LLM_TEMPERATURE_P1,
            max_tokens=max_tokens,
        )


#: ``auto`` tries these in order. MLX first because it is in-process (no server
#: to be running), then the two HTTP backends.
AUTO_ORDER = ("mlx", "lmstudio", "ollama")

_CLASSES = {
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider,
    "mlx": MLXProvider,
    "none": NoneProvider,
    "external": ExternalProvider,
}


def get_provider(name: str | None = None, alias_map=None, profile=None, quiet=False):
    """The provider called ``name``; ``auto`` probes and falls back to ``none``.

    ``alias_map`` (surface form → skill id) lets an extracting provider
    normalise the posting's vocabulary against the index. ``profile`` is only
    read by ``external``.
    """
    requested = str(name or config.LLM_PROVIDER or "auto").strip().lower()
    say = (lambda *_a, **_k: None) if quiet else (
        lambda message: print(message, file=sys.stderr))

    if requested == "external":
        return ExternalProvider(profile)
    if requested == "none":
        return NoneProvider()
    if requested in _CLASSES and requested != "auto":
        return _CLASSES[requested](alias_map=alias_map)
    if requested != "auto":
        raise ValueError(
            f"unknown provider {name!r}; expected one of: {', '.join(config.PROVIDERS)}"
        )

    for candidate in AUTO_ORDER:
        provider = _CLASSES[candidate](alias_map=alias_map)
        try:
            healthy = provider.healthy()
        except Exception:
            healthy = False
        if healthy:
            return provider
    say("  Warning: no LLM provider is reachable (tried "
        f"{', '.join(AUTO_ORDER)}); continuing with the deterministic parser")
    return NoneProvider()


def provider_names() -> tuple[str, ...]:
    return config.PROVIDERS

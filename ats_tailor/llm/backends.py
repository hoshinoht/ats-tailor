"""Talking to a local model: Ollama, LM Studio, MLX, and the JSON helpers.

Moved out of the old flat ``ats_tailor/llm.py`` in Phase 3c without a change in
behaviour — :mod:`ats_tailor.llm.expand` (v1's keyword expansion) and
:mod:`ats_tailor.llm.provider` (the v2 interface) are the two callers, and both
reach a model only through here.

Two timeouts live in this file and they are deliberately far apart:
``config.LLM_TIMEOUT`` is how long a *generation* may take (a 20B model on a
laptop is measured in minutes), while ``config.LLM_PROBE_TIMEOUT`` is how long
a *health probe* may take. ``get_provider("auto")`` asks three hosts in a row
whether they exist, and a machine running none of them must not wait ten
minutes to be told so.
"""

import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .. import config


def _think_option(value=None):
    """``think`` for ``/api/generate``: a reasoning *budget*, not a boolean.

    A reasoning model ignores ``think: false`` — gpt-oss reasons anyway, and
    Ollama returns that reasoning in a separate ``thinking`` field while it
    still counts against ``num_predict``. On a full job posting the whole
    budget went to reasoning and ``response`` came back empty, whatever the
    budget was (measured: 4096 tokens, 20k characters of ``thinking``, nothing
    to parse). The string efforts *are* honoured, so ``low`` is the setting
    that makes the answer arrive at all.

    Booleans still pass through, because the older models this backend serves
    take one and would reject an effort string.
    """
    raw = str(config.OLLAMA_THINK if value is None else value).strip().lower()
    if raw in ("false", "0", "no", "off", "none", ""):
        return False
    if raw in ("true", "1", "yes", "on"):
        return True
    return raw  # "low" | "medium" | "high" — Ollama validates it, not us


def _post_json(url, body, timeout):
    req = Request(url, data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _ollama_generate(model_name, prompt, label="LLM", temperature=None,
                     max_tokens=512, num_ctx=None):
    """Send a single prompt to ollama and return the raw response text.

    Only ``response`` is the answer. ``thinking`` is the model's scratchpad and
    is deliberately dropped: handing a reasoning transcript to the JSON parser
    downstream would let a discarded draft become the extraction.
    """
    opts = {"num_predict": int(max_tokens),
            "num_ctx": int(num_ctx or config.LLM_NUM_CTX)}
    if temperature is not None:
        opts["temperature"] = temperature
    body = {"model": model_name, "prompt": prompt, "stream": False,
            "options": opts, "think": _think_option()}
    url = f"{config.OLLAMA_URL}/api/generate"
    try:
        data = _post_json(url, body, config.LLM_TIMEOUT)
    except HTTPError as exc:
        # A model with no thinking capability rejects the field outright; the
        # prompt is still perfectly answerable without it.
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")
        except Exception:
            pass
        if "think" not in detail.lower():
            raise
        body.pop("think", None)
        data = _post_json(url, body, config.LLM_TIMEOUT)

    raw = data.get("response") or ""
    p_tok = data.get("prompt_eval_count", 0)
    g_tok = data.get("eval_count", 0)
    print(f"  {label} tokens: {p_tok} prompt + {g_tok} generated "
          f"= {p_tok + g_tok} / {config.LLM_NUM_CTX} ctx",
          file=sys.stderr)
    if not raw and data.get("thinking"):
        print(f"  Warning: {label} spent its whole budget reasoning "
              f"(done_reason={data.get('done_reason')}) and answered nothing; "
              f"lower ATS_OLLAMA_THINK or raise the token budget",
              file=sys.stderr)
    return raw


# Module-level cache for MLX model/tokenizer (avoid reloading between passes)
_mlx_model_cache = {}


def _mlx_generate(model_name, prompt, label="LLM", temperature=None,
                  max_tokens=512, num_ctx=None):
    """Generate text using mlx-lm. Raises ImportError if mlx-lm is not installed."""
    import mlx_lm

    if model_name not in _mlx_model_cache:
        print(f"  Loading MLX model {model_name}...", file=sys.stderr)
        model, tokenizer = mlx_lm.load(model_name)
        _mlx_model_cache[model_name] = (model, tokenizer)

    model, tokenizer = _mlx_model_cache[model_name]
    kwargs = {"prompt": prompt, "max_tokens": int(max_tokens), "verbose": False}
    if temperature is not None:
        kwargs["temp"] = temperature
    response = mlx_lm.generate(model, tokenizer, **kwargs)
    # Estimate token counts for logging
    p_tok = len(tokenizer.encode(prompt))
    g_tok = len(tokenizer.encode(response))
    print(f"  {label} tokens: {p_tok} prompt + {g_tok} generated",
          file=sys.stderr)
    return response


def _lmstudio_generate(model_name, prompt, label="LLM", temperature=None,
                       max_tokens=2048, num_ctx=None):
    """Send a prompt to LM Studio's OpenAI-compatible API."""
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": temperature if temperature is not None else 0.3}
    if model_name:
        body["model"] = model_name
    payload = json.dumps(body).encode()
    url = f"{config.LMSTUDIO_URL}/v1/chat/completions"
    req = Request(url, data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
        data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        # Prefer separated content (LM Studio reasoning_content setting)
        raw = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not raw and not reasoning:
            raw = ""
        elif not raw and reasoning:
            # reasoning_content not separated — content has everything
            pass
        # Strip <think>…</think> if content wasn't separated
        if "<think>" in raw:
            if "</think>" in raw:
                raw = re.sub(r"<think>.*?</think>\s*",
                             "", raw, flags=re.DOTALL)
            else:
                raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        usage = data.get("usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        g_tok = usage.get("completion_tokens", 0)
        thinking_note = f" ({g_tok - len(raw.split())} thinking)" if reasoning else ""
        print(f"  {label} tokens: {p_tok} prompt + {g_tok} generated{thinking_note}",
              file=sys.stderr)
    return raw


def _generate(model_name, prompt, label="LLM", temperature=None, **kwargs):
    """Dispatch to MLX, LM Studio, or Ollama based on config.LLM_BACKEND.

    ``kwargs`` (``max_tokens``, ``num_ctx``) reach the backend unchanged; the
    defaults are v1's, so an expansion call behaves exactly as it did.
    """
    backend = config.LLM_BACKEND

    if backend == "mlx":
        return _mlx_generate(config.MLX_MODEL, prompt, label, temperature, **kwargs)
    elif backend == "lmstudio":
        return _lmstudio_generate(model_name, prompt, label, temperature, **kwargs)
    elif backend == "ollama":
        return _ollama_generate(model_name, prompt, label, temperature, **kwargs)
    else:  # auto
        try:
            return _mlx_generate(config.MLX_MODEL, prompt, label, temperature, **kwargs)
        except ImportError:
            print("  MLX not available, trying LM Studio",
                  file=sys.stderr)
        except Exception as e:
            print(f"  MLX failed ({e}), trying LM Studio",
                  file=sys.stderr)
        try:
            return _lmstudio_generate(model_name, prompt, label, temperature, **kwargs)
        except (URLError, OSError):
            print("  LM Studio not available, falling back to Ollama",
                  file=sys.stderr)
        except Exception as e:
            print(f"  LM Studio failed ({e}), falling back to Ollama",
                  file=sys.stderr)
        return _ollama_generate(model_name, prompt, label, temperature, **kwargs)


def _parse_json_array(raw):
    """Extract a JSON array from LLM output (may be wrapped in fences/tags)."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        terms = json.loads(match.group())
        if isinstance(terms, list):
            return [str(t) for t in terms if isinstance(t, str)]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ── Health probes ────────────────────────────────────────────────────────────
def _http_json(url, timeout):
    """GET ``url`` and parse the body, or ``None`` for anything that goes wrong.

    Deliberately total: a probe answers one question — "is a model reachable
    here right now" — and every way of failing is the same answer.
    """
    try:
        with urlopen(Request(url), timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def ollama_models(timeout=None):
    """Model names Ollama is serving, or ``None`` when it is not reachable."""
    data = _http_json(f"{config.OLLAMA_URL}/api/tags",
                      config.LLM_PROBE_TIMEOUT if timeout is None else timeout)
    if not isinstance(data, dict):
        return None
    return [str(m.get("name", "")) for m in data.get("models", []) if m.get("name")]


def lmstudio_models(timeout=None):
    """Model ids the OpenAI-compatible endpoint lists, or ``None`` if absent."""
    data = _http_json(f"{config.LMSTUDIO_URL}/v1/models",
                      config.LLM_PROBE_TIMEOUT if timeout is None else timeout)
    if not isinstance(data, dict):
        return None
    return [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]


def mlx_available():
    """Whether ``mlx-lm`` can be imported at all (the model loads lazily)."""
    try:
        import mlx_lm  # noqa: F401
    except Exception:
        return False
    return True


def _parse_json_object(raw):
    """Extract the outermost JSON object from a model's answer.

    Models wrap JSON in fences, prose, or a ``<think>`` block; the object is
    found by scanning braces rather than by a regex, so a nested object inside
    a trailing sentence cannot truncate it.
    """
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.DOTALL)
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None

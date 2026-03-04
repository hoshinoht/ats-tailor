# ats-tailor

Semantic resume tailoring for ATS. Scores your YAML resume index against a job description using sentence embeddings, keyword matching, and recency weighting, then renders a one-page LaTeX resume. All processing is local.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m ats_tailor.tailor \
  --index examples/index \
  --profile examples/profile.yaml \
  --jd examples/jd/sample-backend.txt \
  --company acme \
  --role backend \
  --output /tmp/ats-test
```

| Arg             | Default                      | Description                                      |
| --------------- | ---------------------------- | ------------------------------------------------ |
| `--jd`          | stdin                        | Job description text file                        |
| `--company`     | required                     | Company name                                     |
| `--role`        | required                     | Role name                                        |
| `--index`       | `index/`                     | YAML index directory                             |
| `--profile`     | `profile.yaml` next to index | Profile YAML (name, email, links)                |
| `--output`      | `output/{company}-{role}`    | Output directory                                 |
| `--model`       | `all-MiniLM-L12-v2`          | SentenceTransformer model name                   |
| `--llm`         | off                          | Expand JD keywords via local LLM (see below)     |
| `--llm-backend` | `auto`                       | LLM backend: `auto`, `mlx`, `lmstudio`, `ollama` |
| `--rerank`      | off                          | Re-score top candidates with a cross-encoder     |

## Configuration

All CLI flags can be set via a `.env` file in the ats-tailor repo root. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable                    | Default                       | CLI override    |
| --------------------------- | ----------------------------- | --------------- |
| `ATS_EMBED_MODEL`           | `all-MiniLM-L12-v2`           | `--model`       |
| `ATS_LLM_MODEL`             | *(disabled)*                  | `--llm`         |
| `ATS_LLM_BACKEND`           | `auto`                        | `--llm-backend` |
| `ATS_LLM_NUM_CTX`           | `4096`                        | —               |
| `ATS_LLM_TEMPERATURE`       | `0.3`                         | —               |
| `ATS_LLM_TWO_PASS`          | `false`                       | —               |
| `ATS_LLM_TEMPERATURE_P1`    | `0.2`                         | —               |
| `ATS_LLM_TEMPERATURE_P2`    | `0.5`                         | —               |
| `ATS_LMSTUDIO_URL`          | `http://localhost:1234`       | —               |
| `ATS_MLX_MODEL`             | `mlx-community/Qwen3-8B-4bit` | —               |
| `ATS_RERANK`                | `false`                       | `--rerank`      |
| `ATS_MAX_EXPERIENCE`        | `3`                           | —               |
| `ATS_MAX_PROJECTS`          | `4`                           | —               |
| `ATS_MIN_PROJECTS`          | `4`                           | —               |
| `ATS_MAX_SKILL_LINES`       | `4`                           | —               |
| `ATS_MAX_PROJECT_BULLETS`   | `3`                           | —               |
| `ATS_MAX_EXP_BULLETS`       | `3`                           | —               |
| `ATS_CHARS_PER_BULLET_LINE` | `80`                          | —               |
| `ATS_MAX_PAGE_LINES`        | `72`                          | —               |

CLI flags override env vars; env vars override `.env` file values.

## LLM keyword expansion

Tag overlap scoring uses exact string matching — if the JD says "AI Engineer" it won't match an `ats_tag` like "Deep Learning" because the literal string isn't there. The `--llm` flag sends the JD to a local [ollama](https://ollama.com) model to infer implied technical keywords, which are appended to the matching pool before scoring.

```bash
# use default model (qwen2.5-coder:7b)
python -m ats_tailor.tailor --llm ...

# use a specific model
python -m ats_tailor.tailor --llm qwen2.5-coder:1.5b ...

# omit flag entirely to skip expansion (default)
python -m ats_tailor.tailor ...
```

**Default model:** `qwen3.5:9b` (~6.6 GB). Context window defaults to 4096 (`ATS_LLM_NUM_CTX`).

**Category-aware prompting.** The candidate's skill category names are injected into the prompt so the LLM prioritises terms the candidate can actually match, while still including important terms outside those areas.

**Two-pass mode** (`ATS_LLM_TWO_PASS=true`). Uses two LLM calls: pass 1 infers the 3-5 most relevant domains from the candidate's skill categories, pass 2 expands keywords scoped to those domains. ~2x latency but more targeted results on generic JDs. Falls back to single-pass automatically on failure. Each pass has its own temperature setting (`ATS_LLM_TEMPERATURE_P1` for domain classification, `ATS_LLM_TEMPERATURE_P2` for keyword expansion).

### Backends

Three backends are supported (`--llm-backend` / `ATS_LLM_BACKEND`):

| Backend    | How it works                                                   | Setup                                     |
| ---------- | -------------------------------------------------------------- | ----------------------------------------- |
| `mlx`      | In-process via [mlx-lm](https://github.com/ml-explore/mlx-lm)  | `pip install mlx-lm`                      |
| `lmstudio` | HTTP to [LM Studio](https://lmstudio.ai) OpenAI-compatible API | Start server: `lms server start`          |
| `ollama`   | HTTP to [Ollama](https://ollama.com) REST API                  | `ollama serve` + `ollama pull qwen3.5:9b` |
| `auto`     | Try MLX → LM Studio → Ollama, skip on failure                  | Any of the above                          |

**LM Studio notes:**
- Default URL: `http://localhost:1234` (override with `ATS_LMSTUDIO_URL`)
- Uses whatever model is loaded in the app; the `--llm` model name is sent as a hint
- For thinking models (e.g. Qwen3.5), disable reasoning in LM Studio's server settings for faster, cleaner output

If the backend is unreachable the pipeline prints a warning and continues without expansion. Token usage is logged to stderr after each LLM call.

Expanded terms appear in the console output and in the `match_report.md` under **LLM-Expanded Keywords**.

## Output

| File              | Description                                     |
| ----------------- | ----------------------------------------------- |
| `resume.tex`      | One-page LaTeX resume (compile with `pdflatex`) |
| `prompt.md`       | LLM prompt for optional bullet rewriting        |
| `match_report.md` | Similarity scores and rankings                  |
| `match_viz.png`   | Visual score breakdown                          |

## Index format

See `examples/index/` for the expected YAML structure:

- **skills.yaml** -- categories with `ats_keywords` and `proficiency`
- **projects.yaml** -- projects with `tech_stack`, `responsibilities`, `ats_tags`
- **experience.yaml** -- roles with `bullets`, `skills_used`, `ats_tags`; education entries
- **certifications.yaml** -- certs with `issuer` and `ats_keywords`

To start with your own data, copy `examples/`, fill in your details, and run against real job descriptions.

## Scoring

```
score(i) = (semantic(i) + keyword_bonus(i) + tag_overlap(i)) * recency(i)
```

- **Semantic** -- max-pooled cosine similarity between JD chunks and item embedding
- **Keyword bonus** -- `0.06 * log2(1 + matches)` (logarithmic, no hard cap)
- **Tag overlap** -- proportion of item's ATS tags found in JD, scaled to max +0.25
- **Recency** -- x1.10 current, x1.05 <12mo, x1.02 <24mo, x1.00 older

See [docs/algorithm.pdf](docs/algorithm.pdf) for the full writeup.

## License

GNU GPLv3 (see [LICENSE](LICENSE))

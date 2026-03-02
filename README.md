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

| Arg | Default | Description |
|-----|---------|-------------|
| `--jd` | stdin | Job description text file |
| `--company` | required | Company name |
| `--role` | required | Role name |
| `--index` | `index/` | YAML index directory |
| `--profile` | `profile.yaml` next to index | Profile YAML (name, email, links) |
| `--output` | `output/{company}-{role}` | Output directory |
| `--llm` | off | Expand JD keywords via local LLM (see below) |

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

**Default model:** `qwen2.5-coder:7b` (~4.7 GB download, ~5s per JD). Produces 20-25 focused technical terms with minimal noise.

**Choosing a model:** Any ollama model works. Smaller models like `qwen2.5-coder:1.5b` (~1 GB, ~2s) are faster but produce noisier output (soft skills, generic terms). Thinking models like `qwen3` tend to be slow due to extended reasoning; non-thinking instruction models work best for this structured extraction task.

**Requirements:** [ollama](https://ollama.com) must be installed and running (`ollama serve`). Pull the model before first use:

```bash
ollama pull qwen2.5-coder:7b
```

If ollama is unreachable the pipeline prints a warning and continues without expansion.

Expanded terms appear in the console output and in the `match_report.md` under **LLM-Expanded Keywords**.

## Output

| File | Description |
|------|-------------|
| `resume.tex` | One-page LaTeX resume (compile with `pdflatex`) |
| `prompt.md` | LLM prompt for optional bullet rewriting |
| `match_report.md` | Similarity scores and rankings |
| `match_viz.png` | Visual score breakdown |

## Index format

See `examples/index/` for the expected YAML structure:

- **skills.yaml** -- categories with `ats_keywords` and `proficiency`
- **projects.yaml** -- projects with `tech_stack`, `responsibilities`, `ats_tags`
- **experience.yaml** -- roles with `bullets`, `skills_used`, `ats_tags`; education entries
- **certifications.yaml** -- certs with `issuer` and `ats_keywords`

To start with your own data, copy `examples/`, fill in your details, and run against real job descriptions.

## Editor

Browser-based editor for the YAML index. Provides structured editing with live validation; changes write directly to disk.

### Docker

```bash
docker compose up --build    # serves at localhost:8070
```

The compose file mounts `../index` into the container. For standalone use, override the volume:

```bash
docker compose run -v /path/to/index:/data/index editor
```

### Local

Requires Go 1.22+.

```bash
cd editor && go build -o yamledit .
./yamledit                       # auto-detects <git-root>/index
./yamledit -dir /path/to/index   # explicit path
./yamledit -port 9090            # custom port (default 8070)
./yamledit -no-browser           # skip auto-open
```

## Scoring

```
score(i) = (semantic(i) + keyword_bonus(i) + tag_overlap(i)) * recency(i)
```

- **Semantic** -- max-pooled cosine similarity between JD chunks and item embedding
- **Keyword bonus** -- +0.06 per exact ATS keyword match (capped at +0.20)
- **Tag overlap** -- proportion of item's ATS tags found in JD, scaled to max +0.25
- **Recency** -- x1.10 current, x1.05 <12mo, x1.02 <24mo, x1.00 older

See [docs/algorithm.pdf](docs/algorithm.pdf) for the full writeup.

## License

MIT

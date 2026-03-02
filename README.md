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

# ats-tailor

Semantic resume tailoring for ATS — match your YAML index against any job description.

## How it works

1. You maintain a YAML index of your skills, projects, experience, and certifications
2. Point the tool at a job description
3. It uses sentence embeddings ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) + keyword matching + recency weighting to score every item against the JD
4. The best content is selected, budget-fitted to one A4 page, and rendered into LaTeX

All processing is local — no API calls.

## Output

| File | Purpose |
|------|---------|
| `resume.tex` | 1-page LaTeX resume, compile with `pdflatex` |
| `prompt.md` | LLM prompt for optional bullet rewriting (paste into Claude/ChatGPT) |
| `match_report.md` | Full similarity scores and rankings |
| `match_viz.png` | 4-panel visual score breakdown |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m ats_tailor.tailor \
  --profile examples/profile.yaml \
  --index examples/index \
  --jd examples/jd/sample-backend.txt \
  --company acme \
  --role backend \
  --output /tmp/ats-test
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--jd` | stdin | Path to job description text file |
| `--company` | required | Company name |
| `--role` | required | Role name |
| `--index` | `index/` | Path to YAML index directory |
| `--profile` | `profile.yaml` next to index dir | Path to profile YAML with personal info |
| `--output` | `output/{company}-{role}` | Output directory |

## Your own data

1. Copy `examples/` as a starting point
2. Edit `profile.yaml` with your real name, email, phone, and links
3. Populate `index/` with your skills, projects, experience, and certifications
4. Run against real job descriptions

### Index schema

See `examples/index/` for the expected YAML structure. Each file follows a consistent pattern:

- **skills.yaml** — categories → skills with `ats_keywords` and `proficiency`
- **projects.yaml** — projects with `tech_stack`, `responsibilities`, `ats_tags`
- **experience.yaml** — roles with `bullets`, `skills_used`, `ats_tags`; education entries
- **certifications.yaml** — certs with `issuer` and `ats_keywords`

## Scoring

Each item gets a hybrid score:

```
score(i) = (semantic(i) + keyword_bonus(i)) × recency(i)
```

- **Semantic**: max-pooled cosine similarity between JD chunks and item embedding
- **Keyword bonus**: +0.06 per exact ATS keyword match in JD (capped at +0.20)
- **Recency**: ×1.10 current, ×1.05 <12mo, ×1.02 <24mo, ×1.00 older

## Algorithm details

See [docs/algorithm.pdf](docs/algorithm.pdf) for a full writeup of the scoring formula, selection algorithm, page budget enforcement, and design evolution.

## License

MIT

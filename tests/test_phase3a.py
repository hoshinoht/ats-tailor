"""Phase 3a: the package boundary, the index→claims adapter, and the v2 pipeline.

Everything needing the embedding model or LaTeX is gated behind ``ATS_SMOKE=1``,
like the Phase 1 and 2 suites; the rest runs on the example index and on
synthetic postings so the default suite stays fast.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from ats_match import Claim, Requirement, Selection, Step

from ats_tailor.adapt import (
    PAGE_TERMS_ATTR,
    alias_map,
    claims_from_index,
    render_context,
    section_rules,
    skill_claims,
)
from ats_tailor.atscheck import index_term_forms
from ats_tailor.index_io import load_index_v2
from ats_tailor.jd.requirements import requirements_from_jd
from ats_tailor.models import Bullet, Index, Period, Project, Role, Skill, SkillCategory

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "index"
MATCH_PKG = REPO / "packages" / "ats-match"
SMOKE = os.environ.get("ATS_SMOKE") in {"1", "true", "yes"}
smoke = pytest.mark.skipif(not SMOKE, reason="set ATS_SMOKE=1 to run")


@pytest.fixture(scope="module")
def index() -> Index:
    return load_index_v2(EXAMPLES)


# ── 1. Package boundary ───────────────────────────────────────────────────────
def test_importing_ats_match_never_pulls_in_ats_tailor():
    """The core is a library the application uses, never the other way round."""
    code = "import ats_match, sys; print(json.dumps([m for m in sys.modules if m.startswith('ats_tailor')]))"
    result = subprocess.run(
        [sys.executable, "-c", "import json\n" + code],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_the_core_has_no_application_vocabulary():
    """No file under ``packages/ats-match`` names what the claims describe.

    ``pyproject.toml`` is exempt for the one word it cannot avoid: its
    ``[project]`` table is the packaging format's, not ours.
    """
    forbidden = re.compile(r"\b(resume|project|role|latex|yaml)", re.IGNORECASE)
    offenders = []
    for path in sorted(MATCH_PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if forbidden.search(line):
                offenders.append(f"{path.relative_to(MATCH_PKG)}:{i}: {line.strip()}")
    assert not offenders, "application vocabulary leaked into the core:\n" + "\n".join(offenders)


def test_the_core_declares_only_numpy():
    text = (MATCH_PKG / "pyproject.toml").read_text()
    dependencies = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
    assert dependencies
    assert [d.strip().strip('"') for d in dependencies.group(1).split(",") if d.strip()] == ["numpy"]


# ── 2. Adapter: index → claims ────────────────────────────────────────────────
def test_one_claim_per_visible_bullet(index):
    claims = claims_from_index(index)
    expected = sum(
        len([b for b in item.bullets if b.text.strip()])
        for item in [*[r for r in index.roles if not r.hidden], *index.projects]
    )
    assert len(claims) == expected
    assert len({c.id for c in claims}) == len(claims)
    assert {c.section for c in claims} <= {"role", "project"}


def test_hidden_entries_contribute_no_claims():
    index = Index(
        roles=[
            Role(id="shown", company="Acme", title="Engineer", type="internship",
                 period=Period(start="2025-01", end="2025-06"),
                 bullets=[Bullet(id="shown.b1", kind="responsibility", text="Did a thing")]),
            Role(id="hidden", company="Acme", title="Intern", type="internship",
                 period=Period(start="2024-01", end="2024-06"), hidden=True),
        ]
    )
    assert [c.group for c in claims_from_index(index)] == ["shown"]


def test_claim_terms_include_every_alias_of_a_referenced_skill():
    index = _tiny_index()
    claim = next(c for c in claims_from_index(index) if c.id == "p1.r1")
    lowered = {t.lower() for t in claim.terms}
    assert {"kubernetes", "k8s", "container orchestration"} <= lowered
    # keywords of the item come along too
    assert "platform engineering" in lowered


def test_page_terms_hold_only_what_the_bullet_actually_says():
    """The keyword check reads the page; a tag the words never use is not on it."""
    index = _tiny_index()
    claims = {c.id: c for c in claims_from_index(index)}
    spoken = {t.lower() for t in claims["p1.r1"].attrs[PAGE_TERMS_ATTR]}
    silent = {t.lower() for t in claims["p1.i1"].attrs[PAGE_TERMS_ATTR]}
    assert "k8s" in spoken and "kubernetes" in spoken  # the bullet says "K8s"
    assert "kubernetes" not in silent                  # this one only tags it


def test_claim_cost_counts_rendered_lines():
    index = _tiny_index()
    claims = {c.id: c for c in claims_from_index(index, chars_per_line=20)}
    assert claims["p1.r1"].cost == -(-len(claims["p1.r1"].text) // 20)
    assert all(c.cost >= 1 for c in claims.values())


def test_skill_claims_are_free_and_grouped_by_category(index):
    claims = skill_claims(index)
    assert claims and all(c.cost == 0 and c.section == "skill" for c in claims)
    assert {c.group for c in claims} == {c.category for c in index.skills}


def test_familiar_skills_wait_to_be_named():
    index = _tiny_index()
    unnamed = {c.attrs["skill_id"] for c in skill_claims(index)}
    assert "terraform" not in unnamed
    named = {
        c.attrs["skill_id"]
        for c in skill_claims(
            index, [Requirement(id="r1", text="Terraform please", kind="must",
                                terms=("Terraform",))]
        )
    }
    assert "terraform" in named


def test_alias_map_points_every_form_at_a_skill(index):
    """A form can be listed by two skills; the one it *names* wins, the rest resolve."""
    mapping = alias_map(index)
    ids = {s.id for s in index.all_skills()}
    for skill in index.all_skills():
        assert mapping[skill.name.lower()] == skill.id
        for form in skill.surface_forms():
            assert mapping[form.lower()] in ids


def test_section_rules_come_from_the_page_budget_settings():
    from ats_tailor import config

    rules = section_rules()
    assert rules["role"].max_groups == config.MAX_EXPERIENCE
    assert rules["project"].min_claims == config.MIN_PROJECT_BULLETS
    assert rules["project"].required_attr == "impact"
    assert rules["skill"].max_groups == config.MAX_SKILL_LINES
    assert rules["role"].group_cost < rules["project"].group_cost  # no tech line


# ── 3. Adapter: selection → template context ──────────────────────────────────
def test_render_context_keeps_bullet_order_and_chronology(index):
    claims = claims_from_index(index) + skill_claims(index)
    by_id = {c.id: c for c in claims}
    picked = [c.id for c in claims if c.section != "skill"][::-1]  # deliberately reversed
    selection = Selection(
        claim_ids=picked,
        trace=[Step(claim_id=cid, gain=1.0) for cid in picked],
    )
    experience, entries, _skill_lines = render_context(selection, index, claims)

    order = {
        b.text: i
        for item in [*index.roles, *index.projects]
        for i, b in enumerate(item.bullets)
    }
    assert by_id  # the reversed pick order is the thing being undone
    for context in [*experience, *entries]:
        positions = [order[text] for text in context["bullets"]]
        assert positions == sorted(positions), context["id"]

    from ats_tailor.loaders import parse_start_date

    for group in (experience, entries):
        dates = [parse_start_date(c["period"]) for c in group]
        assert dates == sorted(dates, reverse=True)


def test_render_context_produces_what_the_template_consumes(index):
    claims = claims_from_index(index) + skill_claims(index)
    selection = Selection(
        claim_ids=[c.id for c in claims],
        trace=[Step(claim_id=c.id, gain=1.0) for c in claims],
    )
    experience, entries, skill_lines = render_context(selection, index, claims)
    assert experience and entries and skill_lines
    assert {"company", "title", "period", "bullets"} <= set(experience[0])
    assert {"name", "tagline", "period", "tech_line", "bullets"} <= set(entries[0])
    assert set(skill_lines[0]) == {"category", "skills"}


# ── 4. Requirement parser ─────────────────────────────────────────────────────
CPF_STYLE_JD = """\
Site Reliability Engineer Intern

What we are looking for
Containerization knowledge: a solid understanding of Kubernetes and Docker.
CI/CD Awareness: familiarity with Continuous Integration pipelines.
Able to explain technical ideas in a simple way.

Nice to have
Some experience with Terraform or other Infrastructure as Code tooling.
Interest in Python scripting for automation.

Benefits
Free parking.
"""


def test_requirements_are_tiered_by_the_heading_above_them(index):
    forms = index_term_forms(index)
    requirements = requirements_from_jd(CPF_STYLE_JD, forms)
    by_text = {r.text: r for r in requirements}

    containers = next(r for t, r in by_text.items() if t.startswith("Containerization"))
    assert containers.kind == "must" and containers.weight == 1.0
    terraform = next(r for t, r in by_text.items() if t.startswith("Some experience"))
    assert terraform.kind == "nice"
    assert all(r.id.startswith("req-") for r in requirements)
    assert len({r.id for r in requirements}) == len(requirements)


def test_requirement_terms_are_the_index_vocabulary_the_line_names(index):
    forms = index_term_forms(index)
    requirements = requirements_from_jd(CPF_STYLE_JD, forms)
    containers = next(r for r in requirements if r.text.startswith("Containerization"))
    assert any("kubernetes" in t.lower() for t in containers.terms)


def test_lines_with_neither_vocabulary_nor_a_requirement_verb_are_dropped(index):
    requirements = requirements_from_jd(CPF_STYLE_JD, index_term_forms(index))
    assert not any(r.text.startswith("Free parking") for r in requirements)


def test_short_and_wordless_lines_are_dropped(index):
    text = "Requirements\n123 456 789 000\nAble to work with a team of engineers.\n"
    requirements = requirements_from_jd(text, index_term_forms(index))
    assert [r.text for r in requirements] == ["Able to work with a team of engineers."]


def test_gap_terms_become_requirement_terms(index):
    requirements = requirements_from_jd(CPF_STYLE_JD, index_term_forms(index), ["Terraform"])
    terraform = next(r for r in requirements if r.text.startswith("Some experience"))
    assert "Terraform" in terraform.terms


def test_the_sample_posting_parses_into_weighted_requirements(index):
    jd = (REPO / "examples" / "jd" / "sample-backend.txt").read_text()
    requirements = requirements_from_jd(jd, index_term_forms(index))
    assert len(requirements) >= 5
    assert {r.kind for r in requirements} <= {"must", "nice", "responsibility"}
    assert any(r.kind == "must" for r in requirements)
    assert any(r.terms for r in requirements)


def test_an_identical_line_under_two_headings_takes_the_stronger_tier(index):
    text = (
        "Nice to have\nExperience with Kubernetes in production.\n"
        "Requirements\nExperience with Kubernetes in production.\n"
    )
    requirements = requirements_from_jd(text, index_term_forms(index))
    assert len(requirements) == 1
    assert requirements[0].kind == "must"


SOFT_JD = """\
What we are looking for
A solid understanding of Kubernetes and containerization.
Works well independently and with others, takes initiative.

Nice to have
Able to explain technical ideas in a simple way.
"""


def test_a_requirement_that_names_nothing_is_soft(index):
    """It stays in the list — it is a real ask — but it stops outbidding the rest."""
    requirements = requirements_from_jd(SOFT_JD, index_term_forms(index))
    named = next(r for r in requirements if r.text.startswith("A solid understanding"))
    soft = next(r for r in requirements if r.text.startswith("Works well"))
    nice_soft = next(r for r in requirements if r.text.startswith("Able to explain"))

    assert named.terms and named.weight == 1.0
    assert not soft.terms and soft.weight == 0.25
    # the soft rule ignores the heading: a nice-to-have that names nothing is
    # weighted the same as a must-have that names nothing
    assert not nice_soft.terms and nice_soft.weight == 0.25
    assert soft.kind == "must" and nice_soft.kind == "nice"


def test_the_soft_weight_is_configurable(index):
    requirements = requirements_from_jd(
        SOFT_JD, index_term_forms(index), soft_weight=0.1
    )
    soft = next(r for r in requirements if r.text.startswith("Works well"))
    assert soft.weight == 0.1


def test_skill_rule_carries_the_gain_floor():
    from ats_tailor import config

    rules = section_rules()
    assert rules["skill"].min_gain == config.SKILL_MIN_GAIN
    assert rules["role"].min_gain is None and rules["project"].min_gain is None


def test_pinned_skill_groups_are_parsed_from_the_setting():
    from ats_tailor import adapt

    class _Settings:
        PIN_SKILL_GROUPS = " Programming Languages , Databases ,, "

    assert adapt.pinned_skill_groups(_Settings) == {"Programming Languages", "Databases"}

    class _Empty:
        PIN_SKILL_GROUPS = ""

    assert adapt.pinned_skill_groups(_Empty) == set()


def test_the_default_pin_keeps_a_languages_line(index):
    """v1 always rendered one; a resume without it reads as incomplete."""
    from ats_tailor.adapt import pinned_skill_groups

    assert "Programming Languages" in pinned_skill_groups()
    assert "Programming Languages" in {c.group for c in skill_claims(index)}


def test_header_claims_carry_the_entry_line(index):
    from ats_tailor.adapt import header_claims

    headers = {c.group: c for c in header_claims(index)}
    assert set(headers) == {i.id for _s, i in _visible(index)}
    for claim in headers.values():
        assert claim.attrs["header"] is True
        assert claim.cost == 0.0
    role = next(iter(index.roles))
    assert headers[role.id].text == f"{role.title} -- {role.company}"
    entry = next(iter(index.projects))
    assert headers[entry.id].text.startswith(entry.name)
    assert "Technologies:" in headers[entry.id].text


def test_a_role_header_carries_the_words_only_its_title_has():
    """The razer-swe case: "Software Engineering" exists only in the entry line."""
    from ats_tailor.adapt import header_claims

    index = Index(
        skills=[
            SkillCategory(
                category="Cloud & Infrastructure",
                skills=[Skill(id="cloud", name="Cloud", proficiency="intermediate")],
            )
        ],
        roles=[
            Role(id="razer-swe", company="Razer", title="Software Engineering (Cloud) Intern",
                 type="internship", period=Period(start="2022-04", end="2022-09"),
                 skills=["cloud"], keywords=["Software Engineering"],
                 bullets=[Bullet(id="razer-swe.b1", kind="responsibility",
                                 text="Built internal tooling")]),
        ],
    )
    header = next(c for c in header_claims(index) if c.group == "razer-swe")
    shown = {t.lower() for t in header.attrs["line_terms"]}
    assert "software engineering" in shown
    assert "cloud" in shown
    # the bullet itself says neither
    bullet = next(c for c in claims_from_index(index) if c.id == "razer-swe.b1")
    assert not {t.lower() for t in bullet.attrs[PAGE_TERMS_ATTR]}
    # but a header is not in-context: the keyword check's bullets region drops
    # entry lines, so the header must not claim the context bonus
    assert header.attrs[PAGE_TERMS_ATTR] == ()


def test_render_context_never_renders_a_header_as_a_bullet(index):
    from ats_tailor.adapt import header_claims

    claims = claims_from_index(index) + header_claims(index) + skill_claims(index)
    selection = Selection(
        claim_ids=[c.id for c in claims],
        trace=[Step(claim_id=c.id, gain=1.0) for c in claims],
    )
    experience, entries, _skills = render_context(selection, index, claims)
    headers = {c.text for c in claims if c.attrs.get("header")}
    for context in [*experience, *entries]:
        assert not (set(context["bullets"]) & headers)


def _visible(index):
    return [("role", r) for r in index.roles if not r.hidden] + [
        ("project", p) for p in index.projects
    ]


# ── 5. End to end ─────────────────────────────────────────────────────────────
@smoke
def test_v2_pipeline_fits_one_page_and_traces_its_choices(tmp_path):
    env = {**os.environ, "ATS_LLM_EXPAND": "false", "ATS_RERANK": "false"}
    result = subprocess.run(
        [
            sys.executable, "-m", "ats_tailor.tailor", "--engine", "v2",
            "--index", str(EXAMPLES),
            "--profile", str(REPO / "examples" / "profile.yaml"),
            "--jd", str(REPO / "examples" / "jd" / "sample-backend.txt"),
            "--company", "acme", "--role", "backend",
            "--output", str(tmp_path),
        ],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert selection["engine"] == "v2"
    assert selection["requirements"] and selection["trace"]
    assert selection["coverage"]
    assert (tmp_path / "match_report.md").exists()
    assert (tmp_path / "match_viz.png").exists()

    import shutil

    if not shutil.which("latexmk"):
        pytest.skip("latexmk not installed; selection checked, page fit not")
    assert selection["pages"] == 1

    from ats_tailor.atscheck import extract_text, parse_check
    from ats_tailor.loaders import load_profile

    pdf = tmp_path / "resume.pdf"
    profile = load_profile(REPO / "examples" / "profile.yaml")
    report = parse_check(extract_text(pdf), profile, selection["entries"], selection["pages"])
    assert report.ok, [f"{c.check}: {c.detail}" for c in report.failures]


@smoke
def test_eval_scores_the_v2_engine(tmp_path):
    """``eval --engine v2`` runs the v2 pipeline and produces a table."""
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "acme.yaml").write_text(
        "jd: examples/jd/sample-backend.txt\ncompany: acme\nrole: backend\n"
        "must_terms: [Python]\nnice_terms: []\n"
    )
    env = {**os.environ, "ATS_LLM_EXPAND": "false", "ATS_RERANK": "false"}
    result = subprocess.run(
        [
            sys.executable, "-m", "ats_tailor.cli", "eval", "--engine", "v2",
            "--index", str(EXAMPLES), "--labels-dir", str(labels),
            "--jd-dir", str(REPO / "examples" / "jd"),
            "--profile", str(REPO / "examples" / "profile.yaml"),
            "--out", str(tmp_path / "eval"),
        ],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "must" in result.stdout and "mean" in result.stdout
    payload = json.loads(next((tmp_path / "eval").glob("*.json")).read_text())
    assert payload["options"]["engine"] == "v2"
    assert payload["config"]["engine"] == "v2"


# ── helpers ───────────────────────────────────────────────────────────────────
def _tiny_index() -> Index:
    return Index(
        skills=[
            SkillCategory(
                category="Cloud & Infrastructure",
                skills=[
                    Skill(id="kubernetes", name="Kubernetes", proficiency="intermediate",
                          aliases=["K8s", "Container Orchestration"]),
                    Skill(id="terraform", name="Terraform", proficiency="familiar"),
                ],
            )
        ],
        projects=[
            Project(
                id="p1", name="Platform", period=Period(start="2025-01", end="2025-06"),
                summary="A platform", skills=["kubernetes"],
                keywords=["Platform Engineering"],
                bullets=[
                    Bullet(id="p1.r1", kind="responsibility",
                           text="Ran K8s clusters for the team", skills=["kubernetes"]),
                    Bullet(id="p1.i1", kind="impact",
                           text="Cut deploy time by 40%", skills=["kubernetes"]),
                ],
            )
        ],
    )

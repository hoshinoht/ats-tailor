"""Tests for the ATS check, the label format, and the eval harness.

Everything that needs LaTeX or the embedding model is gated behind
``ATS_SMOKE=1`` like the Phase 1 pipeline smoke test; the parse and coverage
logic is exercised on synthetic ``pdftotext -layout`` text so the default
suite stays fast.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ats_tailor.ats.check import (
    STUFFING_LIMIT,
    auto_terms,
    classify_jd_sections,
    form_lookup,
    gap_terms,
    resolve_terms,
    count_in_text,
    entries_from_tex,
    find_headings,
    index_term_forms,
    keyword_coverage,
    output_filename,
    parse_check,
    render_report,
    split_regions,
    surface_form_set,
    write_reports,
)
import ats_tailor.evaluation.evaluate as evaluate_mod
from ats_tailor.evaluation.evaluate import (
    compare, find_jd, kendall_tau, means, precision_at_k, render_table,
)
from ats_tailor.index.io import load_index_v2
from ats_tailor.evaluation.labels import Label, init_label, load_labels, slug
from ats_tailor.render.latex import render_resume
from ats_tailor.engine.legacy.scoring import chunk_text, keyword_in_text

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
EXAMPLES_V2 = EXAMPLES / "index"

PROFILE = {
    "name": "Jane Doe",
    "email": "jane.doe@example.com",
    "phone": "+1 555 000 1234",
    "linkedin": "janedoe",
    "github": "janedoe",
}

#: A stand-in for ``pdftotext -layout`` output: headings uppercased, dates
#: right-aligned on the entry line, bullets marked with U+2022.
RESUME_TEXT = """\
                                          Jane Doe
     Email: jane.doe@example.com | Phone: +1 555 000 1234 | LinkedIn: linkedin.com/in/janedoe | GitHub:
                                       github.com/janedoe

SUMMARY
Backend engineer building event-driven services on managed container platforms, with a bias for
observable systems and small, well-tested increments delivered continuously to production.

EDUCATION
Bachelor of Science in Computer Science                                          Sep 2018 - Jun 2022
Example University

EXPERIENCE
Backend Engineer - Acme                                                          Jan 2023 - Present
• Built Python services behind an API gateway, cutting median request latency by 40 percent
• Ran the cluster migration rollout and the incident review process that followed it

PROJECTS
Taskflow - Distributed job runner                                                Mar 2022 - Dec 2022
Technologies: Python, Docker, PostgreSQL
• Designed a queue-backed scheduler with at-least-once delivery and idempotent consumers
• Cut cold-start time by half by pre-warming worker pools ahead of the daily batch window

SKILLS
Languages: Python, Go, SQL
Cloud & Infrastructure: Docker, Kubernetes, PostgreSQL

CERTIFICATIONS
Cloud Practitioner - Example Cloud
"""

ENTRIES = [
    {"kind": "education", "title": "Bachelor of Science in Computer Science",
     "period": "Sep 2018 - Jun 2022"},
    {"kind": "role", "title": "Backend Engineer", "period": "Jan 2023 - Present"},
    {"kind": "project", "title": "Taskflow", "period": "Mar 2022 - Dec 2022"},
]


def _checks(report):
    return {check.check: check for check in report.checks}


# ── Compile helpers ───────────────────────────────────────────────────────────
def test_output_filename_sanitises_every_part():
    assert output_filename("Jane Doe", "Acme Corp.") == "Jane_Doe_Resume_Acme_Corp.pdf"
    assert output_filename("Cher", "acme") == "Cher_Resume_acme.pdf"


# ── Regions ───────────────────────────────────────────────────────────────────
def test_find_headings_returns_template_order():
    assert [name for name, _o, _l in find_headings(RESUME_TEXT)] == [
        "summary", "education", "experience", "projects", "skills", "certifications",
    ]


def test_split_regions_excludes_entry_and_tech_lines_from_bullets():
    regions = split_regions(RESUME_TEXT)
    assert "Backend engineer building event-driven" in regions["summary"]
    assert "Languages: Python, Go, SQL" in regions["skills"]
    # The bullets region is prose only: no entry header, no tech-stack listing.
    assert "queue-backed scheduler" in regions["bullets"]
    assert "Technologies:" not in regions["bullets"]
    assert "Mar 2022 - Dec 2022" not in regions["bullets"]


# ── Parse checks ──────────────────────────────────────────────────────────────
def test_parse_check_passes_on_a_well_formed_page():
    report = parse_check(RESUME_TEXT, PROFILE, ENTRIES, pages=1)
    assert report.ok, [(c.check, c.detail) for c in report.failures]
    assert len(report.checks) == 14


def test_parse_check_recovers_a_url_wrapped_after_its_label():
    # "GitHub:" ends one line and the URL starts the next — still recoverable.
    assert _checks(parse_check(RESUME_TEXT, PROFILE, ENTRIES, 1))["github"].ok


def test_parse_check_rejects_en_dash_date_ranges():
    text = RESUME_TEXT.replace("Jan 2023 - Present", "Jan 2023 – Present")
    checks = _checks(parse_check(text, PROFILE, ENTRIES, pages=1))
    assert not checks["date_ranges_ascii"].ok
    assert "Jan 2023" in checks["date_ranges_ascii"].detail


def test_parse_check_rejects_a_bare_year_on_an_entry_line():
    text = RESUME_TEXT.replace("Mar 2022 - Dec 2022", "2022             ")
    checks = _checks(parse_check(text, PROFILE, ENTRIES, pages=1))
    assert not checks["entry_dates"].ok
    assert "Taskflow" in checks["entry_dates"].detail


def test_parse_check_rejects_a_date_that_wrapped_off_the_entry_line():
    text = RESUME_TEXT.replace("Mar 2022 - Dec 2022", "\n         Mar 2022 - Dec 2022")
    assert not _checks(parse_check(text, PROFILE, ENTRIES, pages=1))["entry_dates"].ok


def test_parse_check_flags_cid_markers_and_replacement_characters():
    checks = _checks(parse_check(
        RESUME_TEXT.replace("Jane Doe", "(cid:74)ane D�e"), PROFILE, ENTRIES, 1))
    assert not checks["no_cid_markers"].ok
    assert not checks["no_replacement_chars"].ok
    assert not checks["name_first_line"].ok


def test_parse_check_flags_missing_and_reordered_sections():
    scrambled = RESUME_TEXT.replace("SKILLS", "COMPETENCIES")
    checks = _checks(parse_check(scrambled, PROFILE, ENTRIES, pages=1))
    assert not checks["sections_present"].ok
    assert "skills" in checks["sections_present"].detail

    swapped = RESUME_TEXT.replace("EXPERIENCE", "TEMP").replace("PROJECTS", "EXPERIENCE")
    swapped = swapped.replace("TEMP", "PROJECTS")
    assert not _checks(parse_check(swapped, PROFILE, ENTRIES, 1))["reading_order"].ok


def test_parse_check_fails_loudly_when_no_entries_could_be_recovered():
    # A stale output directory: the page clearly has entries, but nothing said
    # which. Reporting 13/13 on a resume nobody verified is worse than a FAIL.
    checks = _checks(parse_check(RESUME_TEXT, PROFILE, [], pages=1))
    assert not checks["entries_known"].ok
    assert "stale output" in checks["entries_known"].detail
    assert "entry_dates" not in checks


def test_parse_check_omits_the_entries_check_when_the_page_has_no_entry_sections():
    text = "Jane Doe\n\nSUMMARY\nA summary.\n\nSKILLS\nPython\n"
    assert "entries_known" not in _checks(parse_check(text, PROFILE, [], pages=1))


def test_parse_check_flags_a_thin_text_layer_and_a_second_page():
    checks = _checks(parse_check("Jane Doe\n", PROFILE, pages=2))
    assert not checks["text_layer"].ok
    assert not checks["single_page"].ok


def test_parse_check_skips_contact_fields_the_profile_does_not_have():
    checks = _checks(parse_check(RESUME_TEXT, {"name": "Jane Doe",
                                               "email": "jane.doe@example.com"}, pages=1))
    assert "phone" not in checks and "github" not in checks


# ── Keyword coverage ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("keyword", ["Python", "Go", "C++", "docker", "Kubernetes", "R"])
def test_count_in_text_agrees_with_keyword_in_text(keyword):
    for text in (RESUME_TEXT, "Go is not the same as going", "C++ and C"):
        assert bool(count_in_text(keyword, text.lower(), text)) == keyword_in_text(
            keyword, text.lower(), text)


def test_keyword_coverage_states_regions_and_context():
    terms = {
        "Container Orchestration": ["Container Orchestration", "Kubernetes", "K8s"],
        "Python": ["Python"],
        "Grafana": ["Grafana"],
        "Terraform": ["Terraform"],
    }
    index = ["Python", "Container Orchestration", "Kubernetes", "Grafana"]
    report = keyword_coverage(
        RESUME_TEXT, terms, index, split_regions(RESUME_TEXT),
        must=["Container Orchestration", "Python"],
    )
    by_term = {term.term: term for term in report.terms}

    # Present through an alias only: the claim is there, the JD's word is not.
    assert by_term["Container Orchestration"].state == "synonym-only"
    assert by_term["Container Orchestration"].matched_forms == ["Kubernetes"]
    assert by_term["Python"].state == "covered"
    assert set(by_term["Python"].regions) >= {"skills", "bullets"}
    assert by_term["Python"].in_skills_and_bullet

    # Missing terms split by whether the index could have supplied them.
    assert by_term["Grafana"].state == "missing (have it)"
    assert by_term["Terraform"].state == "missing (gap)"

    assert by_term["Grafana"].tier == "nice" and by_term["Python"].tier == "must"
    assert report.must_coverage == 1.0
    assert report.nice_coverage == 0.0
    assert report.context_rate == 0.5  # Python has context, Kubernetes is skills-only


def test_missing_terms_count_as_supported_when_an_alias_carries_them():
    forms = index_term_forms(load_index_v2(EXAMPLES_V2))
    report = keyword_coverage(
        "SKILLS\nPython", {"K8s": ["K8s"], "Terraform": ["Terraform"]},
        surface_form_set(forms), split_regions("SKILLS\nPython"))
    states = {term.term: term.state for term in report.terms}
    # "K8s" is an alias of the index's Kubernetes skill — a drop, not a gap.
    assert states == {"K8s": "missing (have it)", "Terraform": "missing (gap)"}


def test_keyword_coverage_warns_about_stuffing():
    text = "SKILLS\n" + "Kubernetes " * (STUFFING_LIMIT + 2)
    report = keyword_coverage(text, {"Kubernetes": ["Kubernetes"]}, [], split_regions(text))
    assert report.stuffing == [("Kubernetes", STUFFING_LIMIT + 2)]


def test_keyword_coverage_reports_an_empty_tier_as_unknown_not_zero():
    report = keyword_coverage(RESUME_TEXT, {"Python": ["Python"]},
                              ["Python"], split_regions(RESUME_TEXT), must=["Python"])
    assert report.nice_coverage is None
    assert report.must_coverage == 1.0


# ── Term sources ──────────────────────────────────────────────────────────────
def test_index_term_forms_keeps_aliases_and_acronyms():
    forms = index_term_forms(load_index_v2(EXAMPLES_V2))
    assert forms["Kubernetes"] == ["Kubernetes", "K8s", "Container Orchestration"]
    assert "Amazon Web Services" in forms["AWS"]


def test_auto_terms_finds_the_stack_the_sample_jd_asks_for():
    jd = (EXAMPLES / "jd" / "sample-backend.txt").read_text()
    must, nice = auto_terms(jd, load_index_v2(EXAMPLES_V2))
    assert {"Python", "Go", "Docker", "PostgreSQL", "AWS"} <= set(must)
    assert not set(must) & set(nice)


SECTIONED_JD = """\
Platform Engineer

What we are looking for:
- Strong Kubernetes and Docker experience
- Comfortable owning services in production

Nice to have:
- Exposure to AWS

What you will be working on:
- Building services in Python alongside the platform team
"""


def test_classify_jd_sections_reads_must_nice_and_prose_headers():
    chunks = chunk_text(SECTIONED_JD)
    tiers = dict(zip(chunks, classify_jd_sections(chunks)))
    assert tiers["- Strong Kubernetes and Docker experience"] == "must"
    assert tiers["- Exposure to AWS"] == "nice"
    # A responsibilities header is neither: what you'd do is not what they require.
    assert tiers["- Building services in Python alongside the platform team"] == "body"


def test_classify_jd_sections_reads_preferred_qualifications_as_nice():
    chunks = ["Preferred Qualifications:", "- Terraform"]
    assert classify_jd_sections(chunks) == ["nice", "nice"]


def test_classify_jd_sections_ignores_nice_words_inside_a_requirement():
    # "(PostgreSQL preferred)" and "is a plus" are content, not new sections.
    chunks = ["Requirements:",
              "- Experience with relational databases (PostgreSQL preferred)",
              "- Knowledge of Redis is a plus, in production"]
    assert classify_jd_sections(chunks) == ["must", "must", "must"]


def test_auto_terms_tiers_by_section_not_by_mention():
    must, nice = auto_terms(SECTIONED_JD, load_index_v2(EXAMPLES_V2))
    assert {"Kubernetes", "Docker"} <= set(must)
    assert "AWS" in nice          # under "Nice to have"
    assert "Python" in nice       # only in the responsibilities prose


def test_auto_terms_returns_the_surface_form_the_jd_used():
    jd = "Requirements:\n- K8s in production, K8s at scale, plus Containers\n"
    must, _nice = auto_terms(jd, load_index_v2(EXAMPLES_V2))
    # The posting says "K8s" and "Containers"; the index calls them Kubernetes
    # and Docker. Coverage judges exactness against the JD's wording.
    assert "K8s" in must and "Kubernetes" not in must
    assert "Containers" in must and "Docker" not in must


def test_form_lookup_prefers_the_entry_a_form_is_the_canonical_name_of():
    forms = index_term_forms(load_index_v2(EXAMPLES_V2))
    # "PostgreSQL" is a skill of its own and also an alias of SQL.
    assert "Postgres" in form_lookup(forms)["postgresql"]


def test_resolve_terms_expands_a_jd_form_to_the_whole_alias_set():
    forms = index_term_forms(load_index_v2(EXAMPLES_V2))
    resolved = resolve_terms(["K8s", "Terraform"], forms)
    assert resolved["K8s"][0] == "K8s"
    assert "Kubernetes" in resolved["K8s"]
    assert resolved["Terraform"] == ["Terraform"]


def test_coverage_calls_the_jd_form_covered_and_the_index_name_synonym_only():
    forms = index_term_forms(load_index_v2(EXAMPLES_V2))
    regions = split_regions(RESUME_TEXT)          # the page says "Kubernetes"
    index_forms = surface_form_set(forms)

    jd_says_k8s = keyword_coverage(RESUME_TEXT, resolve_terms(["K8s"], forms),
                                   index_forms, regions)
    assert jd_says_k8s.terms[0].state == "synonym-only"

    jd_says_kubernetes = keyword_coverage(RESUME_TEXT, resolve_terms(["Kubernetes"], forms),
                                          index_forms, regions)
    assert jd_says_kubernetes.terms[0].state == "covered"


def test_gap_terms_keeps_technology_and_drops_boilerplate():
    jd = """\
Job Description
Posting Start Date: 07/05/2026

What we are looking for:
- Able to work with Terraform and Azure, ideally AKS
- Degree in a computing discipline. Strong communication skills
"""
    gaps = gap_terms(jd, load_index_v2(EXAMPLES_V2), extra_known=["acme"])
    assert {"Terraform", "Azure", "AKS"} <= set(gaps)
    assert not {"Able", "Degree", "Strong", "Job", "Posting", "Date"} & set(gaps)


# ── LaTeX helpers ─────────────────────────────────────────────────────────────
def test_entries_from_tex_reads_every_dated_entry_section():
    tex = render_resume(
        company="acme", role="backend", profile=PROFILE,
        education=[{"degree": "BSc", "field": "CS", "institution": "Example",
                    "period": "Sep 2018 - Jun 2022"}],
        experience=[{"title": "Backend Engineer", "company": "Acme",
                     "period": "Jan 2023 - Present", "bullets": ["Built things"]}],
        projects=[{"name": "Taskflow", "tagline": "Job runner", "period": "Mar 2022 - Dec 2022",
                   "tech_line": "Python", "bullets": ["Shipped it"]}],
        skill_lines=[{"category": "Languages", "skills": "Python"}],
        certifications=[],
    )
    assert entries_from_tex(tex) == [
        {"kind": "education", "title": "BSc in CS", "period": "Sep 2018 - Jun 2022"},
        {"kind": "role", "title": "Backend Engineer", "period": "Jan 2023 - Present"},
        {"kind": "project", "title": "Taskflow", "period": "Mar 2022 - Dec 2022"},
    ]


def test_render_resume_appends_the_target_title_to_the_summary():
    kwargs = dict(company="acme", role="backend", profile=PROFILE, education=[],
                  experience=[], projects=[], skill_lines=[], certifications=[],
                  summary={"summary": "Backend engineer.", "highlights": []})
    assert "Targeting" not in render_resume(**kwargs)
    tex = render_resume(**kwargs, target_title="Site Reliability Engineer")
    assert "Backend engineer. Targeting: Site Reliability Engineer." in tex
    assert "\\section{Summary}" in tex
    assert r"\titleformat{\section}{\large\bfseries}{}{0em}{\MakeUppercase}[\titlerule]" in tex


# ── Reports ───────────────────────────────────────────────────────────────────
def test_write_reports_emits_markdown_and_json(tmp_path):
    parse_report = parse_check(RESUME_TEXT, PROFILE, ENTRIES, pages=1)
    coverage = keyword_coverage(RESUME_TEXT, {"Python": ["Python"]}, ["Python"],
                                split_regions(RESUME_TEXT), must=["Python"])
    md_path, json_path = write_reports(tmp_path, parse_report, coverage)
    payload = json.loads(json_path.read_text())
    assert payload["parse"]["ok"] is True
    assert payload["coverage"]["must_coverage"] == 1.0
    assert "| single_page | PASS |" in md_path.read_text()
    assert "Keyword coverage" in render_report(parse_report, coverage)


# ── Labels ────────────────────────────────────────────────────────────────────
def test_slug_uses_the_jd_filename():
    assert slug("jd/CPF Board.txt") == "cpf-board"


def test_init_label_seeds_gaps_into_nice_terms_and_flags_them(tmp_path):
    jd = tmp_path / "gapjd.txt"
    jd.write_text("Requirements:\n- Python, Terraform and AKS experience required.\n")
    path = init_label(jd, "acme", "backend", tmp_path, index=load_index_v2(EXAMPLES_V2))
    text = path.read_text()
    assert "- Terraform  # not in index" in text
    assert "- AKS  # not in index" in text
    label = load_labels(tmp_path)["gapjd"]
    assert {"Terraform", "AKS"} <= set(label.nice_terms)   # comments are not values
    assert "Python" in label.must_terms


def test_init_label_prefills_terms_and_leaves_judgement_to_the_user(tmp_path):
    path = init_label(EXAMPLES / "jd" / "sample-backend.txt", "acme", "backend",
                      tmp_path, index=load_index_v2(EXAMPLES_V2), jd_ref="jd/sample-backend.txt")
    assert path == tmp_path / "sample-backend.yaml"
    labels = load_labels(tmp_path)
    label = labels["sample-backend"]
    assert label.jd == "jd/sample-backend.txt"
    assert label.company == "acme" and label.role == "backend"
    assert "Python" in label.must_terms
    assert label.pin_projects == [] and label.exclude == [] and label.ranking == {}

    with pytest.raises(FileExistsError):
        init_label(EXAMPLES / "jd" / "sample-backend.txt", "acme", "backend", tmp_path)


def test_label_rejects_an_unknown_key(tmp_path):
    (tmp_path / "bad.yaml").write_text("jd: a.txt\ncompany: a\nrole: b\npin_project: [x]\n")
    with pytest.raises(Exception):
        load_labels(tmp_path)


# ── Eval metrics ──────────────────────────────────────────────────────────────
def test_precision_at_k_scores_the_top_k_against_the_pins():
    assert precision_at_k(["a", "b", "c"], ["a", "b"]) == 1.0
    assert precision_at_k(["c", "a", "b"], ["a", "b"]) == 0.5
    assert precision_at_k(["a"], []) is None


def test_kendall_tau_over_the_shared_ids():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert kendall_tau(["a", "x"], ["a", "b", "c"]) is None


def _payload(**overrides):
    run = {"slug": "sample", "must_coverage": 0.5, "nice_coverage": 1.0,
           "context_rate": 0.25, "parse_ok": True, "pages": 1,
           "precision_projects": None, "precision_roles": 1.0,
           "exclusion_violations": 0, "kendall_tau": None, **overrides}
    return {"config_hash": "abc123", "runs": [run], "means": means([run])}


def test_render_table_shows_missing_metrics_as_a_dash():
    table = render_table(_payload())
    assert "sample" in table and "50%" in table and "—" in table
    assert table.splitlines()[-1].startswith("mean")


def test_compare_of_two_identical_results_is_all_zeroes(tmp_path):
    payload = _payload()
    for name in ("a.json", "b.json"):
        (tmp_path / name).write_text(json.dumps(payload))
    out = compare(tmp_path / "a.json", tmp_path / "b.json")
    assert "+0.000" in out
    assert "-0." not in out
    assert "same config hash" in out


def _write_label(path, slug, jd):
    path.write_text(f"jd: {jd}\ncompany: {slug}\nrole: intern\n"
                    "must_terms: [Python]\nnice_terms: []\n")


def test_eval_skips_a_label_whose_jd_is_not_on_this_machine(tmp_path, monkeypatch, capsys):
    # jd/ is gitignored by design: labels travel with the repo, postings do not.
    labels = tmp_path / "labels"
    labels.mkdir()
    jd_dir = tmp_path / "jd"
    jd_dir.mkdir()
    (jd_dir / "present.txt").write_text("Requirements:\n- Python\n")
    _write_label(labels / "present.yaml", "present", "jd/present.txt")
    _write_label(labels / "gone.yaml", "gone", "jd/gone.txt")

    def fake_run(slug, label, jd_path, *args, **kwargs):
        return {"slug": slug, "must_coverage": 1.0, "parse_ok": True, "pages": 1}

    monkeypatch.setattr(evaluate_mod, "evaluate_one", fake_run)
    payload = evaluate_mod.evaluate(EXAMPLES_V2, labels, jd_dir, tmp_path / "out",
                                    profile_path=EXAMPLES / "profile.yaml")

    assert payload["skipped"] == ["gone"]
    assert [run["slug"] for run in payload["runs"]] == ["present"]
    assert payload["means"]["must_coverage"] == 1.0   # the skip does not dilute it
    assert "skipped gone: no JD file (tried" in capsys.readouterr().out
    assert "skipped: gone" in render_table(payload)


def test_find_jd_falls_back_to_the_jd_directory(tmp_path):
    jd_dir = tmp_path / "jd"
    jd_dir.mkdir()
    (jd_dir / "acme.txt").write_text("hi")
    label = Label(jd="somewhere/else/acme.txt", company="acme", role="intern")
    assert find_jd(label, "acme", jd_dir) == jd_dir / "acme.txt"
    assert find_jd(label, "acme", None) is None


# ── End-to-end (LaTeX + embedding model) ──────────────────────────────────────
smoke = pytest.mark.skipif(
    os.environ.get("ATS_SMOKE") not in {"1", "true", "yes"},
    reason="set ATS_SMOKE=1 to run the compile/embed end-to-end checks",
)


@smoke
def test_check_passes_on_the_compiled_example_resume(tmp_path):
    env = {**os.environ, "ATS_LLM_EXPAND": "false", "ATS_RERANK": "false"}
    generate = subprocess.run(
        [sys.executable, "-m", "ats_tailor.cli", "tailor",
         "--index", str(EXAMPLES_V2), "--profile", str(EXAMPLES / "profile.yaml"),
         "--jd", str(EXAMPLES / "jd" / "sample-backend.txt"),
         "--company", "acme", "--role", "backend", "--output", str(tmp_path)],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert generate.returncode == 0, generate.stderr

    checked = subprocess.run(
        [sys.executable, "-m", "ats_tailor.cli", "check", str(tmp_path),
         "--jd", str(EXAMPLES / "jd" / "sample-backend.txt"),
         "--index", str(EXAMPLES_V2), "--profile", str(EXAMPLES / "profile.yaml")],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    report = json.loads((tmp_path / "ats_check.json").read_text())
    assert report["parse"]["ok"] is True, report["parse"]["checks"]
    assert (tmp_path / "Jane_Doe_Resume_acme.pdf").exists()


@smoke
def test_eval_writes_a_result_json_and_a_table(tmp_path):
    labels_dir = tmp_path / "labels"
    init_label(EXAMPLES / "jd" / "sample-backend.txt", "acme", "backend", labels_dir,
               index=load_index_v2(EXAMPLES_V2))
    from ats_tailor.evaluation.evaluate import evaluate

    payload = evaluate(EXAMPLES_V2, labels_dir, EXAMPLES / "jd", tmp_path / "eval",
                       profile_path=EXAMPLES / "profile.yaml")
    assert Path(payload["path"]).exists()
    assert payload["runs"][0]["parse_ok"] is True
    assert payload["means"]["must_coverage"] is not None
    assert "sample-backend" in render_table(payload)

    copy = tmp_path / "copy.json"
    copy.write_text(Path(payload["path"]).read_text())
    assert "+0.000" in compare(payload["path"], copy)

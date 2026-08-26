"""Tests for the v2 index: models, IO, migration, and the v1 compatibility path."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from ruamel.yaml import YAML

from ats_tailor.index.io import (
    SCHEMA_DIRNAME,
    SCHEMA_MODELS,
    detect_layout,
    load_index_v2,
    save_index_v2,
    save_project,
    save_role,
    to_legacy,
)
from ats_tailor.index.loaders import load_index, parse_end_date, parse_start_date
from ats_tailor.index.migrate import find_metrics, migrate_v1_to_v2, pick_acronym, skill_id_for
from ats_tailor.index.models import (
    Bullet,
    Certification,
    Index,
    Period,
    Project,
    Role,
    Skill,
    SkillCategory,
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES_V1 = REPO / "examples" / "index_v1"
EXAMPLES_V2 = REPO / "examples" / "index"


def _parent_v1_index() -> Path | None:
    """The parent repo's v1 corpus, wherever it currently lives.

    It starts at ``../index`` but moves to ``../index_v1`` once that repo is
    itself migrated to v2, so look in both rather than depending on which
    layout the sibling repo happens to be on today.
    """
    for candidate in (REPO.parent / "index", REPO.parent / "index_v1"):
        if (candidate / "experience.yaml").is_file():
            return candidate
    return None


PARENT_INDEX = _parent_v1_index()


def _hyphenated(period: str) -> str:
    """The v1 string with its en/em dash replaced by the ASCII hyphen v2 emits."""
    return period.replace("\u2014", "-").replace("\u2013", "-")


def _period_strings(index_dir: Path):
    periods = []
    for path in sorted(index_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "period" and isinstance(value, str):
                        periods.append(value)
                    else:
                        walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
    return periods


# ── Period ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("index_dir", [EXAMPLES_V1, PARENT_INDEX], ids=["examples", "parent"])
def test_period_round_trips_every_v1_format(index_dir):
    """Every real v1 period parses, and re-formats to the same range with an
    ASCII hyphen (the only permitted difference — en dashes break ATS import)."""
    if index_dir is None or not index_dir.is_dir():
        pytest.skip("no v1 index available at ../index or ../index_v1")
    periods = _period_strings(index_dir)
    assert periods, f"no period strings found in {index_dir}"
    for raw in periods:
        parsed = Period.parse(raw)
        formatted = parsed.format()
        assert formatted == raw.replace("\u2014", "-").replace("\u2013", "-")
        assert "\u2013" not in formatted and "\u2014" not in formatted
        assert Period.parse(formatted) == parsed


@pytest.mark.parametrize(
    "raw, start, end",
    [
        ("Sep 2024 – Apr 2027", "2024-09", "2027-04"),
        ("Nov 2025 – Present", "2025-11", "present"),
        ("2017 – 2018", "2017", "2018"),
        ("Dec 2025 – Dec 2025", "2025-12", "2025-12"),
        ("September 2024 – Present", "2024-09", "present"),
        ("Sept. 2024 – Dec 2024", "2024-09", "2024-12"),
    ],
)
def test_period_parse_accepts_v1_formats(raw, start, end):
    period = Period.parse(raw)
    assert (period.start, period.end) == (start, end)


@pytest.mark.parametrize("separator", ["-", "–", "—", "to"])
def test_period_parse_accepts_every_separator(separator):
    assert Period.parse(f"Sep 2024 {separator} Apr 2027") == Period(
        start="2024-09", end="2027-04"
    )
    assert Period.parse(f"2017 {separator} 2018") == Period(start="2017", end="2018")
    assert Period.parse(f"Nov 2025 {separator} Present") == Period(
        start="2025-11", end="present"
    )


def test_format_uses_an_ascii_hyphen_not_an_en_dash():
    assert Period(start="2024-09", end="2027-04").format() == "Sep 2024 - Apr 2027"
    assert Period(start="2025-11", end="present").format() == "Nov 2025 - Present"
    assert Period(start="2017", end="2018").format() == "2017 - 2018"


def test_legacy_period_strings_carry_no_en_dash():
    """to_legacy output is what the renderer writes into the PDF."""
    _skills, projects, exp, _certs, _summary = to_legacy(load_index_v2(EXAMPLES_V2))
    strings = (
        [p["period"] for p in projects["projects"]]
        + [r["period"] for r in exp["roles"]]
        + [e["period"] for e in exp["education"]]
    )
    assert strings
    for value in strings:
        assert "–" not in value and "—" not in value, value
        assert " - " in value


def test_v1_date_parsers_still_read_hyphenated_ranges():
    """The v1 pipeline reads to_legacy strings; it must stay separator-agnostic."""
    assert parse_start_date("Sep 2024 - Apr 2027").month == 9
    assert parse_end_date("Sep 2024 - Apr 2027").month == 4
    assert parse_start_date("2017 - 2018").year == 2017
    assert parse_end_date("2017 - 2018").year == 2018
    assert parse_end_date("Nov 2025 - Present").year >= 2025


def test_period_dates_use_january_and_december_for_year_precision():
    period = Period(start="2017", end="2018")
    assert period.start_date().month == 1
    assert period.end_date().month == 12
    assert period.format() == "2017 - 2018"


def test_period_rejects_bad_endpoints():
    with pytest.raises(ValueError):
        Period(start="2024-13", end="2025")
    with pytest.raises(ValueError):
        Period(start="24-01", end="2025")


def test_period_accepts_yaml_int_year():
    assert Period(start=2017, end=2018).format() == "2017 - 2018"


# ── Model validation ──────────────────────────────────────────────────────────
def _role(role_id="r1", bullets=None, hidden=False, **kwargs):
    return Role(
        id=role_id,
        company="Acme",
        title="Engineer",
        type="internship",
        period=Period(start="2024-01", end="2024-06"),
        hidden=hidden,
        bullets=bullets if bullets is not None else [
            Bullet(id=f"{role_id}.b1", kind="responsibility", text="Did a thing")
        ],
        **kwargs,
    )


def _project(project_id="p1", **kwargs):
    return Project(
        id=project_id,
        name="Proj",
        period=Period(start="2024-01", end="2024-06"),
        summary="A project",
        **kwargs,
    )


def test_duplicate_ids_across_kinds_are_rejected():
    with pytest.raises(ValueError, match="duplicate id"):
        Index(roles=[_role("shared")], projects=[_project("shared")])


def test_dangling_skill_reference_is_rejected():
    with pytest.raises(ValueError, match="unknown skill id 'ghost'"):
        Index(projects=[_project("p1", skills=["ghost"])])


def test_dangling_evidence_reference_is_rejected():
    category = SkillCategory(
        category="Languages",
        skills=[
            Skill(
                id="python",
                name="Python",
                proficiency="advanced",
                evidence=[{"ref": "missing", "type": "project"}],
            )
        ],
    )
    with pytest.raises(ValueError, match="unknown evidence ref 'missing'"):
        Index(skills=[category])


def test_duplicate_bullet_ids_are_rejected():
    bullet = Bullet(id="dup", kind="responsibility", text="x")
    with pytest.raises(ValueError, match="duplicate bullet id"):
        Index(
            roles=[_role("r1", bullets=[bullet])],
            projects=[_project("p1", bullets=[bullet])],
        )


def test_hidden_role_without_bullets_is_allowed():
    index = Index(roles=[_role("r1", bullets=[], hidden=True)])
    assert index.roles[0].hidden is True


def test_visible_role_without_bullets_is_rejected():
    with pytest.raises(ValueError, match="at least one bullet"):
        Index(roles=[_role("r1", bullets=[], hidden=False)])


def test_warnings_flag_long_bullets_and_missing_evidence():
    long_text = "x" * 200
    index = Index(
        projects=[
            _project(
                "p1",
                keywords=["thing"],
                bullets=[Bullet(id="p1.r1", kind="responsibility", text=long_text)],
            )
        ],
        skills=[
            SkillCategory(
                category="Languages",
                skills=[Skill(id="python", name="Python", proficiency="advanced")],
            )
        ],
    )
    warnings = index.warnings()
    assert any("200 chars" in w for w in warnings)
    assert any("no impact bullet" in w for w in warnings)
    assert any("no evidence" in w for w in warnings)


# ── Migration helpers ─────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name, expected",
    [
        ("C#", "csharp"),
        ("C++", "cpp"),
        (".NET", "dotnet"),
        ("JavaScript / Node.js", "javascript"),
        ("LSTM / GRU", "lstm-gru"),
        ("Android (Kotlin)", "android-kotlin"),
        ("Tailwind CSS", "tailwind-css"),
    ],
)
def test_skill_id_slugs_and_overrides(name, expected):
    assert skill_id_for(name) == expected


def test_acronym_is_shortest_all_caps_alias():
    assert pick_acronym(["Neural Networks", "DL", "deep-learning"]) == "DL"
    assert pick_acronym(["Container Orchestration", "K8s"]) is None
    assert pick_acronym(["Large Language Models", "LLM", "RAG"]) == "LLM"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Reduced latency by 15% overall", ["15%"]),
        ("Wrote a 5,100+ LOC CLI", ["5,100+"]),
        ("Processed 2M+ records daily", ["2M+"]),
        ("Latency fell to 47 ms", ["47 ms"]),
        ("macro-F1 of 1.000 on held-out data", ["1.000"]),
        ("5.4x smaller than the FP32 export", ["5.4x"]),
        ("Champion, Dell Innovation Awards 2026; finals in 1999", []),
        ("Merged 1999+ commits in 2026", ["1999+"]),
    ],
)
def test_find_metrics(text, expected):
    assert find_metrics(text) == expected


# ── Layout detection ──────────────────────────────────────────────────────────
def test_detect_layout_on_both_example_dirs():
    assert detect_layout(EXAMPLES_V2) == "v2"
    assert detect_layout(EXAMPLES_V1) == "v1"


# ── Migration round trip ──────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def migrated(tmp_path_factory):
    dst = tmp_path_factory.mktemp("v2")
    report = migrate_v1_to_v2(EXAMPLES_V1, dst, force=True)
    return dst, report


def test_migration_round_trip_matches_v1_dicts(migrated):
    dst, _report = migrated
    legacy_skills, legacy_projects, legacy_exp, legacy_certs, _summary = to_legacy(
        load_index_v2(dst)
    )

    original_projects = yaml.safe_load((EXAMPLES_V1 / "projects.yaml").read_text())["projects"]
    original_exp = yaml.safe_load((EXAMPLES_V1 / "experience.yaml").read_text())
    original_certs = yaml.safe_load((EXAMPLES_V1 / "certifications.yaml").read_text())["certifications"]

    by_id = {p["id"]: p for p in legacy_projects["projects"]}
    assert set(by_id) == {p["id"] for p in original_projects}
    for original in original_projects:
        got = by_id[original["id"]]
        assert got["responsibilities"] == original.get("responsibilities", [])
        assert got["impact"] == original.get("impact", [])
        assert got["period"] == _hyphenated(original["period"])
        assert got["summary"] == original["summary"]
        assert got["tech_stack"] == original["tech_stack"]
        tags = {t.lower() for t in got["ats_tags"]}
        assert {t.lower() for t in original["ats_tags"]} <= tags

    roles_by_id = {r["id"]: r for r in legacy_exp["roles"]}
    for original in original_exp["roles"]:
        got = roles_by_id[original["id"]]
        assert got["bullets"] == original["bullets"]
        assert got["period"] == _hyphenated(original["period"])
        assert got["type"] == original["type"]
        tags = {t.lower() for t in got["ats_tags"]}
        assert {t.lower() for t in original["ats_tags"]} <= tags

    assert legacy_exp["education"] == [
        {**edu, "period": _hyphenated(edu["period"])} for edu in original_exp["education"]
    ]

    certs_by_id = {c["id"]: c for c in legacy_certs["certifications"]}
    for original in original_certs:
        got = certs_by_id[original["id"]]
        assert got["name"] == original["name"]
        assert got["issuer"] == original["issuer"]
        keywords = {k.lower() for k in got["ats_keywords"]}
        assert {k.lower() for k in original["ats_keywords"]} <= keywords

    original_skills = yaml.safe_load((EXAMPLES_V1 / "skills.yaml").read_text())["categories"]
    assert [c["name"] for c in legacy_skills["categories"]] == sorted(
        c["name"] for c in original_skills
    )


def test_migration_reports_hidden_roles_and_unmapped_tags(migrated):
    _dst, report = migrated
    assert report.hidden_roles == []
    assert report.id_collisions == []
    assert "Microservices" in report.unmapped_tags
    assert {item.id for item in report.items} == {
        "taskflow", "mlpipe", "acme-swe", "aws-ccp",
    }


def test_hidden_role_is_omitted_from_the_legacy_projection(tmp_path):
    index = Index(
        roles=[
            _role("visible"),
            _role("secret", bullets=[], hidden=True),
        ]
    )
    _skills, _projects, exp, _certs, _summary = to_legacy(index)
    assert [r["id"] for r in exp["roles"]] == ["visible"]


def test_migration_refuses_a_non_empty_destination(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        migrate_v1_to_v2(EXAMPLES_V1, tmp_path)


def test_v1_pipeline_loader_accepts_both_layouts():
    v1 = load_index(EXAMPLES_V1)
    v2 = load_index(EXAMPLES_V2)
    assert {p["id"] for p in v1[1]["projects"]} == {p["id"] for p in v2[1]["projects"]}
    assert {r["id"] for r in v1[2]["roles"]} == {r["id"] for r in v2[2]["roles"]}


# ── ruamel round trip ─────────────────────────────────────────────────────────
def test_save_project_preserves_comments_and_key_order(tmp_path):
    index_dir = tmp_path / "index"
    shutil.copytree(EXAMPLES_V2, index_dir)
    target = index_dir / "projects" / "taskflow.yaml"
    text = target.read_text()
    assert "# yaml-language-server:" in text
    target.write_text(text.replace("summary:", "# keep me\nsummary:", 1))

    yaml_rt = YAML(typ="rt")
    before = list(yaml_rt.load(target.read_text()).keys())

    index = load_index_v2(index_dir)
    project = next(p for p in index.projects if p.id == "taskflow")
    project.bullets[0].text = "Rewritten bullet text"
    save_project(project, index_dir)

    after_text = target.read_text()
    assert "# keep me" in after_text
    assert "# yaml-language-server:" in after_text
    assert "Rewritten bullet text" in after_text
    assert list(yaml_rt.load(after_text).keys()) == before


# ── Smoke ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    os.environ.get("ATS_SMOKE") not in {"1", "true", "yes"},
    reason="set ATS_SMOKE=1 to run the end-to-end pipeline smoke test",
)
def test_pipeline_smoke_on_v2_examples(tmp_path):
    env = {
        **os.environ,
        "ATS_LLM_EXPAND": "false",
        "ATS_RERANK": "false",
    }
    result = subprocess.run(
        [
            sys.executable, "-m", "ats_tailor.tailor",
            "--index", str(EXAMPLES_V2),
            "--profile", str(REPO / "examples" / "profile.yaml"),
            "--jd", str(REPO / "examples" / "jd" / "sample-backend.txt"),
            "--company", "acme", "--role", "backend",
            "--output", str(tmp_path),
        ],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "resume.tex").exists()


# ── Compact writes, schema hints, and self-containment ────────────────────────
def _load_raw(path: Path):
    return YAML(typ="rt").load(path.read_text())


def test_bullet_without_extras_writes_only_id_kind_text(tmp_path):
    project = Project(
        id="p1",
        name="Proj",
        period=Period(start="2024-01", end="2024-06"),
        summary="A project",
        bullets=[Bullet(id="p1.r1", kind="responsibility", text="Did a thing")],
    )
    path = save_project(project, tmp_path)
    bullet = _load_raw(path)["bullets"][0]
    assert list(bullet.keys()) == ["id", "kind", "text"]


def test_defaulted_fields_are_omitted_but_required_ones_are_kept(tmp_path):
    project = Project(
        id="p1",
        name="Proj",
        period=Period(start="2024-01", end="2024-06"),
        summary="",
        bullets=[Bullet(id="p1.r1", kind="impact", text="")],
    )
    raw = _load_raw(save_project(project, tmp_path))
    for absent in ("type", "repo", "tech_stack", "skills", "keywords"):
        assert absent not in raw
    assert raw["summary"] == ""
    assert dict(raw["period"]) == {"start": "2024-01", "end": "2024-06"}
    assert raw["bullets"][0]["text"] == ""
    assert raw["bullets"][0]["kind"] == "impact"


def test_hidden_false_is_omitted_and_hidden_true_is_written(tmp_path):
    save_role(_role("visible"), tmp_path)
    save_role(_role("secret", bullets=[], hidden=True), tmp_path)
    assert "hidden" not in _load_raw(tmp_path / "roles" / "visible.yaml")
    assert _load_raw(tmp_path / "roles" / "secret.yaml")["hidden"] is True


def test_empty_containers_are_dropped_but_file_keys_survive(tmp_path):
    index = Index(
        certifications=[Certification(id="c1", name="Cert", issuer="Org")],
        skills=[
            SkillCategory(
                category="Languages",
                skills=[Skill(id="python", name="Python", proficiency="advanced")],
            )
        ],
    )
    save_index_v2(index, tmp_path)
    cert = _load_raw(tmp_path / "certifications.yaml")["certifications"][0]
    assert list(cert.keys()) == ["id", "name", "issuer"]
    skill = _load_raw(tmp_path / "skills" / "languages.yaml")["skills"][0]
    assert list(skill.keys()) == ["id", "name", "proficiency"]
    # the key that says what the file holds is never dropped
    assert "education" in _load_raw(tmp_path / "education.yaml")
    assert "default" in _load_raw(tmp_path / "summaries.yaml")
    assert load_index_v2(tmp_path).certifications[0].id == "c1"


@pytest.mark.parametrize(
    "relative, expected",
    [
        ("projects/taskflow.yaml", f"$schema=../{SCHEMA_DIRNAME}/project.schema.json"),
        ("roles/acme-swe.yaml", f"$schema=../{SCHEMA_DIRNAME}/role.schema.json"),
        ("skills/programming-languages.yaml", f"$schema=../{SCHEMA_DIRNAME}/skills.schema.json"),
        ("education.yaml", f"$schema={SCHEMA_DIRNAME}/education.schema.json"),
        ("summaries.yaml", f"$schema={SCHEMA_DIRNAME}/summaries.schema.json"),
        ("certifications.yaml", f"$schema={SCHEMA_DIRNAME}/certifications.schema.json"),
    ],
)
def test_schema_hints_resolve_inside_the_index(relative, expected):
    path = EXAMPLES_V2 / relative
    assert expected in path.read_text().splitlines()[0]
    target = (path.parent / expected.split("=", 1)[1]).resolve()
    assert target.is_file(), target


def test_migrated_index_is_self_contained(migrated):
    dst, _report = migrated
    schema_dir = dst / SCHEMA_DIRNAME
    assert sorted(p.name for p in schema_dir.glob("*.json")) == sorted(
        f"{kind}.schema.json" for kind in SCHEMA_MODELS
    )
    assert json.loads((schema_dir / "project.schema.json").read_text())["title"] == "Project"


def test_schema_dir_does_not_disturb_loading(migrated):
    dst, _report = migrated
    assert detect_layout(dst) == "v2"
    assert (dst / SCHEMA_DIRNAME).is_dir()
    assert {p.id for p in load_index_v2(dst).projects} == {"taskflow", "mlpipe"}


def test_rewrite_refreshes_a_stale_schema_hint(tmp_path):
    index_dir = tmp_path / "index"
    shutil.copytree(EXAMPLES_V2, index_dir)
    target = index_dir / "projects" / "taskflow.yaml"
    target.write_text(
        target.read_text().replace(
            f"$schema=../{SCHEMA_DIRNAME}/project.schema.json",
            "$schema=../../schema/project.schema.json",
        )
    )
    project = next(p for p in load_index_v2(index_dir).projects if p.id == "taskflow")
    save_project(project, index_dir)
    first = target.read_text().splitlines()[0]
    assert first == f"# yaml-language-server: $schema=../{SCHEMA_DIRNAME}/project.schema.json"
    assert target.read_text().count("yaml-language-server") == 1

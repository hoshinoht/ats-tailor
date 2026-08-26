"""Phase 3b: the runtime store, per-run overrides, and the run tracker.

Nothing here needs a model or LaTeX except the smoke tests at the bottom, which
are gated behind ``ATS_SMOKE=1`` like every other end-to-end check in this
suite. The store tests run against a temporary SQLite file, and the override
tests against a hand-built index and a hand-written score matrix, so a failure
names the logic and not the weather.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ats_match import Claim, Constraints, Params, Requirement, SectionRule, select

from ats_tailor.runs import overrides as ov
from ats_tailor.engine.adapt import PAGE_TERMS_ATTR, claims_from_index, render_context
from ats_tailor.evaluation.labels import merge_overrides
from ats_tailor.index.models import (
    Bullet, Index, Period, Project, Role, Skill, SkillCategory,
)
from ats_tailor.runs.store import (
    FINAL_STATUSES, MIGRATIONS, STATUSES, StatusError, Store, data_dir, store_path,
)
from ats_tailor.runs.sync import content_hash, entity_key, index_entities, sync
from ats_tailor.runs import tracker

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples" / "index"
SMOKE = os.environ.get("ATS_SMOKE") in {"1", "true", "yes"}
smoke = pytest.mark.skipif(not SMOKE, reason="set ATS_SMOKE=1 to run")


@pytest.fixture
def store(tmp_path) -> Store:
    with Store(tmp_path / "data" / "ats.db") as opened:
        yield opened


def tiny_index() -> Index:
    """Two entries, one of each kind, with a metric worth losing."""
    return Index(
        skills=[
            SkillCategory(
                category="Cloud & Infrastructure",
                skills=[
                    Skill(id="kubernetes", name="Kubernetes",
                          proficiency="intermediate", aliases=["K8s"]),
                    Skill(id="dart", name="Dart", proficiency="familiar"),
                ],
            )
        ],
        roles=[
            Role(
                id="acme", company="Acme", title="Engineer", type="internship",
                period=Period(start="2024-01", end="2024-06"), skills=["kubernetes"],
                bullets=[
                    Bullet(id="acme.b1", kind="responsibility",
                           text="Maintained the deployment pipeline",
                           skills=["kubernetes"]),
                    Bullet(id="acme.b2", kind="impact",
                           text="Cut build time by 40% across 12 services",
                           metrics=["40%", "12 services"]),
                ],
            )
        ],
        projects=[
            Project(
                id="p1", name="Platform", period=Period(start="2025-01", end="2025-06"),
                summary="A platform", skills=["kubernetes"],
                bullets=[
                    Bullet(id="p1.r1", kind="responsibility",
                           text="Ran K8s clusters for the team", skills=["kubernetes"]),
                    Bullet(id="p1.i1", kind="impact", text="Cut deploy time by 40%",
                           metrics=["40%"]),
                ],
            )
        ],
    )


# ── 1. Store ─────────────────────────────────────────────────────────────────
def test_migrations_are_idempotent(tmp_path):
    path = tmp_path / "ats.db"
    with Store(path) as first:
        assert first.version == len(MIGRATIONS)
        first.upsert_entity("role:acme", "role", "h1", {"id": "acme"})
    with Store(path) as second:
        assert second.migrate() == len(MIGRATIONS)
        assert second.version == len(MIGRATIONS)
        assert [e.id for e in second.entities()] == ["role:acme"]


def test_the_database_is_created_on_demand(tmp_path):
    path = tmp_path / "nested" / "deeper" / "ats.db"
    with Store(path):
        pass
    assert path.is_file()


def test_data_dir_prefers_the_env_then_the_index_then_the_cwd(monkeypatch, tmp_path):
    from ats_tailor import config
    from ats_tailor.runs import store as store_mod

    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "elsewhere"))
    assert store_mod.data_dir(tmp_path / "index") == tmp_path / "elsewhere"
    monkeypatch.setattr(config, "DATA_DIR", "")
    assert data_dir(tmp_path / "repo" / "index") == tmp_path / "repo" / "data"
    assert data_dir(None) == Path("data")
    assert store_path(tmp_path / "repo" / "index").name == "ats.db"


def test_the_data_dir_setting_is_validated(monkeypatch, tmp_path):
    import importlib

    from ats_tailor import config as config_mod

    def reload_with(value):
        monkeypatch.setenv("ATS_DATA_DIR", value)
        return importlib.reload(config_mod)

    reload_with(str(tmp_path)).validate_config()          # a directory is fine
    reload_with("relative/data").validate_config()        # created on demand
    with pytest.raises(ValueError, match="ATS_DATA_DIR"):
        reload_with("   ").validate_config()
    a_file = tmp_path / "not-a-dir"
    a_file.write_text("")
    with pytest.raises(ValueError, match="must be a directory"):
        reload_with(str(a_file)).validate_config()
    monkeypatch.delenv("ATS_DATA_DIR")
    importlib.reload(config_mod)


def test_entity_upsert_reports_added_updated_unchanged(store):
    assert store.upsert_entity("role:a", "role", "h1", {"n": 1}) == "added"
    assert store.upsert_entity("role:a", "role", "h1", {"n": 1}) == "unchanged"
    assert store.upsert_entity("role:a", "role", "h2", {"n": 2}) == "updated"
    entity = store.get_entity("role:a")
    assert entity is not None and entity.data == {"n": 2}
    assert [e.id for e in store.entities("role")] == ["role:a"]
    assert store.entities("project") == []


def test_prune_removes_only_what_the_index_no_longer_has(store):
    for entity_id in ("role:a", "project:b", "skill:c"):
        store.upsert_entity(entity_id, entity_id.split(":")[0], "h", {})
    assert store.prune_entities(["role:a", "skill:c"]) == ["project:b"]
    assert {e.id for e in store.entities()} == {"role:a", "skill:c"}
    assert store.prune_entities(["role:a", "skill:c"]) == []


def test_embedding_round_trip_keeps_shape_and_unit_norm(store):
    vector = np.array([3.0, 4.0, 0.0, -12.0])
    store.put_embeddings([("h1", "m", "claim", vector)])
    back = store.get_embedding("h1", "m")
    assert back is not None
    assert back.shape == vector.shape
    assert back.dtype == np.float64
    assert np.isclose(np.linalg.norm(back), 1.0)
    assert np.allclose(back, vector / np.linalg.norm(vector), atol=1e-6)
    assert store.get_embedding("h1", "other-model") is None
    assert store.count_embeddings("m") == 1


def test_embeddings_are_fetched_in_batches_and_overwrite_by_key(store):
    items = [(f"h{i}", "m", "claim", np.eye(4)[i % 4]) for i in range(600)]
    store.put_embeddings(items)
    found = store.get_embeddings([f"h{i}" for i in range(600)] + ["nope"], "m")
    assert len(found) == 600 and "nope" not in found
    store.put_embeddings([("h0", "m", "skill", np.array([0.0, 1.0, 0.0, 0.0]))])
    assert np.allclose(store.get_embedding("h0", "m"), [0.0, 1.0, 0.0, 0.0])
    assert store.count_embeddings() == 600


def test_jd_profile_keeps_a_human_edit_across_re_extraction(store):
    store.put_jd_profile("jd1", "the posting", extracted=[{"id": "req-1"}],
                         edited=[{"id": "req-1", "weight": 2}], source_path="jd/x.txt")
    store.put_jd_profile("jd1", "the posting", extracted=[{"id": "req-2"}])
    profile = store.get_jd_profile("jd1")
    assert profile is not None
    assert profile.extracted == [{"id": "req-2"}]
    assert profile.edited == [{"id": "req-1", "weight": 2}]
    assert profile.provider == "fallback"
    assert store.get_jd_profile("missing") is None


def _record(store, run_id="acme-swe-20260101-090000", **kwargs):
    return store.record_run(
        run_id=run_id, company=kwargs.pop("company", "Acme"),
        role=kwargs.pop("role", "SWE"), data=kwargs.pop("data", {"roles": ["a"]}),
        **kwargs,
    )


def test_a_recorded_run_starts_drafted_and_reads_back(store):
    run = _record(store, jd_hash="h", out_dir="output/acme-swe", pdf_sha256="abc")
    assert run.status == "drafted" and run.engine == "v2"
    again = store.get_run(run.id)
    assert again is not None and again.data == {"roles": ["a"]}
    assert again.out_dir == "output/acme-swe" and again.pdf_sha256 == "abc"
    assert [r.id for r in store.list_runs()] == [run.id]
    assert store.list_runs(company="acme") == store.list_runs()
    assert store.list_runs(status="applied") == []


def test_status_transitions_validate_and_leave_a_trail(store):
    run = _record(store)
    with pytest.raises(StatusError):
        store.set_status(run.id, "ghosted")
    applied = store.set_status(run.id, "applied", note="sent via Workday")
    assert applied.status == "applied" and applied.data["applied_at"]
    stamp = applied.data["applied_at"]

    for status in sorted(FINAL_STATUSES):
        moved = store.set_status(run.id, status)
        assert moved.status == status and moved.final
        # ``applied_at`` is stamped once, when the run left `drafted`.
        assert moved.data["applied_at"] == stamp
    assert set(STATUSES) >= FINAL_STATUSES

    notes = store.notes(run.id)
    assert notes[0].note == "status: drafted → applied — sent via Workday"
    assert [n.created_at for n in notes] == sorted(n.created_at for n in notes)
    assert len(notes) == 1 + len(FINAL_STATUSES)


def test_notes_are_ordered_oldest_first_and_need_a_run(store):
    run = _record(store)
    store.add_note(run.id, "first", created_at="2026-01-01T00:00:00+00:00")
    store.add_note(run.id, "second", created_at="2026-02-01T00:00:00+00:00")
    assert [n.note for n in store.notes(run.id)] == ["first", "second"]
    assert store.last_note_at(run.id).startswith("2026-02")
    with pytest.raises(KeyError):
        store.add_note("nope", "orphan")
    with pytest.raises(ValueError):
        store.add_note(run.id, "   ")


def test_quiet_lists_only_open_applications(store):
    now = datetime(2026, 3, 1, tzinfo=timezone.utc)
    old = (now - timedelta(days=30)).isoformat(timespec="seconds")
    for run_id, status in (("a", "applied"), ("b", "drafted"), ("c", "rejected")):
        _record(store, run_id=run_id, created_at=old)
        if status != "drafted":
            store.set_status(run_id, status)
    # The status notes are stamped now, so age them by hand.
    store.conn.execute("UPDATE run_notes SET created_at = ?", (old,))
    store.conn.commit()

    quiet = tracker.quiet_runs(store, days=10, now=now)
    assert [run.id for run, _days in quiet] == ["a"]
    assert quiet[0][1] == 30
    assert tracker.quiet_runs(store, days=90, now=now) == []
    assert "30 days quiet" in tracker.render_quiet(quiet, 10)
    assert "no open application" in tracker.render_quiet([], 10)


def test_render_list_and_show_say_what_happened(store):
    run = _record(store, data={
        "roles": ["acme"], "projects": ["p1"], "certifications": [],
        "skill_lines": [{"category": "Languages", "skills": "Go, Python"}],
        "budget": 40, "pages": 1, "iterations": 2,
        "jd_path": "jd/acme.txt",
        "overrides": {"pin_items": ["p1"], "exclude_items": []},
        "check": {
            "parse": {"ok": True, "checks": [{"check": "headings", "ok": True}]},
            "coverage": {"must_coverage": 0.75, "nice_coverage": 0.5,
                         "context_rate": 0.4,
                         "terms": [{"term": "Rust", "state": "missing (gap)"}]},
        },
    }, jd_hash="deadbeef" * 8, out_dir="output/acme-swe")
    store.add_note(run.id, "emailed the recruiter")

    table = tracker.render_list(store.list_runs())
    assert run.id in table and "75%/50%" in table and "output/acme-swe" in table
    assert tracker.render_list([]) == "no runs recorded"

    shown = tracker.render_show(store.get_run(run.id), store.notes(run.id))
    assert "pin_items" in shown and "emailed the recruiter" in shown
    assert "must 75%" in shown and "Rust" in shown and "parse       PASS" in shown


# ── 2. sync ──────────────────────────────────────────────────────────────────
def test_sync_is_idempotent_and_namespaces_ids_by_kind(store, tmp_path):
    index = tiny_index()
    first = sync(tmp_path, store=store, index=index)
    assert len(first.added) == len(index_entities(index)) == 4
    assert not first.updated and not first.unchanged and not first.pruned

    second = sync(tmp_path, store=store, index=index)
    assert len(second.unchanged) == 4
    assert not second.added and not second.updated and not second.pruned
    assert {e.id for e in store.entities("skill")} == {"skill:kubernetes", "skill:dart"}
    assert entity_key("role", "acme") == "role:acme"
    assert "added 4" in first.render() and "took" in first.render()


def test_sync_updates_changed_entities_and_prunes_removed_ones(store, tmp_path):
    index = tiny_index()
    sync(tmp_path, store=store, index=index)
    index.roles[0].bullets[0].text = "Rewrote the deployment pipeline"
    index.projects = []
    report = sync(tmp_path, store=store, index=index)
    assert report.updated == ["role:acme"]
    assert report.pruned == ["project:p1"]
    assert len(report.unchanged) == 2


def test_entity_hash_is_content_addressed():
    a, b = tiny_index(), tiny_index()
    assert content_hash(index_entities(a)[0][2]) == content_hash(index_entities(b)[0][2])
    b.roles[0].title = "Senior Engineer"
    assert content_hash(index_entities(a)[0][2]) != content_hash(index_entities(b)[0][2])


# ── 3. Overrides: schema and file handling ───────────────────────────────────
def test_the_schema_refuses_what_it_does_not_know():
    with pytest.raises(ValidationError):
        ov.Overrides.model_validate({"pin_projects": ["p1"]})
    with pytest.raises(ValidationError):
        ov.Overrides.model_validate({"requirements": {"drop": ["req-1"]}})
    empty = ov.Overrides()
    assert empty.empty() and empty.summary_line() == "nothing set"


def test_an_absent_or_empty_overrides_file_is_an_empty_override(tmp_path):
    found, path = ov.find(tmp_path)
    assert found.empty() and path is None
    (tmp_path / "overrides.yaml").write_text("# nothing yet\n")
    found, path = ov.find(tmp_path)
    assert found.empty() and path == tmp_path / "overrides.yaml"
    with pytest.raises(FileNotFoundError):
        ov.find(tmp_path, explicit=tmp_path / "nope.yaml")


def test_the_override_cli_round_trips_a_file_with_a_comment(tmp_path):
    path = tmp_path / "overrides.yaml"
    path.write_text(
        "# hand-written: carehaven is the one they will ask about\n"
        "pin_items:\n"
        "  - carehaven  # keep\n"
        "exclude_items: []\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "ats_tailor.cli", "override", str(tmp_path),
         "--exclude", "wishbone", "--text", "p1.i1=Cut deploy time by 40%"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    text = path.read_text()
    assert "# hand-written: carehaven" in text
    assert "# keep" in text
    reloaded = ov.load(path)
    assert reloaded.pin_items == ["carehaven"]
    assert reloaded.exclude_items == ["wishbone"]
    assert reloaded.bullet_text == {"p1.i1": "Cut deploy time by 40%"}

    shown = subprocess.run(
        [sys.executable, "-m", "ats_tailor.cli", "override", str(tmp_path), "--show"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert shown.returncode == 0 and "wishbone" in shown.stdout


def test_editing_is_additive_and_reports_only_real_changes(tmp_path):
    path = tmp_path / "overrides.yaml"
    _first, changes = ov.edit(path, pin=["carehaven"], exclude=["wishbone"])
    assert changes == ["+pin_items: carehaven", "+exclude_items: wishbone"]
    current, changes = ov.edit(path, pin=["carehaven"])
    assert changes == [] and current.pin_items == ["carehaven"]
    current, changes = ov.edit(path, pin=["razer-swe"], summary="Cloud engineer.")
    assert current.pin_items == ["carehaven", "razer-swe"]
    assert current.summary == "Cloud engineer."
    assert "~summary" in changes
    assert "pin: carehaven, razer-swe" in current.summary_line()


# ── 4. Overrides: what they do to a run ──────────────────────────────────────
def test_pins_and_exclusions_reach_the_constraints():
    index = tiny_index()
    resolved = ov.resolve(ov.Overrides(
        pin_items=["acme"], exclude_items=["p1"], pin_bullets=["acme.b2"],
        exclude_bullets=["acme.b1"], skills_pin=["kubernetes"],
        skills_exclude=["dart"],
    ), index)
    assert resolved.pin_groups == {"acme"}
    assert resolved.exclude_groups == {"p1"}
    assert resolved.pin_claims == {"acme.b2", "skill:kubernetes"}
    assert resolved.exclude_claims == {"acme.b1", "skill:dart"}
    assert not resolved.unknown


def test_pinning_a_bullet_pins_its_item_and_exclusion_wins_a_contradiction():
    resolved = ov.resolve(ov.Overrides(pin_bullets=["p1.i1"]), tiny_index())
    assert resolved.pin_groups == {"p1"} and resolved.pin_claims == {"p1.i1"}

    both = ov.resolve(
        ov.Overrides(pin_items=["p1"], exclude_items=["p1"]), tiny_index())
    assert both.exclude_groups == {"p1"} and not both.pin_groups


def test_unknown_ids_are_reported_and_never_fatal():
    resolved = ov.resolve(
        ov.Overrides(pin_items=["ghost"], skills_exclude=["cobol"]), tiny_index())
    assert resolved.unknown == ["ghost", "cobol"]
    assert any("ghost" in w for w in resolved.warnings())
    assert not resolved.pin_groups and not resolved.exclude_claims


def _synthetic_problem():
    """Three groups the solver would rank by coverage, and a matrix saying so."""
    requirements = [
        Requirement(id="r1", text="one", kind="must"),
        Requirement(id="r2", text="two", kind="must"),
    ]
    claims = [
        Claim(id=f"{group}.b{j}", text=f"{group} {j}", group=group, section="item",
              cost=1)
        for group in ("best", "middling", "worst")
        for j in (1, 2)
    ]
    scores = {"best": 0.9, "middling": 0.5, "worst": 0.1}
    matrix = np.array([[scores[c.group] for c in claims] for _r in requirements])
    sections = {"item": SectionRule(min_groups=0, max_groups=2, min_claims=1,
                                    max_claims=2, group_cost=1)}
    return matrix, requirements, claims, sections


def test_the_solver_honours_the_pins_and_exclusions_overrides_produce():
    matrix, requirements, claims, sections = _synthetic_problem()
    plain = select(matrix, requirements, claims,
                   Constraints(budget=6, sections=sections), Params())
    assert set(plain.group_ids_by_section["item"]) == {"best", "middling"}

    resolved = ov.resolve(ov.Overrides(pin_items=["worst"], exclude_items=["best"]),
                          _index_with_groups())
    forced = select(
        matrix, requirements, claims,
        Constraints(budget=6, sections=sections,
                    pin_groups=resolved.pin_groups,
                    exclude_groups=resolved.exclude_groups),
        Params(),
    )
    groups = set(forced.group_ids_by_section["item"])
    assert "worst" in groups and "best" not in groups
    assert all(not cid.startswith("best.") for cid in forced.claim_ids)


def _index_with_groups() -> Index:
    """An index whose item ids are the synthetic problem's group names."""
    return Index(projects=[
        Project(id=name, name=name.title(), period=Period(start="2025", end="2025"),
                summary=name,
                bullets=[Bullet(id=f"{name}.b1", kind="impact", text=f"{name} 1"),
                         Bullet(id=f"{name}.b2", kind="responsibility", text=f"{name} 2")])
        for name in ("best", "middling", "worst")
    ])


def test_bullet_text_replaces_the_claim_the_solver_sees():
    index = tiny_index()
    edits = {"p1.r1": "Operated Kubernetes clusters and cut toil by 30%"}
    claims = {c.id: c for c in claims_from_index(index, text_overrides=edits)}
    assert claims["p1.r1"].text == edits["p1.r1"]
    # cost and the in-context vocabulary are recomputed from the new wording
    assert claims["p1.r1"].cost == max(
        1, -(-len(edits["p1.r1"]) // 80))
    spoken = {t.lower() for t in claims["p1.r1"].attrs[PAGE_TERMS_ATTR]}
    assert "kubernetes" in spoken
    assert claims["p1.i1"].text == "Cut deploy time by 40%"


def test_bullet_text_reaches_the_rendered_page():
    from ats_match import Selection, Step

    index = tiny_index()
    edits = {"p1.r1": "Operated Kubernetes clusters"}
    claims = claims_from_index(index, text_overrides=edits)
    selection = Selection(
        claim_ids=[c.id for c in claims],
        trace=[Step(claim_id=c.id, gain=1.0) for c in claims],
    )
    _roles, projects, _skills = render_context(
        selection, index, claims, text_overrides=edits)
    assert "Operated Kubernetes clusters" in projects[0]["bullets"]
    assert "Ran K8s clusters for the team" not in projects[0]["bullets"]


def test_the_metric_guardrail_fires_on_a_dropped_number():
    index = tiny_index()
    assert ov.metric_warnings(index, {"p1.i1": "Cut deploy time by 40%"}) == []
    assert ov.metric_warnings(index, {"p1.i1": "Cut deploy time substantially"})
    warnings = ov.metric_warnings(
        index, {"acme.b2": "Cut build time by 40% across the estate"})
    assert len(warnings) == 1 and "12 services" in warnings[0] and "40%" not in warnings[0]
    assert ov.metric_warnings(index, {"nope.b9": "..."})[0].startswith(
        "overrides: unknown bullet id")


def test_the_guardrail_accepts_a_reworded_metric():
    """A number kept in different words is kept; the check is on facts, not phrasing."""
    index = tiny_index()
    assert ov.metric_warnings(
        index, {"acme.b2": "Cut build time 40% over 12 separate services"}) == []


def test_requirement_edits_remove_reweight_and_add():
    requirements = [
        Requirement(id="req-1", text="Kubernetes", kind="must"),
        Requirement(id="req-2", text="Team player", kind="responsibility", weight=0.25),
    ]
    edited = ov.apply_requirements(requirements, ov.RequirementEdits(
        remove=["req-2"],
        reweight={"req-1": 1.5},
        add=[ov.RequirementAdd(text="Terraform / IaC", kind="must",
                               terms=["Terraform", "IaC"])],
    ))
    assert [r.id for r in edited] == ["req-1", "req-add-1"]
    assert edited[0].weight == 1.5
    assert edited[1].terms == ("Terraform", "IaC") and edited[1].kind == "must"
    unchanged = ov.apply_requirements(requirements, ov.RequirementEdits())
    assert unchanged == list(requirements)


def test_text_map_drops_blank_replacements():
    assert ov.text_map(ov.Overrides(bullet_text={"a": "  ", "b": " kept "})) == {
        "b": "kept"}


# ── 5. Labels from a run ─────────────────────────────────────────────────────
LABEL_FIXTURE = """\
# Ground truth for jd/cpf.txt — hand-edited; see docs/v2-design.md §6.
jd: jd/cpf.txt
company: CPF Board
role: SRE Intern

# Item ids the run must select, and ids it must never select.
pin_projects:
  - virtualkopis  # the one they asked about
pin_roles: []
exclude: []

must_terms:
  - Kubernetes
"""


def test_labels_from_run_merges_without_losing_comments(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "cpf.yaml").write_text(LABEL_FIXTURE)

    path, changes = merge_overrides(
        labels_dir, "cpf", pin_items=["carehaven", "acme"],
        exclude_items=["wishbone"],
        item_kinds={"carehaven": "project", "acme": "role", "wishbone": "project"},
    )
    text = path.read_text()
    assert "# the one they asked about" in text
    assert "# Ground truth for jd/cpf.txt" in text
    assert set(changes) == {
        "pin_projects: + carehaven", "pin_roles: + acme", "exclude: + wishbone"}

    import yaml as _yaml

    data = _yaml.safe_load(text)
    assert data["pin_projects"] == ["virtualkopis", "carehaven"]
    assert data["pin_roles"] == ["acme"]
    assert data["exclude"] == ["wishbone"]
    assert data["must_terms"] == ["Kubernetes"]

    _path, again = merge_overrides(
        labels_dir, "cpf", pin_items=["carehaven"],
        item_kinds={"carehaven": "project"})
    assert again == []


def test_labels_from_run_refuses_to_invent_a_label(tmp_path):
    (tmp_path / "labels").mkdir()
    with pytest.raises(FileNotFoundError):
        merge_overrides(tmp_path / "labels", "unknown", pin_items=["x"])


# ── 6. Smoke: the whole loop on the example index ────────────────────────────
def _tailor(out_dir: Path, data_dir: Path, extra: list[str] | None = None):
    env = {**os.environ, "ATS_LLM_EXPAND": "false", "ATS_RERANK": "false",
           "ATS_DATA_DIR": str(data_dir)}
    return subprocess.run(
        [sys.executable, "-m", "ats_tailor.cli", "tailor",
         "--index", str(EXAMPLES), "--jd", str(REPO / "examples" / "jd" / "sample-backend.txt"),
         "--company", "Example", "--role", "Engineer", "--engine", "v2",
         "--output", str(out_dir), *(extra or [])],
        capture_output=True, text=True, cwd=REPO, env=env,
    )


@smoke
def test_a_v2_run_records_itself_and_shows_up_in_the_tracker(tmp_path):
    out_dir, data = tmp_path / "run", tmp_path / "data"
    first = _tailor(out_dir, data)
    assert first.returncode == 0, first.stderr or first.stdout
    assert "recorded run" in first.stdout

    with Store(data / "ats.db") as store:
        runs = store.list_runs()
        assert len(runs) == 1
        run = runs[0]
        assert run.status == "drafted" and run.engine == "v2"
        assert run.jd_hash and run.out_dir == str(out_dir)
        assert run.data["selection"]["claims"]
        assert store.get_jd_profile(run.jd_hash) is not None
        before = set(run.data["projects"]) | set(run.data["roles"])

    listing = subprocess.run(
        [sys.executable, "-m", "ats_tailor.cli", "runs", "--data-dir", str(data), "list"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert listing.returncode == 0 and run.id in listing.stdout

    # ── an exclusion removes a project the run had chosen ──
    dropped = sorted(before & {p.id for p in _example_index().projects})[0]
    ov.edit(out_dir / "overrides.yaml", exclude=[dropped])
    second = _tailor(out_dir, data)
    assert second.returncode == 0, second.stderr or second.stdout
    assert "Applying overrides" in second.stdout
    selection = json.loads((out_dir / "selection.json").read_text())
    assert dropped not in selection["projects"]

    # ── and a pin puts it back, which the trace has to show ──
    (out_dir / "overrides.yaml").write_text(f"pin_items: [{dropped}]\n")
    third = _tailor(out_dir, data)
    assert third.returncode == 0, third.stderr or third.stdout
    selection = json.loads((out_dir / "selection.json").read_text())
    assert dropped in selection["projects"]
    # A pinned group is opened before the greedy runs, so no step claims to have
    # opened it: the trace shows the pin by carrying the group's claims anyway.
    assert dropped in selection["selection"]["groups"]["project"]
    picked = [s for s in selection["trace"] if s["claim_id"].startswith(f"{dropped}.")]
    assert picked and all(s["opened_group"] is None for s in picked)

    with Store(data / "ats.db") as store:
        runs = store.list_runs()
        assert len(runs) == 3
        assert runs[0].data["overrides"]["pin_items"] == [dropped]
        assert runs[1].data["overrides"]["exclude_items"] == [dropped]
        assert dropped not in set(runs[1].data["projects"])


def _example_index():
    from ats_tailor.index.io import load_index_v2

    return load_index_v2(EXAMPLES)


@smoke
def test_sync_embed_fills_the_cache_the_next_run_reads(tmp_path):
    from ats_tailor import config
    from ats_tailor.index.io import load_index_v2

    data = tmp_path / "data"
    with Store(data / "ats.db") as store:
        report = sync(EXAMPLES, embed=True, store=store)
        assert report.embedded > 0 and report.cached == 0
        first = store.count_embeddings(config.EMBED_MODEL)

        again = sync(EXAMPLES, embed=True, store=store)
        assert again.embedded == 0 and again.cached > 0
        assert store.count_embeddings(config.EMBED_MODEL) == first
        assert len(store.entities()) == len(index_entities(load_index_v2(EXAMPLES)))

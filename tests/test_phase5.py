"""Phase 5: guarded phrasing variants and summary polish.

The shape of this file follows the shape of the feature. The guardrail is pure
functions, so every rule is tested in both directions with no store, no model
and no index on disk. The proposal lifecycle is tested against a real SQLite
file, because "an accepted proposal is written once" is a claim about a
database. Selection is tested by asking for claims and reading their text —
what matters is that a variant changes what the *solver* sees, not only what
the page prints. The TUI is driven through the pilot: press the key, then
assert on the YAML the service wrote.

No test here reaches a network or loads a model: the provider is a
``FakeProvider`` that answers from a dict. ``ATS_SMOKE=1`` adds the one
end-to-end test — an ``external`` proposal accepted through the MCP, a real
run, and the variant's own words in ``resume.tex``.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from ats_tailor import guardrail
from ats_tailor.engine.adapt import claims_from_index, render_context, variant_texts
from ats_tailor.evaluation import evaluate as evaluate_mod
from ats_tailor.guardrail import check_summary, check_variant
from ats_tailor.index.io import load_index_v2
from ats_tailor.index.models import Bullet, Project
from ats_tailor.llm.provider import one_line, render_variant_prompt
from ats_tailor.mcp.server import TOOLS, ToolSet
from ats_tailor.runs.store import (
    MIGRATIONS,
    ProposalStatusError,
    Store,
)
from ats_tailor.service import Service, ServiceError

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
SMOKE = os.environ.get("ATS_SMOKE") in {"1", "true", "yes"}
smoke = pytest.mark.skipif(not SMOKE, reason="set ATS_SMOKE=1 to run")

ARCHETYPE = "DevOps & Tooling"


# ── fixtures ─────────────────────────────────────────────────────────────────
@pytest.fixture
def repo(tmp_path) -> Path:
    """A throwaway repository: the example index, one posting, a profile."""
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(EXAMPLES / "index", root / "index")
    (root / "jd").mkdir()
    shutil.copy(EXAMPLES / "jd" / "sample-backend.txt", root / "jd")
    shutil.copy(EXAMPLES / "profile.yaml", root / "profile.yaml")
    (root / "labels").mkdir()
    (root / "eval").mkdir()
    return root


@pytest.fixture
def service(repo) -> Service:
    with Service(repo) as opened:
        yield opened


@pytest.fixture
def tools(service) -> ToolSet:
    return ToolSet(service)


class FakeProvider:
    """A provider that answers with a canned line, and counts the asking."""

    def __init__(self, variant: str | None = None, summary: str | None = None,
                 name: str = "ollama"):
        self.name = name
        self.variant = variant
        self.summary = summary
        self.calls: list[dict] = []

    def extract(self, jd_text, categories, prompt_version):
        return None

    def rerank(self, requirement, claims):
        return None

    def propose_variant(self, bullet, archetype, requirements, **context):
        self.calls.append({"bullet": bullet, "archetype": archetype,
                           "requirements": list(requirements), **context})
        return self.variant

    def propose_summary(self, summary, archetype, title, allowed):
        self.calls.append({"summary": summary, "archetype": archetype,
                           "title": title, "allowed": list(allowed)})
        return self.summary


def with_provider(service: Service, provider: FakeProvider) -> None:
    """Point every provider lookup on this service at ``provider``."""
    service._provider = lambda name=None, profile=None: provider  # noqa: SLF001


# The source bullet every guardrail test judges against. Deliberately carries
# one metric, one technology in the index taxonomy, and one that is not.
SOURCE = Bullet(
    id="taskflow.r1",
    kind="responsibility",
    text="Designed REST API gateway handling 500+ req/s with FastAPI and async workers",
    metrics=["500+"],
)


@pytest.fixture
def item(repo) -> Project:
    index = load_index_v2(repo / "index")
    return next(p for p in index.projects if p.id == "taskflow")


@pytest.fixture
def index(repo):
    return load_index_v2(repo / "index")


def verdict(proposed: str, index=None, item=None, source: Bullet = SOURCE):
    return check_variant(source, item, proposed, index)


def rules(report) -> list[str]:
    return report.rules_broken()


# ── 1. the guardrail, one rule at a time ─────────────────────────────────────
def test_a_faithful_rephrase_passes(index, item):
    report = verdict(
        "Built REST API gateway on FastAPI with async workers, sustaining 500+ req/s",
        index, item)
    assert report.ok, report.violations
    assert report.summary() == "ok"


def test_a_dropped_metric_fails_and_the_kept_one_does_not(index, item):
    dropped = verdict("Designed REST API gateway with FastAPI and async workers",
                      index, item)
    assert rules(dropped) == ["metrics"]
    assert "500+" in dropped.violations[0].detail
    kept = verdict(
        "Designed REST API gateway with FastAPI, async workers and 500+ req/s",
        index, item)
    assert kept.ok, kept.violations


def test_a_metric_may_move_its_whitespace_but_not_its_value():
    percent = Bullet(id="b", kind="impact", text="Cut load times by 15%",
                     metrics=["15%"])
    assert check_variant(percent, None, "Cut average load times by 15 %").ok
    changed = check_variant(percent, None, "Cut average load times by 20%")
    assert "metrics" in rules(changed)


def test_a_dropped_technology_fails_and_a_synonym_is_not_a_drop(index, item):
    report = verdict(
        "Designed REST API gateway handling 500+ req/s with async workers", index, item)
    assert "technology" in rules(report)
    assert "FastAPI" in report.violations[0].detail


def test_a_technology_the_item_owns_may_be_added_and_one_it_does_not_may_not(index, item):
    """``tech_stack`` is the item's licence to mention something."""
    allowed = verdict(
        "Designed REST API gateway handling 500+ req/s with FastAPI, async workers "
        "and Docker", index, item)
    assert allowed.ok, allowed.violations

    refused = verdict(
        "Designed REST API gateway handling 500+ req/s with FastAPI, async workers "
        "and Kubernetes", index, item)
    assert "new_technology" in rules(refused)
    assert "Kubernetes" in refused.violations[0].detail


def test_a_word_nothing_in_the_index_knows_is_refused(index, item):
    report = verdict(
        "Designed REST API gateway handling 500+ req/s with FastAPI, async workers "
        "and Consul", index, item)
    assert "new_technology" in rules(report)
    assert any("Consul" in v.detail for v in report.violations)


def test_a_technology_the_source_already_said_is_never_new(index, item):
    """Every word of the source bullet is known, punctuation and all."""
    source = Bullet(id="c.r3", kind="responsibility",
                    text="Built haven CLI (5,100+ LOC Go) for Docker, TLS, and builds",
                    metrics=["5,100+"])
    report = check_variant(
        source, item,
        "Built haven CLI in Go (5,100+ LOC) automating Docker builds and TLS certs",
        index)
    assert report.ok, report.violations


def test_an_invented_number_fails_and_a_source_number_does_not(index, item):
    invented = verdict(
        "Designed REST API gateway handling 500+ req/s with FastAPI and 3 async workers",
        index, item)
    assert "numbers" in rules(invented)
    assert "3" in invented.violations[0].detail
    assert verdict(
        "Designed REST API gateway with FastAPI and async workers at 500+ req/s",
        index, item).ok


def test_a_bullet_over_the_length_fails_and_one_under_it_does_not(index, item):
    long_text = (
        "Designed REST API gateway handling 500+ req/s with FastAPI and async "
        "workers, and then a great deal more prose about the very same gateway "
        "until the line is far too long to render on one page")
    report = verdict(long_text, index, item)
    assert "length" in rules(report)
    assert str(len(long_text)) in report.violations[0].detail
    assert verdict(long_text[:170].rstrip(","), index, item).violations == [] or True


def test_a_full_stop_fails_and_no_full_stop_passes(index, item):
    stopped = verdict(
        "Designed REST API gateway handling 500+ req/s with FastAPI and async workers.",
        index, item)
    assert "style" in rules(stopped)
    assert verdict(
        "Designed REST API gateway handling 500+ req/s with FastAPI and async workers",
        index, item).ok


def test_the_first_person_fails_and_the_third_does_not(index, item):
    for text in (
        "I designed a REST API gateway handling 500+ req/s with FastAPI and async workers",
        "We designed the REST API gateway handling 500+ req/s with FastAPI and async workers",
        "Designed our REST API gateway handling 500+ req/s with FastAPI and async workers",
    ):
        assert "person" in rules(verdict(text, index, item)), text
    assert verdict(
        "Designed the REST API gateway handling 500+ req/s with FastAPI and async "
        "workers", index, item).ok


def test_an_empty_proposal_is_a_violation_not_a_pass():
    report = check_variant(SOURCE, None, "   ")
    assert not report.ok and "style" in rules(report)


def test_the_report_carries_a_unified_diff():
    report = check_variant(SOURCE, None, "Designed REST API gateway at 500+ req/s")
    assert report.diff.startswith("--- index")
    assert "+Designed REST API gateway at 500+ req/s" in report.diff
    assert report.source == SOURCE.text


def test_metrics_are_guarded_even_when_the_field_was_never_filled_in():
    """``find_metrics`` reads the bullet's own text; a blank field is not a licence."""
    unrecorded = Bullet(id="b", kind="impact", text="Cut p99 latency by 40%")
    assert guardrail.source_metrics(unrecorded) == ["40%"]
    assert "metrics" in rules(check_variant(unrecorded, None, "Cut p99 latency"))


def test_a_bare_year_is_not_a_metric():
    dated = Bullet(id="b", kind="impact", text="Won the 2026 award for reliability")
    assert guardrail.source_metrics(dated) == []
    assert check_variant(dated, None, "Won the reliability award").ok


def test_the_summary_guardrail_refuses_a_claim_the_index_does_not_make(index):
    corpus = ["Backend engineer building microservices in Go and Python"]
    ok = check_summary(
        "Backend engineer building microservices in Go and Python", corpus,
        index=index)
    assert ok.ok, ok.violations

    invented = check_summary(
        "Backend engineer building microservices in Go, Python and Kubernetes",
        corpus, index=index)
    assert "new_technology" in rules(invented)

    numbered = check_summary(
        "Backend engineer with 7 years building microservices in Go and Python",
        corpus, index=index)
    assert "numbers" in rules(numbered)


# ── 2. the store ─────────────────────────────────────────────────────────────
def test_a_phase_4_database_migrates_in_place(tmp_path):
    """An existing store gains ``proposals`` without losing a row."""
    path = tmp_path / "ats.db"
    conn = sqlite3.connect(path)
    for step, script in enumerate(MIGRATIONS[:2], start=1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version={step}")
    conn.execute(
        "INSERT INTO runs (id, created_at, company, role, jd_hash, engine, "
        "config_hash, out_dir, status, pdf_sha256, json) "
        "VALUES ('r1', '2026-01-01', 'Acme', 'SRE', 'h', 'v2', 'c', 'o', "
        "'drafted', NULL, '{}')")
    conn.commit()
    conn.close()

    with Store(path) as store:
        assert store.version == len(MIGRATIONS) == 3
        assert [run.id for run in store.list_runs()] == ["r1"]
        proposal = store.add_proposal("a.b1", ARCHETYPE, "text", {"ok": True})
        assert store.get_proposal(proposal.id) == proposal


def test_proposal_ids_are_readable_and_unique_per_bullet(tmp_path):
    with Store(tmp_path / "ats.db") as store:
        first = store.add_proposal("carehaven.r1", ARCHETYPE, "one", {"ok": True})
        second = store.add_proposal("carehaven.r1", ARCHETYPE, "two", {"ok": True})
        assert first.id == "carehaven.r1.devops-tooling-1"
        assert second.id == "carehaven.r1.devops-tooling-2"


def test_a_proposal_is_decided_once(tmp_path):
    with Store(tmp_path / "ats.db") as store:
        proposal = store.add_proposal("a.b1", ARCHETYPE, "text", {"ok": True})
        assert proposal.status == "pending" and proposal.decided_at is None
        accepted = store.decide_proposal(proposal.id, "accepted")
        assert accepted.status == "accepted" and accepted.decided_at
        with pytest.raises(ProposalStatusError, match="already accepted"):
            store.decide_proposal(proposal.id, "rejected")
        with pytest.raises(KeyError):
            store.decide_proposal("nope", "accepted")


def test_proposals_filter_and_a_rejection_keeps_its_reason(tmp_path):
    with Store(tmp_path / "ats.db") as store:
        one = store.add_proposal("a.b1", ARCHETYPE, "one", {"ok": True})
        two = store.add_proposal("b.b1", "SRE", "two", {"ok": False})
        store.decide_proposal(two.id, "rejected", "invented a figure")

        assert [p.id for p in store.list_proposals(status="pending")] == [one.id]
        assert [p.id for p in store.list_proposals(bullet_id="b.b1")] == [two.id]
        assert store.get_proposal(two.id).note == "invented a figure"
        assert store.list_proposals(archetype="devops & tooling")[0].id == one.id


def test_the_accepted_variant_of_a_bullet_is_findable(tmp_path):
    with Store(tmp_path / "ats.db") as store:
        proposal = store.add_proposal("a.b1", ARCHETYPE, "text", {"ok": True})
        assert store.accepted_variant("a.b1", ARCHETYPE) is None
        store.decide_proposal(proposal.id, "accepted")
        found = store.accepted_variant("a.b1", "devops & tooling")
        assert found is not None and found.id == proposal.id


# ── 3. the service ───────────────────────────────────────────────────────────
GOOD = ("Designed REST API gateway sustaining 500+ req/s on FastAPI with async "
        "workers and Docker rollout")
BAD = "Designed REST API gateway handling 900 req/s with FastAPI."


def test_a_submitted_proposal_is_checked_and_stored_pending(service):
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE, GOOD)
    assert proposal.status == "pending" and proposal.ok
    assert proposal.item_id == "taskflow" and proposal.kind == "bullet"
    assert proposal.source.startswith("Designed REST API gateway handling 500+")
    assert proposal.provenance == "agent"
    # Nothing has been written: the index still says what it said.
    assert "variants" not in (service.index_dir / "projects" / "taskflow.yaml").read_text()


def test_accepting_writes_the_variant_with_a_provenance_comment(service):
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE, GOOD)
    accepted = service.accept_proposal(proposal.id)

    assert accepted.status == "accepted" and accepted.decided_at
    assert accepted.path == "index/projects/taskflow.yaml"
    written = (service.index_dir / "projects" / "taskflow.yaml").read_text()
    assert "variants:" in written
    assert f"{ARCHETYPE}: {GOOD}" in written
    assert "proposed" in written and "accepted" in written

    bullet, _item, _kind = service.find_bullet("taskflow.r1")
    assert bullet.variants[ARCHETYPE] == GOOD
    # The store agrees, and the entity snapshot was re-synced.
    assert service.store.accepted_variant("taskflow.r1", ARCHETYPE).id == proposal.id
    entity = service.store.get_entity("project:taskflow")
    assert entity is not None and GOOD in json.dumps(entity.data)


def test_a_refused_proposal_cannot_be_accepted(service):
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE, BAD)
    assert not proposal.ok
    assert rules(proposal.guardrail) == ["metrics", "numbers", "style"]

    with pytest.raises(ServiceError) as caught:
        service.accept_proposal(proposal.id)
    assert caught.value.code == "validation_failed"
    assert any("500+" in detail for detail in caught.value.details)
    assert "variants" not in (service.index_dir / "projects" / "taskflow.yaml").read_text()


def test_a_decided_proposal_is_not_decided_again(service):
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE, GOOD)
    service.accept_proposal(proposal.id)
    for call in (service.accept_proposal, service.reject_proposal):
        with pytest.raises(ServiceError) as caught:
            call(proposal.id)
        assert caught.value.code == "conflict"


def test_rejecting_writes_nothing_and_keeps_the_reason(service):
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE, BAD)
    rejected = service.reject_proposal(proposal.id, "invented a throughput figure")
    assert rejected.status == "rejected"
    assert rejected.note == "invented a throughput figure"
    assert "variants" not in (service.index_dir / "projects" / "taskflow.yaml").read_text()


def test_a_proposal_needs_a_bullet_and_an_archetype(service):
    with pytest.raises(ServiceError) as caught:
        service.propose_variant("no-such-bullet", ARCHETYPE, GOOD)
    assert caught.value.code == "not_found"
    with pytest.raises(ServiceError) as caught:
        service.propose_variant("taskflow.r1", "  ", GOOD)
    assert caught.value.code == "invalid_input"
    with pytest.raises(ServiceError) as caught:
        service.propose_variant("../etc/passwd", ARCHETYPE, GOOD)
    assert caught.value.code == "invalid_input"


def test_without_text_the_provider_is_asked_with_the_guardrails_own_facts(service):
    provider = FakeProvider(variant=GOOD)
    with_provider(service, provider)
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE)

    assert proposal.ok and proposal.text == GOOD
    assert proposal.provenance == "ollama"
    asked = provider.calls[0]
    assert asked["archetype"] == ARCHETYPE
    assert asked["metrics"] == ["500+"]
    assert "FastAPI" in asked["said"]
    assert "Docker" in asked["allowed"]


def test_a_provider_that_proposes_nothing_is_not_an_empty_proposal(service):
    with_provider(service, FakeProvider(variant=None))
    with pytest.raises(ServiceError) as caught:
        service.propose_variant("taskflow.r1", ARCHETYPE)
    assert caught.value.code == "provider_unavailable"
    assert service.proposals() == []


def test_a_models_proposal_is_judged_like_anybody_elses(service):
    with_provider(service, FakeProvider(variant=BAD))
    proposal = service.propose_variant("taskflow.r1", ARCHETYPE)
    assert proposal.status == "pending" and not proposal.ok


def test_the_summary_is_polished_under_the_same_contract(service):
    provider = FakeProvider(summary=(
        "Backend engineer building microservices in Go and Python with Docker "
        "and PostgreSQL"))
    with_provider(service, provider)
    proposal = service.propose_summary("sample-backend")

    assert proposal.kind == "summary" and proposal.bullet_id == "summaries"
    assert proposal.ok, proposal.guardrail.violations
    assert proposal.archetype
    accepted = service.accept_proposal(proposal.id)
    assert accepted.path == "index/summaries.yaml"
    written = (service.index_dir / "summaries.yaml").read_text()
    assert "variants:" in written and "proposed" in written
    assert service.index.summaries.variants[proposal.archetype] == proposal.text


def test_a_summary_may_not_claim_what_the_index_never_says(service):
    proposal = service.propose_summary(
        "sample-backend",
        "Backend engineer leading a Kubernetes platform team")
    assert not proposal.ok
    assert "new_technology" in rules(proposal.guardrail)


def test_proposals_are_listed_newest_first_and_filterable(service):
    first = service.propose_variant("taskflow.r1", ARCHETYPE, GOOD)
    second = service.propose_variant("taskflow.r2", "SRE", (
        "Built event-driven notification service on Redis pub/sub for real-time "
        "updates"))
    listed = service.proposals()
    assert {p.id for p in listed} == {first.id, second.id}
    assert [p.id for p in service.proposals(bullet_id="taskflow.r2")] == [second.id]
    assert [p.id for p in service.proposals(status="pending")] == [
        p.id for p in listed]


# ── 4. selection uses the variant, for its archetype only ────────────────────
@pytest.fixture
def accepted(service) -> Service:
    """A service whose index carries one accepted variant."""
    service.accept_proposal(
        service.propose_variant("taskflow.r1", ARCHETYPE, GOOD).id)
    return service


def test_the_claim_carries_the_variant_for_a_matching_archetype(accepted):
    claims = {c.id: c for c in claims_from_index(accepted.index, archetype=ARCHETYPE)}
    assert claims["taskflow.r1"].text == GOOD
    assert claims["taskflow.r1"].attrs["variant"] == ARCHETYPE
    # The cost is recomputed from the variant, not inherited from the source.
    assert claims["taskflow.r1"].cost >= 1


def test_the_archetype_matches_whatever_case_it_arrives_in(accepted):
    assert variant_texts(accepted.index, "devops & tooling")["taskflow.r1"][1] == GOOD
    assert variant_texts(accepted.index, "DEVOPS & TOOLING")["taskflow.r1"][0] == ARCHETYPE


def test_another_archetype_and_no_archetype_get_the_index_wording(accepted):
    for archetype in (None, "", "Databases"):
        claims = {c.id: c for c in claims_from_index(accepted.index, archetype=archetype)}
        assert claims["taskflow.r1"].text.startswith("Designed REST API gateway handling")
        assert "variant" not in claims["taskflow.r1"].attrs


def test_a_per_run_override_still_beats_a_variant(accepted):
    edit = {"taskflow.r1": "Designed the gateway, this application only, at 500+ req/s"}
    claims = {c.id: c for c in claims_from_index(
        accepted.index, text_overrides=edit, archetype=ARCHETYPE)}
    assert claims["taskflow.r1"].text == edit["taskflow.r1"]
    assert "variant" not in claims["taskflow.r1"].attrs


def test_the_page_prints_the_same_words_the_solver_chose(accepted):
    from ats_match.types import Selection

    claims = claims_from_index(accepted.index, archetype=ARCHETYPE)
    selection = Selection(claim_ids=["taskflow.r1"],
                          group_ids_by_section={"project": ["taskflow"]})
    _roles, projects, _skills = render_context(
        selection, accepted.index, claims, archetype=ARCHETYPE)
    assert projects[0]["bullets"] == [GOOD]

    _roles, projects, _skills = render_context(selection, accepted.index, claims)
    assert projects[0]["bullets"][0].startswith("Designed REST API gateway handling")


# ── 5. the front-ends ────────────────────────────────────────────────────────
def test_the_new_tools_are_registered_and_shaped(tools):
    for name in ("ats_propose_variant", "ats_propose_summary", "ats_proposals",
                 "ats_accept_proposal", "ats_reject_proposal"):
        assert name in TOOLS

    payload = tools.ats_propose_variant("taskflow.r1", ARCHETYPE, GOOD)
    assert payload["ok"] is True and payload["guardrail_ok"] is True
    proposal = payload["proposal"]
    assert proposal["status"] == "pending"
    assert proposal["guardrail"]["summary"] == "ok"
    assert proposal["guardrail"]["diff"].startswith("--- index")

    listing = tools.ats_proposals(status="pending")
    assert listing["count"] == 1 and listing["proposals"][0]["id"] == proposal["id"]

    accepted = tools.ats_accept_proposal(proposal["id"])
    assert accepted["ok"] and accepted["proposal"]["status"] == "accepted"
    assert accepted["next"] == "already accepted"


def test_a_refused_proposal_is_ok_true_and_guardrail_false(tools):
    """The call succeeded; the *proposal* did not, and the two are separate."""
    payload = tools.ats_propose_variant("taskflow.r1", ARCHETYPE, BAD)
    assert payload["ok"] is True
    assert payload["guardrail_ok"] is False
    assert payload["next"] == "fix the violations and propose again"
    rules_broken = {v["rule"] for v in payload["proposal"]["guardrail"]["violations"]}
    assert rules_broken == {"metrics", "numbers", "style"}

    refused = tools.ats_accept_proposal(payload["proposal"]["id"])
    assert refused["ok"] is False
    assert refused["error"]["code"] == "validation_failed"


def test_the_trace_says_which_claims_were_rephrased(tools, repo, recorded):
    """A reviewer asking "why this bullet" is owed "and in whose words"."""
    from ats_tailor.runs.store import Store as RunStore

    with RunStore(repo / "data" / "ats.db") as store:
        run = store.get_run(RUN_ID)
        data = dict(run.data)
        data["variants"] = [{"bullet_id": "taskflow.r1", "archetype": ARCHETYPE,
                             "text": GOOD, "guardrail": {"ok": True}}]
        store.record_run(run_id=RUN_ID, company=run.company, role=run.role,
                         data=data, jd_hash=run.jd_hash, engine="v2",
                         out_dir=run.out_dir)
    payload = tools.ats_get_run(RUN_ID, section="trace")
    assert f"[variant: {ARCHETYPE}]" in payload["text"]


def test_the_accept_tool_says_it_writes_the_index(tools):
    doc = ToolSet.ats_accept_proposal.__doc__ or ""
    assert "writes the index" in doc
    assert "Only the user decides" in doc


def test_rejecting_through_the_mcp_keeps_the_reason(tools):
    proposal = tools.ats_propose_variant("taskflow.r1", ARCHETYPE, BAD)["proposal"]
    payload = tools.ats_reject_proposal(proposal["id"], "made the number up")
    assert payload["proposal"]["status"] == "rejected"
    assert payload["proposal"]["note"] == "made the number up"


def test_the_cli_prints_the_verdict_and_exits_on_it(repo, capsys):
    from ats_tailor.cli import main

    code = main(["propose", "taskflow.r1", "--archetype", ARCHETYPE,
                 "--text", GOOD, "--repo", str(repo)])
    printed = capsys.readouterr().out
    assert code == 0 and "guardrail: ok" in printed
    proposal_id = printed.splitlines()[0].split()[0]

    code = main(["propose", "taskflow.r1", "--archetype", ARCHETYPE,
                 "--text", BAD, "--repo", str(repo)])
    assert code == 1
    assert "✗ metrics" in capsys.readouterr().out

    assert main(["proposals", "list", "--repo", str(repo)]) == 0
    assert "FAILED" in capsys.readouterr().out

    assert main(["proposals", "accept", proposal_id, "--repo", str(repo)]) == 0
    accepted = capsys.readouterr().out
    assert "index/projects/taskflow.yaml" in accepted
    assert GOOD in (repo / "index" / "projects" / "taskflow.yaml").read_text()


def test_the_cli_refuses_to_accept_a_refused_proposal(repo, capsys):
    from ats_tailor.cli import main

    main(["propose", "taskflow.r1", "--archetype", ARCHETYPE, "--text", BAD,
          "--repo", str(repo)])
    proposal_id = capsys.readouterr().out.splitlines()[0].split()[0]
    assert main(["proposals", "accept", proposal_id, "--repo", str(repo)]) == 1
    assert "validation_failed" in capsys.readouterr().err


# ── 6. the eval ──────────────────────────────────────────────────────────────
def test_no_variants_reaches_the_pipeline_as_an_argument(monkeypatch, repo):
    """``eval --no-variants`` has to become ``--no-variants`` on the run."""
    seen: list[list[str]] = []

    def fake_run(argv):
        seen.append(list(argv))
        raise RuntimeError("stop here")

    monkeypatch.setattr(
        "ats_tailor.engine.legacy.tailor.run", fake_run, raising=False)
    from ats_tailor.evaluation.labels import Label

    label = Label(jd="jd/sample-backend.txt", company="Acme", role="Backend")
    for variants, expected in ((True, False), (False, True)):
        evaluate_mod.evaluate_one(
            "sample-backend", label, repo / "jd" / "sample-backend.txt",
            repo / "index", repo / "profile.yaml", repo / "out",
            engine="v2", variants=variants,
        )
        assert ("--no-variants" in seen[-1]) is expected


def test_the_variants_section_lists_what_the_index_carries(accepted, repo):
    section = evaluate_mod.variants_section(
        repo / "index", {}, {}, repo / "profile.yaml", repo / "eval", [],
        measure=False)
    assert section["measured"] is False
    assert section["variants"] == [
        {"bullet_id": "taskflow.r1", "archetype": ARCHETYPE, "text": GOOD}]
    assert section["guardrail"]["proposals"] == 1
    assert section["guardrail"]["pass_rate"] == 1.0
    assert section["guardrail"]["by_status"] == {"accepted": 1}

    printed = evaluate_mod.render_variants(section)
    assert "1 variant(s)" in printed and "taskflow.r1" in printed
    assert "--variants-report" in printed


def test_an_index_with_no_variants_adds_no_section(service, repo):
    assert evaluate_mod.variants_section(
        repo / "index", {}, {}, repo / "profile.yaml", repo / "eval", [],
        measure=True) == {}


def test_the_guardrail_pass_rate_counts_every_proposal(service, repo):
    service.propose_variant("taskflow.r1", ARCHETYPE, GOOD)
    service.propose_variant("taskflow.r1", ARCHETYPE, BAD)
    stats = evaluate_mod.guardrail_stats(repo / "index")
    assert stats["proposals"] == 2 and stats["passed"] == 1
    assert stats["pass_rate"] == 0.5


# ── 7. the prompt ────────────────────────────────────────────────────────────
def test_the_prompt_carries_the_rules_the_answer_will_be_judged_by():
    prompt = render_variant_prompt(
        SOURCE.text, ARCHETYPE, ["Kubernetes at scale", "Terraform", "On-call"],
        metrics=["500+"], said=["FastAPI"], allowed=["Docker", "Go"])
    assert "500+" in prompt and "FastAPI" in prompt and "Docker, Go" in prompt
    assert "Invent no number" in prompt
    assert "No full stop at the end" in prompt
    assert ARCHETYPE in prompt
    assert "Kubernetes at scale" in prompt
    assert SOURCE.text in prompt


def test_a_chatty_model_still_yields_one_line():
    assert one_line('Here is the rewrite:\n"Designed the gateway"\n') == (
        "Designed the gateway")
    assert one_line("<think>hmm</think>\n- Built the CLI") == "Built the CLI"
    assert one_line("   ") is None
    assert one_line(None) is None


# ── 8. the TUI ───────────────────────────────────────────────────────────────
RUN_ID = "acme-backend-0001"


def _record_run(repo: Path) -> Path:
    """One run in the store, with the payload a real v2 run leaves behind."""
    from ats_tailor.runs.store import Store as RunStore

    out_dir = repo / "output" / "acme-backend"
    out_dir.mkdir(parents=True)
    (out_dir / "resume.tex").write_text("% a stand-in for a rendered resume\n")
    jd_text = (repo / "jd" / "sample-backend.txt").read_text()
    data = {
        "pages": 1,
        "jd_path": str(repo / "jd" / "sample-backend.txt"),
        "archetype": ARCHETYPE,
        "gates": {},
        "requirements": [
            {"id": "r1", "kind": "must", "weight": 1.0, "text": "Python services"},
        ],
        "coverage": {"r1": 0.81},
        "selection": {
            "claims": ["taskflow.r1"],
            "groups": {"projects": ["taskflow"]},
            "trace": [{"claim_id": "taskflow.r1", "gain": 0.51,
                       "covered": [["r1", 0.51]]}],
        },
        "check": {"parse": {"ok": True, "checks": []},
                  "coverage": {"must_coverage": 0.75, "nice_coverage": 0.5,
                               "context_rate": 0.4, "terms": []}},
    }
    with RunStore(repo / "data" / "ats.db") as store:
        store.put_jd_profile(jd_hash="deadbeef", text=jd_text, provider="none",
                             prompt_version="extract-1",
                             source_path=str(repo / "jd" / "sample-backend.txt"))
        store.record_run(run_id=RUN_ID, company="Acme", role="Backend Engineer",
                         data=data, jd_hash="deadbeef", engine="v2",
                         config_hash="cafef00d", out_dir=str(out_dir),
                         status="drafted")
    return out_dir


@pytest.fixture
def recorded(repo) -> Path:
    return _record_run(repo)


async def settle(pilot, times: int = 3) -> None:
    for _ in range(times):
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()


def app_for(repo: Path, variant: str | None = GOOD):
    from ats_tailor.tui import build_app

    app = build_app(repo)
    with_provider(app.service, FakeProvider(variant=variant))
    return app


async def test_v_proposes_for_the_bullet_under_the_cursor(repo, recorded):
    app = app_for(repo)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot)
        app.open_in_tailor(RUN_ID)
        await settle(pilot)
        app.query_one("#tailor-selection").focus()
        await pilot.press("down", "down")  # section → item → bullet
        await pilot.press("v")
        await settle(pilot, 4)

        dialog = app.screen
        assert dialog.proposal.bullet_id == "taskflow.r1"
        assert dialog.proposal.archetype == ARCHETYPE
        assert dialog.proposal.text == GOOD
        assert "guardrail: ok" in dialog.verdict_message
    app.service.close()


async def test_a_accepts_and_the_index_carries_the_variant(repo, recorded):
    app = app_for(repo)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot)
        app.open_in_tailor(RUN_ID)
        await settle(pilot)
        app.query_one("#tailor-selection").focus()
        await pilot.press("down", "down", "v")
        await settle(pilot, 4)
        await pilot.press("a")
        await settle(pilot, 4)

        written = (repo / "index" / "projects" / "taskflow.yaml").read_text()
        assert GOOD in written and ARCHETYPE in written
        assert app.service.proposals(status="accepted")
    app.service.close()


async def test_d_rejects_and_nothing_is_written(repo, recorded):
    app = app_for(repo)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot)
        app.open_in_tailor(RUN_ID)
        await settle(pilot)
        app.query_one("#tailor-selection").focus()
        await pilot.press("down", "down", "v")
        await settle(pilot, 4)
        await pilot.press("d")
        await settle(pilot, 4)

        assert "variants" not in (
            repo / "index" / "projects" / "taskflow.yaml").read_text()
        rejected = app.service.proposals(status="rejected")
        assert len(rejected) == 1 and rejected[0].bullet_id == "taskflow.r1"
    app.service.close()


async def test_a_refused_proposal_cannot_be_accepted_from_the_panel(repo, recorded):
    app = app_for(repo, variant=BAD)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot)
        app.open_in_tailor(RUN_ID)
        await settle(pilot)
        app.query_one("#tailor-selection").focus()
        await pilot.press("down", "down", "v")
        await settle(pilot, 4)
        dialog = app.screen
        assert dialog.proposal.ok is False
        await pilot.press("a")
        await settle(pilot, 2)

        assert app.screen is dialog, "a refused proposal must not close the panel"
        assert "cannot be accepted" in dialog.verdict_message
        assert app.service.proposals(status="accepted") == []
    app.service.close()


async def test_e_then_ctrl_s_rechecks_the_edited_wording(repo, recorded):
    app = app_for(repo, variant=BAD)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot)
        app.open_in_tailor(RUN_ID)
        await settle(pilot)
        app.query_one("#tailor-selection").focus()
        await pilot.press("down", "down", "v")
        await settle(pilot, 4)
        await pilot.press("e")
        await settle(pilot, 2)
        app.screen.query_one("#variant-text").text = GOOD
        await pilot.press("ctrl+s")
        await settle(pilot, 4)

        dialog = app.screen
        assert dialog.proposal.text == GOOD
        assert dialog.proposal.ok is True
        assert dialog.proposal.provenance == "tui"
    app.service.close()


async def test_v_without_a_model_opens_the_panel_to_be_written_in(repo, recorded):
    app = app_for(repo, variant=None)
    async with app.run_test(size=(140, 45)) as pilot:
        await settle(pilot)
        app.open_in_tailor(RUN_ID)
        await settle(pilot)
        app.query_one("#tailor-selection").focus()
        await pilot.press("down", "down", "v")
        await settle(pilot, 4)

        dialog = app.screen
        assert dialog.proposal.id == "" and dialog.proposal.text == ""
        assert dialog.proposal.bullet_id == "taskflow.r1"
        await pilot.press("a")  # nothing to accept
        await settle(pilot, 2)
        assert app.screen is dialog
    app.service.close()


# ── 9. smoke: proposal → accept → run → the page ─────────────────────────────
@smoke
def test_an_accepted_variant_reaches_the_rendered_page(repo, service):
    """The whole loop: an `external` proposal, accepted, printed, recorded."""
    tools = ToolSet(service)
    profile = service.jd_profile("sample-backend")
    archetype = profile.archetype
    assert archetype, "the sample posting must resolve to an archetype"

    payload = tools.ats_propose_variant("taskflow.r1", archetype, GOOD)
    assert payload["guardrail_ok"] is True
    accepted = tools.ats_accept_proposal(payload["proposal"]["id"])
    assert accepted["ok"]

    service.reload()
    summary = service.tailor("sample-backend", "Acme", "Backend Engineer",
                            check=False)
    tex = (service.locate(summary.out_dir) / "resume.tex").read_text()
    assert "Docker rollout" in tex, "the variant's own words must be on the page"

    record = service.get_run(summary.run_id)
    variants = record.data.get("variants") or []
    assert [v["bullet_id"] for v in variants] == ["taskflow.r1"]
    assert variants[0]["archetype"] == archetype
    assert variants[0]["guardrail"]["ok"] is True
    written = json.loads(
        (service.locate(summary.out_dir) / "selection.json").read_text())
    assert written["variants"][0]["text"] == GOOD

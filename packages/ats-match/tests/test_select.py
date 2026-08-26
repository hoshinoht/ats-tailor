"""The budgeted greedy: constraints, pins, repair, determinism, and the objective.

The score matrix is written by hand rather than embedded, so every assertion is
about the selection logic and nothing about a model.
"""

import numpy as np
import pytest

from ats_match import (
    Claim,
    Constraints,
    Params,
    Requirement,
    SectionRule,
    coverage_at_budget,
    kendall_tau,
    precision_at_k,
    select,
)

SECTIONS = {
    "alpha": SectionRule(min_groups=1, max_groups=2, min_claims=2, max_claims=3,
                         group_cost=2, required_attr="headline"),
    "beta": SectionRule(min_groups=0, max_groups=2, min_claims=1, max_claims=2,
                        group_cost=1),
    "skill": SectionRule(min_groups=1, max_groups=2, min_claims=1, max_claims=8,
                         group_cost=1),
}


def requirements():
    return [
        Requirement(id="r-must", text="alpha capability", kind="must", terms=("alpha",)),
        Requirement(id="r-nice", text="beta capability", kind="nice", terms=("beta",)),
        Requirement(id="r-resp", text="daily duty", kind="responsibility", weight=0.5),
    ]


def claims():
    made = []
    for group in ("g1", "g2", "g3"):
        for j in range(3):
            made.append(
                Claim(
                    id=f"{group}.c{j}",
                    text=f"{group} claim {j}",
                    group=group,
                    section="alpha",
                    cost=2,
                    terms=("alpha",) if j == 0 else (),
                    prior=0.001 * j,
                    attrs={"headline": j == 2},
                )
            )
    made.append(Claim(id="s.alpha", text="alpha", group="cat-a", section="skill",
                      cost=0, terms=("alpha",), prior=0.03))
    made.append(Claim(id="s.beta", text="beta", group="cat-b", section="skill",
                      cost=0, terms=("beta",), prior=0.01))
    return made


def matrix_for(reqs, pool, hot=("g1.c0", "g2.c1", "s.alpha")):
    """A deterministic matrix: named claims match the first requirement strongly."""
    m = np.zeros((len(reqs), len(pool)))
    for c, claim in enumerate(pool):
        base = 0.2 if claim.section == "alpha" else 0.1
        m[0, c] = 0.95 if claim.id in hot else base
        m[1, c] = 0.6 if claim.id.endswith("c1") else base / 2
        m[2, c] = 0.3
    return m


def solve(budget=16, constraints=None, params=None, sim=None):
    reqs, pool = requirements(), claims()
    m = matrix_for(reqs, pool)
    cons = constraints or Constraints(budget=budget, sections=SECTIONS)
    return reqs, pool, select(m, reqs, pool, cons, params or Params(), claim_sim=sim)


# ── constraints ───────────────────────────────────────────────────────────────
def test_every_constraint_holds():
    _reqs, pool, selection = solve()
    by_id = {c.id: c for c in pool}
    assert selection.budget_used <= 16
    per_group = {}
    for cid in selection.claim_ids:
        per_group.setdefault(by_id[cid].group, []).append(cid)
    for group, held in per_group.items():
        rule = SECTIONS[by_id[held[0]].section]
        assert rule.min_claims <= len(held) <= rule.max_claims
    for section, groups in selection.group_ids_by_section.items():
        assert len(groups) <= SECTIONS[section].max_groups
        assert len(groups) >= SECTIONS[section].min_groups


def test_budget_is_never_exceeded_when_the_minimums_leave_room():
    _reqs, _pool, selection = solve(budget=9)
    assert selection.budget_used <= 9


def test_section_minimums_outrank_the_budget():
    """A budget below the section minimums yields the minimum, not an empty page.

    The caller shrinks the budget and re-solves when the page overflows; a
    solver that silently dropped the last group instead would make that loop
    unable to converge on anything showable.
    """
    _reqs, pool, selection = solve(budget=3)
    by_id = {c.id: c for c in pool}
    held = [c for c in selection.claim_ids if by_id[c].section == "alpha"]
    assert len(held) == SECTIONS["alpha"].min_claims
    assert selection.budget_used > 3


def test_group_cost_is_charged_once_per_opened_group():
    _reqs, pool, selection = solve()
    by_id = {c.id: c for c in pool}
    claim_cost = sum(by_id[cid].cost for cid in selection.claim_ids)
    opened = sum(len(groups) for groups in selection.group_ids_by_section.values())
    group_cost = sum(
        SECTIONS[section].group_cost * len(groups)
        for section, groups in selection.group_ids_by_section.items()
    )
    assert opened >= 2
    assert selection.budget_used == pytest.approx(claim_cost + group_cost)


def test_required_attribute_is_present_in_every_opened_group():
    _reqs, pool, selection = solve()
    by_id = {c.id: c for c in pool}
    for section, groups in selection.group_ids_by_section.items():
        if not SECTIONS[section].required_attr:
            continue
        for group in groups:
            held = [by_id[c] for c in selection.claim_ids if by_id[c].group == group]
            assert any(c.attrs.get("headline") for c in held), group


# ── pins and excludes ─────────────────────────────────────────────────────────
def test_pinned_claim_is_selected_and_opens_its_group():
    cons = Constraints(budget=16, sections=SECTIONS, pin_claims={"g3.c1"})
    _reqs, _pool, selection = solve(constraints=cons)
    assert "g3.c1" in selection.claim_ids
    assert "g3" in selection.group_ids_by_section["alpha"]


def test_excluded_claims_and_groups_never_appear():
    cons = Constraints(
        budget=16, sections=SECTIONS,
        exclude_claims={"g1.c0"}, exclude_groups={"g2"},
    )
    _reqs, pool, selection = solve(constraints=cons)
    by_id = {c.id: c for c in pool}
    assert "g1.c0" not in selection.claim_ids
    assert all(by_id[c].group != "g2" for c in selection.claim_ids)


def test_a_pinned_group_is_opened_and_filled_to_its_minimum():
    cons = Constraints(budget=20, sections=SECTIONS, pin_groups={"g3"})
    _reqs, pool, selection = solve(constraints=cons)
    by_id = {c.id: c for c in pool}
    held = [c for c in selection.claim_ids if by_id[c].group == "g3"]
    assert len(held) >= SECTIONS["alpha"].min_claims


# ── repair ────────────────────────────────────────────────────────────────────
def test_repair_reaches_min_groups_for_a_section_the_greedy_ignored():
    sections = dict(SECTIONS, beta=SectionRule(min_groups=1, max_groups=1,
                                               min_claims=1, max_claims=2, group_cost=1))
    reqs = requirements()
    pool = claims() + [
        Claim(id="b1.c0", text="uninteresting", group="b1", section="beta", cost=1)
    ]
    m = matrix_for(reqs, pool)
    m[:, -1] = 0.0  # the beta claim earns nothing on merit
    selection = select(m, reqs, pool, Constraints(budget=18, sections=sections), Params())
    assert "b1.c0" in selection.claim_ids
    assert selection.trace[-1].repair or any(s.repair for s in selection.trace)


def test_a_group_the_budget_cannot_fill_is_dropped_rather_than_shown_thin():
    _reqs, pool, selection = solve(budget=9)
    by_id = {c.id: c for c in pool}
    counts: dict[str, int] = {}
    for cid in selection.claim_ids:
        counts[by_id[cid].group] = counts.get(by_id[cid].group, 0) + 1
    for group, held in counts.items():
        section = by_id[f"{group}.c0" if group.startswith("g") else "s.alpha"].section
        assert held >= SECTIONS[section].min_claims


# ── objective ─────────────────────────────────────────────────────────────────
def test_adding_a_claim_never_lowers_coverage():
    """Submodular sanity: coverage is monotone in the selected set."""
    reqs, pool = requirements(), claims()
    m = matrix_for(reqs, pool)
    cons = Constraints(budget=30, sections=SECTIONS)
    previous = -1.0
    for budget in (6, 10, 14, 20, 30):
        selection = select(m, reqs, pool, Constraints(budget=budget, sections=cons.sections), Params())
        total = sum(selection.coverage.values())
        assert total >= previous - 1e-9
        previous = total


def test_redundancy_penalty_prefers_a_diverse_pair():
    """Three candidates, room for two: x is best, y nearly duplicates it, z does not."""
    reqs = [Requirement(id="r", text="one thing", kind="must")]
    pool = [
        Claim(id="x", text="x", group="gx", section="flat", cost=1),
        Claim(id="y", text="y", group="gy", section="flat", cost=1),
        Claim(id="z", text="z", group="gz", section="flat", cost=1),
    ]
    m = np.array([[0.9, 0.85, 0.8]])
    sim = np.eye(3)
    sim[0, 1] = sim[1, 0] = 0.99
    sections = {"flat": SectionRule(min_groups=1, max_groups=2, min_claims=1,
                                    max_claims=1, group_cost=0)}
    cons = Constraints(budget=2, sections=sections)
    lenient = select(m, reqs, pool, cons, Params(mu=0.0), claim_sim=sim)
    strict = select(m, reqs, pool, cons, Params(mu=2.0), claim_sim=sim)
    assert set(lenient.claim_ids) == {"x", "y"}
    assert set(strict.claim_ids) == {"x", "z"}


def test_context_bonus_pulls_in_the_skill_that_echoes_a_bullet_term():
    reqs, pool = requirements(), claims()
    m = matrix_for(reqs, pool)
    m[:, [i for i, c in enumerate(pool) if c.section == "skill"]] = 0.0
    cons = Constraints(budget=16, sections=SECTIONS)
    with_bonus = select(m, reqs, pool, cons, Params(beta=5.0))
    assert "s.alpha" in with_bonus.claim_ids


# ── gain floor ────────────────────────────────────────────────────────────────
def _floor_setup():
    """One free section whose claims earn almost nothing, and one that earns a lot."""
    reqs = [Requirement(id="r", text="one thing", kind="must")]
    pool = [
        Claim(id="s.good", text="good", group="cat", section="skill", cost=0, prior=0.03),
        Claim(id="s.weak", text="weak", group="cat", section="skill", cost=0, prior=0.02),
        Claim(id="s.noise", text="noise", group="cat", section="skill", cost=0, prior=0.01),
    ]
    m = np.array([[0.9, 0.02, 0.0]])
    return reqs, pool, m


def test_a_section_gain_floor_keeps_free_noise_off_the_output():
    reqs, pool, m = _floor_setup()
    sections = {"skill": SectionRule(min_groups=1, max_groups=1, min_claims=1,
                                     max_claims=8, group_cost=1)}
    cons = Constraints(budget=6, sections=sections)
    everything = select(m, reqs, pool, cons, Params())
    assert set(everything.claim_ids) == {"s.good", "s.weak", "s.noise"}

    sections["skill"].min_gain = 0.05
    filtered = select(m, reqs, pool, cons, Params())
    assert filtered.claim_ids == ["s.good"]


def test_the_floor_can_be_set_globally_and_overridden_per_section():
    reqs, pool, m = _floor_setup()
    sections = {"skill": SectionRule(min_groups=1, max_groups=1, min_claims=1,
                                     max_claims=8, group_cost=1)}
    cons = Constraints(budget=6, sections=sections)
    assert select(m, reqs, pool, cons, Params(min_gain=0.05)).claim_ids == ["s.good"]
    sections["skill"].min_gain = 0.0  # the section overrides the default
    assert len(select(m, reqs, pool, cons, Params(min_gain=0.05)).claim_ids) == 3


def test_the_floor_does_not_block_reaching_a_minimum():
    """Filling ``min_claims`` is an obligation, not a bid — the floor must yield."""
    reqs, pool, m = _floor_setup()
    sections = {
        "skill": SectionRule(min_groups=1, max_groups=1, min_claims=2, max_claims=8,
                             group_cost=1, min_gain=5.0)  # nothing can clear this
    }
    selection = select(m, reqs, pool, Constraints(budget=6, sections=sections), Params())
    assert len(selection.claim_ids) == 2
    assert all(step.repair for step in selection.trace)
    assert selection.claim_ids[0] == "s.good"  # the best available, by gain then prior


def test_a_pinned_group_is_opened_even_when_nothing_clears_the_floor():
    reqs, pool, m = _floor_setup()
    pool.append(Claim(id="o.one", text="other", group="other", section="skill",
                      cost=0, prior=0.0))
    m = np.hstack([m, np.zeros((1, 1))])
    sections = {"skill": SectionRule(min_groups=1, max_groups=2, min_claims=1,
                                     max_claims=8, group_cost=1, min_gain=0.05)}
    cons = Constraints(budget=6, sections=sections, pin_groups={"other"})
    selection = select(m, reqs, pool, cons, Params())
    assert "other" in selection.group_ids_by_section["skill"]
    assert "o.one" in selection.claim_ids
    assert "s.noise" not in selection.claim_ids


# ── header claims ─────────────────────────────────────────────────────────────
def _with_headers():
    """Each group gets a header claim that answers the second requirement."""
    reqs = requirements()
    pool = claims()
    for group in ("g1", "g2", "g3"):
        pool.append(
            Claim(id=f"{group}.head", text=f"{group} header", group=group,
                  section="alpha", cost=99, terms=("beta",),
                  attrs={"header": True, "headline": True})
        )
    m = matrix_for(reqs, pool)
    for c, claim in enumerate(pool):
        if claim.attrs.get("header"):
            m[:, c] = [0.1, 0.9, 0.1]  # only a header answers r-nice well
    return reqs, pool, m


def test_a_header_arrives_with_its_group_and_is_never_chosen_alone():
    reqs, pool, m = _with_headers()
    selection = select(m, reqs, pool, Constraints(budget=16, sections=SECTIONS), Params())
    by_id = {c.id: c for c in pool}
    for group in selection.group_ids_by_section.get("alpha", []):
        assert f"{group}.head" in selection.claim_ids
    for claim_id in selection.claim_ids:
        if by_id[claim_id].attrs.get("header"):
            assert any(
                cid != claim_id and by_id[cid].group == by_id[claim_id].group
                for cid in selection.claim_ids
            ), "a header was included without any claim of its group"


def test_a_header_costs_nothing_and_does_not_count_as_a_bullet():
    reqs, pool, m = _with_headers()
    selection = select(m, reqs, pool, Constraints(budget=16, sections=SECTIONS), Params())
    by_id = {c.id: c for c in pool}
    assert selection.budget_used <= 16  # despite each header declaring cost 99
    for group in selection.group_ids_by_section.get("alpha", []):
        held = [
            c for c in selection.claim_ids
            if by_id[c].group == group and not by_id[c].attrs.get("header")
        ]
        assert len(held) <= SECTIONS["alpha"].max_claims


def test_a_header_leaves_when_its_group_is_dropped():
    reqs, pool, m = _with_headers()
    wide = select(m, reqs, pool, Constraints(budget=16, sections=SECTIONS), Params())
    narrow = select(m, reqs, pool, Constraints(budget=8, sections=SECTIONS), Params())
    by_id = {c.id: c for c in pool}
    gone = {
        g for g in wide.group_ids_by_section.get("alpha", [])
        if g not in narrow.group_ids_by_section.get("alpha", [])
    }
    assert gone, "the narrow budget should have closed at least one group"
    for group in gone:
        assert f"{group}.head" not in narrow.claim_ids
    for claim_id in narrow.claim_ids:
        assert by_id[claim_id].group in {
            by_id[c].group for c in narrow.claim_ids if not by_id[c].attrs.get("header")
        }


def test_opening_a_group_is_worth_what_its_header_covers():
    """The whole point: a header's coverage shows up in the opening claim's gain."""
    reqs, pool, m = _with_headers()
    plain = [c for c in pool if not c.attrs.get("header")]
    plain_m = m[:, [i for i, c in enumerate(pool) if not c.attrs.get("header")]]
    cons = Constraints(budget=16, sections=SECTIONS)

    with_header = select(m, reqs, pool, cons, Params())
    without = select(plain_m, reqs, plain, cons, Params())
    opening = next(s for s in with_header.trace if s.opened_group and not s.repair)
    same = next((s for s in without.trace if s.claim_id == opening.claim_id), None)
    assert same is None or opening.gain >= same.gain
    assert with_header.coverage["r-nice"] > without.coverage["r-nice"]


def test_the_header_gets_its_own_trace_step():
    reqs, pool, m = _with_headers()
    selection = select(m, reqs, pool, Constraints(budget=16, sections=SECTIONS), Params())
    headers = [s for s in selection.trace if s.claim_id.endswith(".head")]
    assert headers
    for step in headers:
        assert step.opened_group == step.claim_id.split(".")[0]


def test_selection_is_deterministic():
    runs = [solve()[2].claim_ids for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_trace_explains_each_pick():
    _reqs, _pool, selection = solve()
    assert selection.trace
    assert [s.claim_id for s in selection.trace] == selection.claim_ids
    first = selection.trace[0]
    assert first.opened_group is not None
    assert all(isinstance(pair[0], str) for step in selection.trace for pair in step.covered)


def test_empty_inputs_do_not_crash():
    selection = select(np.zeros((0, 0)), [], [], Constraints(budget=5, sections={}), Params())
    assert selection.claim_ids == []
    assert selection.objective == 0.0


# ── metrics ───────────────────────────────────────────────────────────────────
def test_coverage_at_budget_weighs_by_kind():
    reqs = requirements()
    full = {r.id: 1.0 for r in reqs}
    assert coverage_at_budget(full, reqs) == 1.0
    assert coverage_at_budget({}, reqs) == 0.0
    assert coverage_at_budget(full, []) is None


def test_precision_and_tau_move_the_expected_way():
    assert precision_at_k(["a", "b", "c"], ["a", "b"]) == 1.0
    assert precision_at_k(["c", "d", "a"], ["a", "b"]) == 0.0
    assert precision_at_k(["a"], []) is None
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["c", "b", "a"], ["a", "b", "c"]) == -1.0
    assert kendall_tau(["a"], ["a", "b"]) is None

"""Budgeted greedy maximisation of requirement coverage.

The objective (design §5)::

    F(S) = Σ_r w_r · softpool_τ(M[r, S])        coverage, submodular
         + β · ctx(S)                            keyword-in-context bonus
         − μ · redundancy(S)                     near-duplicates inside a section
         + Σ_{c ∈ S} prior_c                     small additive item priors

``softpool_τ(x) = τ · log(1 + Σ_i (exp(x_i / τ) − 1))`` — plain log-sum-exp
would give the empty set ``−∞`` and would let a pile of zero-scoring claims
accumulate ``τ·log|S|`` of imaginary coverage. Subtracting one inside the sum
fixes both: the empty set scores 0, a zero-scoring claim adds nothing, a single
claim scores exactly its own match, and as ``τ → 0`` the pool becomes the max.

Selection is a lazy greedy over *gain per unit cost* (the cost-benefit variant
— with a budget rather than a cardinality limit, ranking by raw gain would
spend the whole page on the longest claims), followed by a repair pass that
enforces the minimums the greedy is free to ignore.

A claim marked ``attrs["header"]`` is never chosen: it comes in free with its
group and leaves with it. Opening a group already costs ``group_cost``, and
whatever that cost buys is displayed — so its content has to count, or the
solver undervalues opening a group by exactly the part of it a reader sees
first.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .lexical import normalise
from .types import Claim, Constraints, Params, Requirement, SectionRule, Selection, Step

_PERMISSIVE = SectionRule(
    min_groups=0, max_groups=1 << 30, min_claims=0, max_claims=1 << 30
)
_MAX_REPAIR_ROUNDS = 12


def select(
    matrix,
    requirements: Sequence[Requirement],
    claims: Sequence[Claim],
    constraints: Constraints,
    params: Params | None = None,
    claim_sim=None,
) -> Selection:
    """Pick the claims that maximise the objective inside ``constraints``."""
    solver = _Solver(matrix, requirements, claims, constraints, params or Params(), claim_sim)
    solver.run()
    return solver.result()


class _Solver:
    def __init__(self, matrix, requirements, claims, constraints, params, claim_sim):
        self.requirements = list(requirements)
        self.claims = list(claims)
        self.constraints = constraints
        self.params = params

        n_req, n_claim = len(self.requirements), len(self.claims)
        self.matrix = np.asarray(matrix, dtype=float).reshape(n_req, n_claim)
        self.sim = (
            np.asarray(claim_sim, dtype=float).reshape(n_claim, n_claim)
            if claim_sim is not None
            else np.zeros((n_claim, n_claim), dtype=float)
        )
        self.weights = np.array(
            [
                float(r.weight) * float(params.kind_weights.get(r.kind, 1.0))
                for r in self.requirements
            ],
            dtype=float,
        )
        tau = max(float(params.tau), 1e-6)
        self.tau = tau
        # exp(M/τ) − 1, the per-claim contribution to a requirement's pool.
        self.contrib = np.expm1(self.matrix / tau)

        self.section_of_group: dict[str, str] = {}
        self.claims_in_group: dict[str, list[int]] = defaultdict(list)
        self.header_of_group: dict[str, int] = {}
        for i, claim in enumerate(self.claims):
            self.section_of_group.setdefault(claim.group, claim.section)
            if claim.attrs.get("header"):
                self.header_of_group.setdefault(claim.group, i)
            else:
                self.claims_in_group[claim.group].append(i)

        self.must_terms = {
            normalise(term)
            for r in self.requirements
            if r.kind == "must"
            for term in r.terms
            if str(term).strip()
        }
        attr = params.context_terms_attr
        self.claim_terms = [
            {
                normalise(t)
                for t in (claim.attrs.get(attr, claim.terms) if attr else claim.terms)
                if str(t).strip()
            }
            & self.must_terms
            for claim in self.claims
        ]

        self.available = [
            i
            for i, claim in enumerate(self.claims)
            if not claim.attrs.get("header")
            and claim.id not in constraints.exclude_claims
            and claim.group not in constraints.exclude_groups
        ]
        self.pinned: set[int] = {
            i for i in self.available if self.claims[i].id in constraints.pin_claims
        }
        self._reset()

    # ── state ────────────────────────────────────────────────────────────────
    def _reset(self) -> None:
        self.selected: list[int] = []
        self.steps: dict[int, Step] = {}
        #: Inclusion order for the trace: chosen claims plus the headers that
        #: came in with their groups.
        self.trace_order: list[int] = []
        self._apply_state()

    def _apply_state(self) -> None:
        """Recompute every incremental accumulator from ``self.selected``.

        Headers are derived, never stored: a group's header is included exactly
        while the group is open, so dropping the group's last claim takes its
        header with it without any bookkeeping.
        """
        self.chosen = set(self.selected)
        self.acc = np.zeros(len(self.requirements), dtype=float)
        self.group_claims: dict[str, list[int]] = defaultdict(list)
        self.open_groups: dict[str, list[str]] = defaultdict(list)
        self.context_hits: dict[str, int] = defaultdict(int)
        self.other_hits: dict[str, int] = defaultdict(int)
        self.included: list[int] = []
        self.used = 0.0

        def include(index: int, free: bool = False) -> None:
            claim = self.claims[index]
            self.acc += self.contrib[:, index]
            hits = (
                self.context_hits
                if claim.section == self.params.context_section
                else self.other_hits
            )
            for term in self.claim_terms[index]:
                hits[term] += 1
            if not free:
                self.used += float(claim.cost)
            self.included.append(index)

        def open_group(group: str, section: str) -> None:
            self.open_groups[section].append(group)
            self.used += self.rule(section).group_cost
            header = self.header_of_group.get(group)
            if header is not None:
                include(header, free=True)

        for i in self.selected:
            claim = self.claims[i]
            if claim.group not in self.group_claims:
                open_group(claim.group, claim.section)
            self.group_claims[claim.group].append(i)
            include(i)
        for group in self.constraints.pin_groups:
            section = self.section_of_group.get(group)
            if section and group not in self.group_claims and group not in self.open_groups[section]:
                self.group_claims[group] = []
                open_group(group, section)
        self.trace_order = [i for i in self.trace_order if i in set(self.included)]
        for i in list(self.steps):
            if i not in set(self.included):
                self.steps.pop(i, None)

    def rule(self, section: str) -> SectionRule:
        return self.constraints.sections.get(section, _PERMISSIVE)

    def min_gain(self, section: str) -> float:
        """The gain a claim of ``section`` must earn before it is worth choosing."""
        floor = self.rule(section).min_gain
        return float(self.params.min_gain if floor is None else floor)

    def worth_choosing(self, i: int) -> bool:
        return self.gain(i)[0] >= self.min_gain(self.claims[i].section)

    # ── objective ────────────────────────────────────────────────────────────
    def _softpool(self, acc: np.ndarray) -> np.ndarray:
        return self.tau * np.log1p(np.maximum(acc, 0.0))

    def _context_rate(self) -> float:
        if not self.must_terms:
            return 0.0
        both = sum(
            1
            for term in self.must_terms
            if self.context_hits.get(term, 0) and self.other_hits.get(term, 0)
        )
        return both / len(self.must_terms)

    def _penalised(self, section: str) -> bool:
        return (
            self.params.redundancy_in_context
            or section != self.params.context_section
        )

    def _redundancy_delta(self, i: int) -> float:
        section = self.claims[i].section
        if not self._penalised(section):
            return 0.0
        floor = self.params.redundancy_floor
        total = 0.0
        for j in self.included:
            if self.claims[j].section != section:
                continue
            total += max(0.0, float(self.sim[i, j]) - floor)
        return total

    def arrivals(self, i: int) -> list[int]:
        """Everything that lands on the output when claim ``i`` is chosen.

        Just the claim, unless choosing it opens its group — then the group's
        header arrives with it, and the gain has to say so.
        """
        if self.claims[i].group in self.group_claims:
            return [i]
        header = self.header_of_group.get(self.claims[i].group)
        return [i] if header is None else [i, header]

    def _context_delta(self, columns: Sequence[int]) -> float:
        terms: set[str] = set()
        for i in columns:
            terms |= self.claim_terms[i]
        if not self.must_terms or not terms:
            return 0.0
        mine, theirs = (
            (self.context_hits, self.other_hits)
            if self.claims[columns[0]].section == self.params.context_section
            else (self.other_hits, self.context_hits)
        )
        gained = sum(
            1 for term in terms if not mine.get(term, 0) and theirs.get(term, 0)
        )
        return gained / len(self.must_terms)

    def _coverage_delta(self, columns: Sequence[int]) -> tuple[float, np.ndarray]:
        before = self._softpool(self.acc)
        after = self._softpool(self.acc + self.contrib[:, list(columns)].sum(axis=1))
        per_requirement = self.weights * (after - before)
        return float(per_requirement.sum()), per_requirement

    def gain(self, i: int) -> tuple[float, np.ndarray]:
        columns = self.arrivals(i)
        coverage, per_requirement = self._coverage_delta(columns)
        value = (
            coverage
            + self.params.beta * self._context_delta(columns)
            - self.params.mu * self._redundancy_delta(i)
            + float(self.claims[i].prior)
        )
        return value, per_requirement

    # ── feasibility ──────────────────────────────────────────────────────────
    def effective_cost(self, i: int) -> float:
        claim = self.claims[i]
        cost = float(claim.cost)
        if claim.group not in self.group_claims:
            cost += self.rule(claim.section).group_cost
        return cost

    def feasible(self, i: int, ignore_budget: bool = False) -> bool:
        if i in self.chosen:
            return False
        claim = self.claims[i]
        rule = self.rule(claim.section)
        if len(self.group_claims.get(claim.group, ())) >= rule.max_claims:
            return False
        if claim.group not in self.group_claims and len(self.open_groups[claim.section]) >= rule.max_groups:
            return False
        if not ignore_budget and self.used + self.effective_cost(i) > self.constraints.budget + 1e-9:
            return False
        return True

    # ── mutation ─────────────────────────────────────────────────────────────
    def _covered(self, per_requirement: np.ndarray) -> list[tuple[str, float]]:
        return sorted(
            (
                (self.requirements[r].id, float(per_requirement[r]))
                for r in np.nonzero(per_requirement > 1e-9)[0]
            ),
            key=lambda pair: -pair[1],
        )[:5]

    def add(self, i: int, repair: bool = False) -> None:
        value, per_requirement = self.gain(i)
        claim = self.claims[i]
        opened = claim.group if claim.group not in self.group_claims else None
        header = self.header_of_group.get(claim.group) if opened else None
        if header is not None:
            # What the header brings on its own, so the trace shows the two
            # halves of "opening this group" separately.
            own_value, own_per_requirement = self._coverage_delta([header])
            self.steps[header] = Step(
                claim_id=self.claims[header].id, gain=own_value, opened_group=opened,
                covered=self._covered(own_per_requirement), repair=repair,
            )
            self.trace_order.append(header)
        self.selected.append(i)
        self.steps[i] = Step(
            claim_id=claim.id, gain=value, opened_group=opened,
            covered=self._covered(per_requirement), repair=repair,
        )
        self.trace_order.append(i)
        self._apply_state()

    def drop(self, indices: Iterable[int]) -> None:
        removed = set(indices)
        self.selected = [i for i in self.selected if i not in removed]
        for i in removed:
            self.steps.pop(i, None)
        self._apply_state()

    # ── greedy ───────────────────────────────────────────────────────────────
    def _priority(self, i: int) -> float:
        value, _ = self.gain(i)
        return value / max(self.effective_cost(i), self.params.min_cost)

    def _key(self, i: int, stamp: int) -> tuple:
        return (
            -round(self._priority(i), 9),
            -round(float(self.claims[i].prior), 9),
            self.claims[i].id,
            i,
            stamp,
        )

    def seed_pins(self) -> None:
        for i in sorted(self.pinned, key=lambda i: self.claims[i].id):
            if self.feasible(i, ignore_budget=True):
                self.add(i)
        self._apply_state()

    def greedy(self) -> None:
        stamp = 0
        heap = [self._key(i, stamp) for i in self.available if self.feasible(i)]
        heapq.heapify(heap)
        while heap:
            entry = heapq.heappop(heap)
            i, entry_stamp = entry[3], entry[4]
            if i in self.chosen:
                continue
            if not self.feasible(i):
                continue
            if entry_stamp == stamp:
                if self._priority(i) < 0.0 or not self.worth_choosing(i):
                    # Above the section's floor, a claim earning nothing new is
                    # still taken while the budget allows — an unfilled page is
                    # signal thrown away, and the tie-break orders such claims
                    # by prior (recency, impact, proficiency), not arbitrarily.
                    continue
                self.add(i)
                stamp += 1
                # A freshly opened group makes its siblings cheaper, which the
                # lazy bound does not anticipate: re-price them right away.
                opened = self.claims[i].group
                for sibling in self.claims_in_group[opened]:
                    if sibling in self.chosen or sibling not in self.available:
                        continue
                    if self.feasible(sibling):
                        heapq.heappush(heap, self._key(sibling, stamp))
                continue
            heapq.heappush(heap, self._key(i, stamp))

    # ── repair ───────────────────────────────────────────────────────────────
    def _candidates_in(self, group: str) -> list[int]:
        return [
            i
            for i in self.claims_in_group[group]
            if i in self.available and i not in self.chosen
        ]

    def _best(self, indices: Sequence[int]) -> int | None:
        ranked = sorted(
            indices,
            key=lambda i: (-round(self.gain(i)[0], 9), -self.claims[i].prior, self.claims[i].id),
        )
        return ranked[0] if ranked else None

    def _fill_min_claims(self) -> bool:
        changed = False
        for section, groups in list(self.open_groups.items()):
            rule = self.rule(section)
            for group in list(groups):
                while len(self.group_claims.get(group, ())) < rule.min_claims:
                    best = self._best(self._candidates_in(group))
                    if best is None or not self.feasible(best, ignore_budget=True):
                        break
                    self.add(best, repair=True)
                    changed = True
        return changed

    def _fill_min_groups(self) -> bool:
        changed = False
        for section, rule in self.constraints.sections.items():
            while len(self.open_groups[section]) < rule.min_groups:
                options = [
                    group
                    for group, group_section in self.section_of_group.items()
                    if group_section == section
                    and group not in self.group_claims
                    and group not in self.constraints.exclude_groups
                    and self._candidates_in(group)
                ]
                if not options:
                    break
                best_by_group = {
                    group: self._best(self._candidates_in(group)) for group in options
                }
                ranked = sorted(
                    (g for g, i in best_by_group.items() if i is not None),
                    key=lambda g: (-round(self.gain(best_by_group[g])[0], 9), g),
                )
                if not ranked:
                    break
                chosen = best_by_group[ranked[0]]
                if not self.feasible(chosen, ignore_budget=True):
                    break
                self.add(chosen, repair=True)
                changed = True
        return changed

    def _attr_holders(self, group: str, attr: str) -> list[int]:
        return [i for i in self.group_claims.get(group, ()) if self.claims[i].attrs.get(attr)]

    def _fill_required_attr(self) -> bool:
        """Give every opened group one claim carrying the section's required attribute.

        When the group is full or the budget is spent, this is a *swap* rather
        than an addition — adding and letting :meth:`_shrink` undo it would
        oscillate, since the new claim is exactly the one with the lowest gain.
        """
        changed = False
        for section, groups in list(self.open_groups.items()):
            rule = self.rule(section)
            attr = rule.required_attr
            if not attr:
                continue
            for group in list(groups):
                held = self.group_claims.get(group, [])
                if not held or self._attr_holders(group, attr):
                    continue
                best = self._best(
                    [i for i in self._candidates_in(group) if self.claims[i].attrs.get(attr)]
                )
                if best is None:
                    continue
                over_budget = (
                    self.used + self.effective_cost(best) > self.constraints.budget + 1e-9
                )
                if len(held) >= rule.max_claims or over_budget:
                    droppable = [i for i in held if i not in self.pinned]
                    if not droppable:
                        continue
                    self.drop([
                        min(droppable, key=lambda i: (self._recorded_gain(i), self.claims[i].id))
                    ])
                if self.feasible(best, ignore_budget=True):
                    self.add(best, repair=True)
                    changed = True
        return changed

    def _recorded_gain(self, i: int) -> float:
        return float(self.steps[i].gain) if i in self.steps else 0.0

    def _sole_attr_holder(self, i: int) -> bool:
        """True when dropping ``i`` would leave its group without the required attribute."""
        claim = self.claims[i]
        attr = self.rule(claim.section).required_attr
        if not attr or not claim.attrs.get(attr):
            return False
        return len(self._attr_holders(claim.group, attr)) <= 1

    def _shrink(self) -> bool:
        """Drop the least valuable non-pinned claim (or group) to fit the budget."""
        loose = [
            i
            for i in self.selected
            if i not in self.pinned
            and not self._sole_attr_holder(i)
            and len(self.group_claims[self.claims[i].group])
            > self.rule(self.claims[i].section).min_claims
        ]
        if loose:
            self.drop([min(loose, key=lambda i: (self._recorded_gain(i), self.claims[i].id))])
            return True

        droppable_groups = []
        for section, groups in self.open_groups.items():
            if len(groups) <= self.rule(section).min_groups:
                continue
            for group in groups:
                if group in self.constraints.pin_groups:
                    continue
                members = self.group_claims.get(group, [])
                if any(i in self.pinned for i in members):
                    continue
                total = sum(self.steps[i].gain for i in members if i in self.steps)
                droppable_groups.append((total, group, members))
        if droppable_groups:
            _total, _group, members = min(droppable_groups, key=lambda item: (item[0], item[1]))
            self.drop(members)
            return True
        return False

    def _prune_thin_groups(self) -> None:
        """A group the budget could not fill to its minimum is better not shown."""
        for section, groups in list(self.open_groups.items()):
            rule = self.rule(section)
            for group in list(groups):
                members = self.group_claims.get(group, [])
                if len(members) >= rule.min_claims or any(i in self.pinned for i in members):
                    continue
                if group in self.constraints.pin_groups:
                    continue
                if len(self.open_groups[section]) <= rule.min_groups:
                    continue
                self.drop(members)

    def repair(self) -> None:
        for _ in range(_MAX_REPAIR_ROUNDS):
            changed = self._fill_min_claims()
            changed = self._fill_min_groups() or changed
            changed = self._fill_required_attr() or changed
            if self.used <= self.constraints.budget + 1e-9:
                if not changed:
                    return
                continue
            if not self._shrink():
                return
        return

    def _affordable(self, group: str) -> float | None:
        """Cost of opening ``group`` and filling it to its minimum, or ``None``."""
        section = self.section_of_group[group]
        rule = self.rule(section)
        candidates = sorted(
            self._candidates_in(group),
            key=lambda i: (-round(self.gain(i)[0], 9), self.claims[i].id),
        )[: max(rule.min_claims, 1)]
        if not candidates:
            return None
        cost = rule.group_cost + sum(float(self.claims[i].cost) for i in candidates)
        return cost if self.used + cost <= self.constraints.budget + 1e-9 else None

    def _open_to_max(self, pay_by_shrinking: bool = False) -> None:
        """Open every section up to ``max_groups`` while the budget allows.

        Groups are taken best-first by the gain of their strongest claim, and
        each is filled straight to ``min_claims`` — an opened group below its
        minimum is not a thing worth showing, so it is not a thing worth
        opening halfway.

        ``max_groups`` is a layout decision, not an objective term: the caller
        said how many groups the output has room for, and the objective cannot
        see the header a group brings with it. So with ``pay_by_shrinking`` the
        remaining budget is *made* by dropping the weakest claims of the groups
        already open (never below their minimums, never a pin).
        """
        skipped: set[str] = set()
        while True:
            ranked: list[tuple[float, str, int]] = []
            for group, section in self.section_of_group.items():
                if group in skipped or group in self.group_claims:
                    continue
                if group in self.constraints.exclude_groups:
                    continue
                rule = self.rule(section)
                if len(self.open_groups[section]) >= rule.max_groups:
                    continue
                best = self._best(self._candidates_in(group))
                if best is None or not self.worth_choosing(best):
                    continue
                ranked.append((self.gain(best)[0], group, best))
            if not ranked:
                return
            ranked.sort(key=lambda item: (-round(item[0], 9), item[1]))
            opened = False
            for _gain, group, best in ranked:
                while self._affordable(group) is None and pay_by_shrinking:
                    if not self._shrink():
                        break
                if self._affordable(group) is None:
                    skipped.add(group)
                    continue
                self.add(best, repair=True)
                rule = self.rule(self.section_of_group[group])
                while len(self.group_claims.get(group, ())) < rule.min_claims:
                    nxt = self._best(self._candidates_in(group))
                    if nxt is None or not self.feasible(nxt):
                        break
                    self.add(nxt, repair=True)
                opened = True
                break
            if not opened:
                return

    # ── driver ───────────────────────────────────────────────────────────────
    def run(self) -> None:
        self.seed_pins()
        self.greedy()
        self.repair()
        if self.params.breadth_first:
            # After the greedy, not before: depth is chosen on merit, and
            # breadth is then paid for out of the claims that earned least.
            self._open_to_max(pay_by_shrinking=True)
            self.repair()
        self.greedy()
        self.repair()
        self._prune_thin_groups()

    def result(self) -> Selection:
        coverage = {
            requirement.id: float(value)
            for requirement, value in zip(self.requirements, self._softpool(self.acc))
        }
        objective = float(
            (self.weights * self._softpool(self.acc)).sum()
            + self.params.beta * self._context_rate()
            - self.params.mu * self._redundancy()
            + sum(float(self.claims[i].prior) for i in self.selected)
        )
        return Selection(
            claim_ids=[self.claims[i].id for i in self.included],
            group_ids_by_section={
                section: list(groups) for section, groups in self.open_groups.items() if groups
            },
            coverage=coverage,
            objective=objective,
            trace=[self.steps[i] for i in self.trace_order if i in self.steps],
            budget_used=float(self.used),
        )

    def _redundancy(self) -> float:
        floor = self.params.redundancy_floor
        total = 0.0
        for a in range(len(self.included)):
            for b in range(a + 1, len(self.included)):
                i, j = self.included[a], self.included[b]
                if self.claims[i].section != self.claims[j].section:
                    continue
                if not self._penalised(self.claims[i].section):
                    continue
                total += max(0.0, float(self.sim[i, j]) - floor)
        return total

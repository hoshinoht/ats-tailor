"""The seam between the index and the matching core.

One direction turns the index into :class:`ats_match.Claim` objects — one per
bullet, one per skill — with the costs, terms and priors the solver needs. The
other turns a :class:`ats_match.Selection` back into the template context
``rendering.render_resume`` already consumes, so v2 changes what is chosen
without changing what a resume looks like.

Everything resume-shaped lives here; ``ats_match`` sees only groups, sections
and costs.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from ats_match import Claim, Requirement, SectionRule

from . import config
from .atscheck import count_in_text
from .index_io import to_legacy
from .loaders import parse_start_date, recency_multiplier
from .models import Index, Project, Role
from .rendering import build_project_context, build_role_context

#: ``recency_multiplier`` returns 1.00–1.10; priors want 0–1.
_RECENCY_SPAN = 0.10
#: Additive priors, small by design: they break ties, they do not rank.
PRIOR_RECENCY = 0.05
PRIOR_IMPACT = 0.02
PROFICIENCY_PRIOR = {"advanced": 0.03, "intermediate": 0.01, "familiar": 0.0}
#: Skills per rendered skills line.
MAX_SKILLS_PER_LINE = 8
#: A group header costs a header line plus the spacing after the entry; a
#: rendered study entry adds its ``Technologies:`` line on top of that.
ROLE_GROUP_COST = 2
PROJECT_GROUP_COST = 3
#: Kept from v1: the rendered line says "Languages", not "Programming Languages".
CATEGORY_LABELS = {"Programming Languages": "Languages"}
#: Claim attribute holding the vocabulary a claim's *rendered* text carries, as
#: opposed to the vocabulary it is retrieved by. A bullet tagged ``docker`` need
#: not say "Docker" anywhere, and the keyword check reads the page, not the tags.
PAGE_TERMS_ATTR = "page_terms"


def recency_prior(period) -> float:
    """``recency_multiplier`` rescaled to ``[0, 1]``."""
    text = period.format() if hasattr(period, "format") else str(period)
    return max(0.0, min(1.0, (recency_multiplier(text) - 1.0) / _RECENCY_SPAN))


def surface_forms(index: Index, skill_ids: Iterable[str]) -> list[str]:
    skills = index.skill_map()
    forms: list[str] = []
    for skill_id in skill_ids:
        skill = skills.get(skill_id)
        if skill is not None:
            forms.extend(skill.surface_forms())
    return forms


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(value).strip())
    return tuple(out)


def visible_items(index: Index) -> list[tuple[str, Role | Project]]:
    """``(section, item)`` for everything that may appear on the page."""
    return [("role", r) for r in index.roles if not r.hidden] + [
        ("project", p) for p in index.projects
    ]


def claims_from_index(
    index: Index,
    chars_per_line: int | None = None,
    text_overrides: Mapping[str, str] | None = None,
) -> list[Claim]:
    """One claim per bullet of every visible entry.

    ``cost`` is the number of rendered lines the bullet is expected to take, on
    the same characters-per-line estimate ``selection.estimate_line_count``
    uses, so the solver's budget and the page-fit estimate speak one unit.

    ``text_overrides`` (bullet id → replacement text) is substituted *here*, so
    the edited wording is what gets embedded, what gets costed, and what the
    keyword-in-context bonus reads — a per-run rephrase changes the decision,
    not only the page.
    """
    chars_per_line = int(chars_per_line or config.CHARS_PER_BULLET_LINE)
    text_overrides = dict(text_overrides or {})
    skills = index.skill_map()
    claims: list[Claim] = []
    for section, item in visible_items(index):
        item_terms = [*surface_forms(index, item.skills), *item.keywords]
        prior_base = PRIOR_RECENCY * recency_prior(item.period)
        for bullet in item.bullets:
            text = str(text_overrides.get(bullet.id, bullet.text)).strip()
            if not text:
                continue
            claims.append(
                Claim(
                    id=bullet.id,
                    text=text,
                    group=item.id,
                    section=section,
                    cost=max(1, math.ceil(len(text) / max(chars_per_line, 1))),
                    terms=_dedupe([*surface_forms(index, bullet.skills), *item_terms]),
                    prior=prior_base + (PRIOR_IMPACT if bullet.kind == "impact" else 0.0),
                    attrs={
                        "kind": bullet.kind,
                        "item_kind": section,
                        "impact": bullet.kind == "impact",
                        PAGE_TERMS_ATTR: page_terms(
                            text, [*bullet.skills, *item.skills], item.keywords, skills
                        ),
                    },
                )
            )
    return claims


def header_claims(index: Index) -> list[Claim]:
    """One claim per entry for the header line the page prints above its bullets.

    A rendered entry is more than its bullets — "Software Engineering (Cloud)
    Intern -- Razer", or a project's tagline and its Technologies line — and an
    ATS reads all of it. The solver already pays ``group_cost`` for those lines;
    this is what that cost buys, so it has to count. The claim is never chosen
    on its own: ``ats_match`` includes it when the group opens and drops it when
    the group goes.
    """
    _skills, legacy_projects, _exp, _certs, _summary = to_legacy(index)
    project_context = {
        entry["id"]: build_project_context(entry) for entry in legacy_projects["projects"]
    }
    skills = index.skill_map()

    claims: list[Claim] = []
    for section, item in visible_items(index):
        if section == "role":
            text = f"{item.title} -- {item.company}"
        else:
            context = project_context.get(item.id, {})
            tagline = context.get("tagline") or item.summary
            tech_line = context.get("tech_line") or ""
            text = f"{item.name} -- {tagline}."
            if tech_line:
                text = f"{text} Technologies: {tech_line}"
        shown = page_terms(text, item.skills, item.keywords, skills)
        claims.append(
            Claim(
                id=f"{item.id}.header",
                text=text,
                group=item.id,
                section=section,
                cost=0.0,
                terms=_dedupe([*shown, *surface_forms(index, item.skills), *item.keywords]),
                prior=0.0,
                attrs={
                    "header": True,
                    "item_kind": section,
                    "line_terms": shown,
                    # Empty on purpose. A header's words cover requirements, but
                    # they are not *in-context* keywords: ``atscheck`` builds its
                    # bullets region by dropping entry lines and Technologies
                    # lines, so a term that only ever appears in a header cannot
                    # satisfy the "skills line *and* a bullet" test the context
                    # bonus exists to optimise. Claiming it here moved the proxy
                    # and not the metric — measured at −21 points of context.
                    PAGE_TERMS_ATTR: (),
                },
            )
        )
    return claims


def skill_claims(
    index: Index, requirements: Sequence[Requirement] | None = None
) -> list[Claim]:
    """One claim per skill, grouped by category.

    A skill costs nothing on its own — the line it sits on is the group's cost.
    ``familiar`` skills are held back unless a requirement names one of their
    surface forms: listing them unprompted dilutes the line without covering
    anything.
    """
    named: set[str] = set()
    for requirement in requirements or ():
        named.update(str(term).strip().lower() for term in requirement.terms)
        named.update(_forms_in_text(requirement.text))

    claims: list[Claim] = []
    for category in index.skills:
        for skill in category.skills:
            forms = skill.surface_forms()
            if skill.proficiency == "familiar" and not any(
                form.strip().lower() in named for form in forms
            ):
                continue
            claims.append(
                Claim(
                    id=f"skill:{skill.id}",
                    text=" ".join(forms),
                    group=category.category,
                    section="skill",
                    cost=0.0,
                    terms=_dedupe(forms),
                    prior=PROFICIENCY_PRIOR.get(skill.proficiency, 0.0),
                    attrs={"skill_id": skill.id, "name": skill.name,
                           "proficiency": skill.proficiency},
                )
            )
    return claims


def page_terms(text, skill_ids, keywords, skills) -> tuple[str, ...]:
    """Every form of every term ``text`` actually spells out.

    A term counts as shown when *any* of its surface forms is in the text, and
    then all of its forms are returned: the keyword check resolves a posting's
    wording to the whole alias set, so a bullet saying "Kubernetes" answers a
    posting asking for "K8s".
    """
    lowered = text.lower()
    found: list[str] = []
    for skill_id in dict.fromkeys(skill_ids):
        skill = skills.get(skill_id)
        if skill is None:
            continue
        forms = skill.surface_forms()
        if any(count_in_text(form, lowered, text) for form in forms):
            found.extend(forms)
    found.extend(kw for kw in keywords if count_in_text(kw, lowered, text))
    return _dedupe(found)


def _forms_in_text(text: str) -> set[str]:
    return {word.strip().lower() for word in str(text).split() if word.strip()}


def alias_map(index: Index) -> dict[str, str]:
    """Every surface form the index knows → the skill id it stands for."""
    mapping: dict[str, str] = {}
    for skill in index.all_skills():
        for form in skill.surface_forms():
            key = form.strip().lower()
            if key and (key not in mapping or key == skill.name.strip().lower()):
                mapping[key] = skill.id
    return mapping


def section_rules(settings=config) -> dict[str, SectionRule]:
    """The ``ATS_MIN/MAX_*`` settings as per-section rules for the solver."""
    return {
        "role": SectionRule(
            min_groups=settings.MIN_EXPERIENCE,
            max_groups=settings.MAX_EXPERIENCE,
            min_claims=settings.MIN_EXP_BULLETS,
            max_claims=settings.MAX_EXP_BULLETS,
            group_cost=ROLE_GROUP_COST,
        ),
        "project": SectionRule(
            min_groups=settings.MIN_PROJECTS,
            max_groups=settings.MAX_PROJECTS,
            min_claims=settings.MIN_PROJECT_BULLETS,
            max_claims=settings.MAX_PROJECT_BULLETS,
            group_cost=PROJECT_GROUP_COST,
            required_attr="impact",
        ),
        "skill": SectionRule(
            min_groups=1,
            max_groups=settings.MAX_SKILL_LINES,
            min_claims=1,
            max_claims=MAX_SKILLS_PER_LINE,
            group_cost=1,
            min_gain=settings.SKILL_MIN_GAIN,
        ),
    }


def pinned_skill_groups(settings=config) -> set[str]:
    """Skill categories that always get a line (``ATS_PIN_SKILL_GROUPS``).

    Pinning opens the line; it does not decide what goes on it. The skills
    listed still have to clear the section's ``min_gain``, and only if none do
    does the repair pass fall back to the best by proficiency prior — so a
    pinned category never becomes a place to park keywords.
    """
    return {
        name.strip()
        for name in str(settings.PIN_SKILL_GROUPS or "").split(",")
        if name.strip()
    }


# ── Selection → template context ─────────────────────────────────────────────
def render_context(
    selection,
    index: Index,
    claims: Sequence[Claim] | None = None,
    text_overrides: Mapping[str, str] | None = None,
    skill_labels: Mapping[str, str] | None = None,
):
    """``(experience, projects, skill_lines)`` in the shapes the template wants.

    Bullets keep the entry's own order (a selection ordered by gain would read
    as a jumble), entries are chronological, and skill lines are ordered by how
    much their category won — that ordering is the only place the solver's
    ranking survives into the page.

    ``text_overrides`` is the same map ``claims_from_index`` was built with, so
    the page prints the wording the solver actually chose on.

    ``skill_labels`` (skill id → label) is how the posting's own spelling
    reaches the page — see :mod:`ats_tailor.render.surface`. Absent, every
    skill prints under its index name, which is what v1 did.
    """
    legacy_skills, legacy_projects, legacy_exp, _certs, _summary = to_legacy(index)
    text_overrides = dict(text_overrides or {})
    by_id = {c.id: c for c in (claims or [])}
    chosen = set(selection.claim_ids)

    bullet_order = {
        bullet.id: position
        for _section, item in visible_items(index)
        for position, bullet in enumerate(item.bullets)
    }
    bullet_text = {
        bullet.id: str(text_overrides.get(bullet.id, bullet.text))
        for _section, item in visible_items(index)
        for bullet in item.bullets
    }

    picked: dict[str, list[str]] = {}
    for claim_id in selection.claim_ids:
        claim = by_id.get(claim_id)
        if claim is not None and claim.attrs.get("header"):
            continue  # the template renders the header from the item itself
        group = claim.group if claim else _group_of(claim_id, index)
        if group is None or claim_id not in bullet_text:
            continue
        picked.setdefault(group, []).append(claim_id)

    def bullets_for(item_id: str) -> list[str]:
        ids = sorted(picked.get(item_id, []), key=lambda cid: bullet_order.get(cid, 0))
        return [bullet_text[cid] for cid in ids]

    role_ctx = []
    for role in legacy_exp["roles"]:
        if role["id"] not in picked:
            continue
        context = build_role_context(role)
        context["bullets"] = bullets_for(role["id"])
        context["id"] = role["id"]
        role_ctx.append(context)

    project_ctx = []
    for entry in legacy_projects["projects"]:
        if entry["id"] not in picked:
            continue
        context = build_project_context(entry)
        context["bullets"] = bullets_for(entry["id"])
        context["id"] = entry["id"]
        project_ctx.append(context)

    role_ctx.sort(key=lambda c: parse_start_date(c["period"]), reverse=True)
    project_ctx.sort(key=lambda c: parse_start_date(c["period"]), reverse=True)

    skill_lines = _skill_lines(selection, index, by_id, chosen, legacy_skills,
                               skill_labels)
    return role_ctx, project_ctx, skill_lines


def _group_of(claim_id: str, index: Index) -> str | None:
    for _section, item in visible_items(index):
        if any(bullet.id == claim_id for bullet in item.bullets):
            return item.id
    return None


def _skill_lines(
    selection, index: Index, by_id, chosen, legacy_skills,
    skill_labels: Mapping[str, str] | None = None,
) -> list[dict]:
    """One rendered line per selected category, best-covering category first.

    Labels are deduped across the whole section, not just within a line: two
    categories can align onto the same posting word, and the same term printed
    twice is the stuffing ``ats-tailor check`` warns about.
    """
    from .render.surface import dedupe_labels

    labels = dict(skill_labels or {})
    gains = {step.claim_id: float(step.gain) for step in selection.trace}
    per_category: dict[str, list[tuple[float, str]]] = {}
    for claim_id in selection.claim_ids:
        claim = by_id.get(claim_id)
        if claim is None or claim.section != "skill":
            continue
        skill_id = str(claim.attrs.get("skill_id") or "")
        name = labels.get(skill_id) or str(claim.attrs.get("name") or claim.text)
        per_category.setdefault(claim.group, []).append((gains.get(claim_id, 0.0), name))

    ordered = sorted(
        per_category.items(),
        key=lambda item: (-sum(gain for gain, _ in item[1]), item[0]),
    )
    lines = []
    printed: set[str] = set()
    for category, entries in ordered:
        names = [name for _gain, name in sorted(entries, key=lambda e: (-e[0], e[1]))]
        names = dedupe_labels(names, printed)[:MAX_SKILLS_PER_LINE]
        if not names:
            continue
        lines.append({
            "category": CATEGORY_LABELS.get(category, category),
            "skills": ", ".join(names),
        })
    return lines


def certification_scores(index: Index) -> list[dict]:
    """The legacy certification dicts, for v1's ``select_certifications``."""
    _skills, _projects, _exp, certs, _summary = to_legacy(index)
    return certs["certifications"]

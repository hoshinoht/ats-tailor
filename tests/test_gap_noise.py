"""Title-Cased prose is not a technology, and a culture section is not an ask.

A posting written as headline-cased sentences ("We View Our Engineers As
Holistic Problem Solvers") and ``Label: sentence`` bullets used to read as
three dozen unknown technologies, each of which became a weighted requirement:
"Tell us about a time you took the lead" arrived as a must-have naming
``Tell``, while "Bachelor's degree in Computer Science" — which names something
real — sank to the soft weight. These tests pin the shape of that posting, not
the posting itself.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ats_tailor.ats.check import (
    classify_jd_sections,
    gap_terms,
    index_term_forms,
    profile_gap_terms,
)
from ats_tailor.engine.legacy.scoring import chunk_text
from ats_tailor.index.io import load_index_v2
from ats_tailor.jd.models import JDProfile, RequirementSpec
from ats_tailor.jd.requirements import requirements_from_jd
from ats_tailor.service import Service

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"

#: A posting shaped like the one that exposed the defect: Title-Cased prose, a
#: label per bullet, a sentence-cased culture heading, a quoted motto, and a
#: requirements section whose realest line names nothing the index knows.
NOISY_JD = """\
Nimbus Labs builds the identity rails behind the national payments network.

Our platform enables high assurance authentication and secure data sharing \
(Datavault) for millions of people. Working here means building systems where \
security and trust are the foundation of everything we do.

What you will be working on

As a Platform Engineer At Nimbus, You Won't Be Confined To a Single Layer Of \
The Stack. We View Our Engineers As Holistic Problem Solvers. You Will

 Solve End-to-End Problems: Participate in the full software development \
lifecycle, from design through to production support.
 Write Robust Code: Craft clean, maintainable, and secure-by-design code. \
While we currently use Java and TypeScript, we are standardizing on Go for new \
services.
 Learn Cloud Native Engineering: Manage and monitor cloud deployments \
(AWS/GCC) with a focus on operational safety.
 Collaborate Across Disciplines: Work closely with UX designers and fellow \
engineers.

What we are looking for

 Are Proactive Self-Starters: You identify areas for improvement and take the \
initiative to learn what you don't know.
 Are Versatile Problem Solvers: You are willing to dive into any part of the \
system to fix a bug or build a feature.
 Have a Security-First Mindset: You understand that this work requires extreme \
attention to detail.

What it is like working here

 High Impact: Your work affects the daily lives of millions of people.
 Polyglot Environment: You'll have the chance to work with Go, TypeScript, \
Java, and modern cloud technologies.
 Mission-Driven Culture: We are "Digital to the Core, and Serve with Heart." \
We care deeply about the public good.

Requirements

 Bachelor's degree in Computer Science, Information Systems, or a related field.
 Experience with at least one modern programming language (e.g., Go, Java, \
TypeScript, Python).
 A portfolio of projects that demonstrates your ability to build and ship \
software.
 Tell us about a time you took the lead on a project or taught yourself a \
complex new skill.
"""

KNOWN = ["Nimbus Labs", "Platform Engineer"]

#: English words the shape-only detector used to call technologies. Every one
#: is a label, a sentence opener, a quoted motto, or Title-Cased prose.
NOISE = {
    "Across", "Confined", "Core", "Craft", "Culture", "Disciplines",
    "Environment", "Heart", "Holistic", "Impact", "Layer", "Mindset",
    "Mission", "Native", "Polyglot", "Proactive", "Robust", "Security",
    "Self", "Serve", "Single", "Solve", "Solvers", "Starters", "Tell",
    "Versatile", "View", "Won",
}


@pytest.fixture(scope="module")
def index():
    return load_index_v2(EXAMPLES / "index")


@pytest.fixture(scope="module")
def gaps(index):
    return gap_terms(NOISY_JD, index, extra_known=KNOWN)


# ── 1. The gap set ────────────────────────────────────────────────────────────
def test_a_capital_is_only_evidence_where_it_was_a_choice(gaps):
    assert {"Datavault", "GCC", "Java", "TypeScript"} <= set(gaps)
    assert not NOISE & set(gaps)


def test_a_label_prefix_names_nothing(index):
    """"Have a Security-First Mindset:" is a heading the posting wrote in caps."""
    jd = "Have a Security-First Mindset: you review your own code.\n"
    assert not gap_terms(jd, index, extra_known=KNOWN)


def test_a_sentence_opener_is_not_a_product_but_a_listed_one_is(index):
    assert not gap_terms("Evidence of a proactive mindset.\n", index)
    # Same position, list shape: these are being enumerated, not narrated.
    assert "Terraform" in gap_terms("- Terraform, Ansible and Vault\n", index)
    assert "Terraform" in gap_terms("- Terraform (IaC) is used throughout\n", index)


def test_a_camelcase_name_keeps_its_capital_wherever_it_stands(index):
    """No amount of Title-Casing turns an English word into "OpenShift"."""
    assert "OpenShift" in gap_terms("OpenShift experience is required.\n", index)
    # The shape survives the positions that silence an ordinary capital.
    assert "OpenShift" in gap_terms(
        "As a Platform Engineer At Nimbus, You Will Run OpenShift.\n",
        index, extra_known=KNOWN)
    assert "PagerDuty" in gap_terms(
        "We run an on-call rotation. PagerDuty pages the duty engineer.\n", index)


def test_a_quoted_motto_is_not_a_stack(index):
    jd = 'We are "Digital to the Core, and Serve with Heart" here.\n'
    assert not gap_terms(jd, index, extra_known=KNOWN)


def test_a_word_the_posting_also_spells_lowercase_is_english(index):
    """"Robust" at the head of a bullet, "robust" in the next line: a word."""
    jd = "Robust Systems: we hold a high bar.\nWe write robust code.\n"
    assert "Robust" not in gap_terms(jd, index)


# ── 2. Sections ───────────────────────────────────────────────────────────────
def test_the_culture_section_is_ignored_not_required():
    chunks = chunk_text(NOISY_JD)
    tiers = dict(zip(chunks, classify_jd_sections(chunks)))
    culture = [text for text in tiers
               if text.startswith(("High Impact", "Polyglot Environment",
                                   "Mission-Driven Culture"))]
    assert len(culture) == 3
    assert {tiers[text] for text in culture} == {"ignore"}


def test_an_interrogative_heading_ends_the_section_above_it():
    """Whatever its casing: "What you will be working on" is a new section."""
    chunks = ["What we are looking for", "- Strong Kubernetes and Docker experience",
              "What you will be working on", "- Building services in Python"]
    assert classify_jd_sections(chunks) == ["must", "must", "body", "body"]


def test_a_question_word_without_a_reader_is_a_requirement_not_a_heading():
    """"How to design scalable systems" is what they want, not a new section."""
    chunks = ["Requirements:", "How to design scalable systems",
              "Strong experience with Kubernetes and Docker",
              "What you will be working on", "- Building services in Python"]
    assert classify_jd_sections(chunks) == ["must", "must", "must", "body", "body"]


def test_nothing_under_an_ignore_heading_becomes_a_requirement(index, gaps):
    requirements = requirements_from_jd(NOISY_JD, index_term_forms(index), gaps)
    assert not [r for r in requirements
                if r.text.startswith(("High Impact", "Polyglot Environment",
                                      "Mission-Driven Culture"))]


# ── 3. The requirements ───────────────────────────────────────────────────────
def test_no_requirement_carries_an_english_word_as_a_term(index, gaps):
    requirements = requirements_from_jd(NOISY_JD, index_term_forms(index), gaps)
    named = {term for r in requirements for term in r.terms}
    assert not NOISE & named


def test_a_prose_prompt_no_longer_outranks_a_qualification(index, gaps):
    requirements = requirements_from_jd(NOISY_JD, index_term_forms(index), gaps)
    by_start = {r.text.split(":")[0][:20]: r for r in requirements}
    tell = next(r for r in requirements if r.text.startswith("Tell us about"))
    degree = next(r for r in requirements if r.text.startswith("Bachelor"))
    portfolio = next(r for r in requirements if r.text.startswith("A portfolio"))

    # It is still a requirement — the posting does ask — but it names nothing,
    # so it takes the soft weight and stops bidding for page space.
    assert not tell.terms
    assert tell.weight == portfolio.weight < 1.0
    assert degree.weight >= tell.weight
    assert by_start  # the posting still parses into something


def test_the_posting_stops_reading_as_three_dozen_requirements(index, gaps):
    requirements = requirements_from_jd(NOISY_JD, index_term_forms(index), gaps)
    assert len(requirements) < 20
    languages = next(r for r in requirements
                     if r.text.startswith("Experience with at least one"))
    assert {"Go", "Java", "Python"} <= set(languages.terms)


# ── 4. Gaps scoped to the profile ─────────────────────────────────────────────
def test_profile_gap_terms_keeps_only_what_a_requirement_names():
    requirements = [RequirementSpec(id="req-1", text="Experience with Java",
                                    terms=["Java"])]
    assert profile_gap_terms(["Java", "Datavault"], requirements) == ["Java"]
    # A term the posting spells only in its own words still counts.
    surfaced = [RequirementSpec(id="req-1", text="Container orchestration",
                                terms=["Kubernetes"], surface=["K8s"])]
    assert profile_gap_terms(["K8s", "Datavault"], surfaced) == ["K8s"]
    assert profile_gap_terms(["Java"], []) == []


def test_profile_gap_terms_reads_a_stored_run_the_same_way():
    """``review_pack`` hands it plain dicts out of the run record."""
    rows = [{"id": "req-1", "text": "Experience with Java", "terms": ["Java"]}]
    assert profile_gap_terms(["Java", "GCC"], rows) == ["Java"]


def test_an_edited_profile_decides_which_gaps_are_reported(tmp_path):
    """The human who corrected the reading is not corrected back by a heuristic."""
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(EXAMPLES / "index", root / "index")
    shutil.copy(EXAMPLES / "profile.yaml", root / "profile.yaml")
    (root / "jd").mkdir()
    (root / "jd" / "noisy.txt").write_text(NOISY_JD)

    with Service(root) as service:
        before = service.fit("noisy")
        assert {"Datavault", "Java"} <= set(before.gaps)

        service.set_requirements("noisy", JDProfile(requirements=[
            RequirementSpec(id="req-1", kind="must",
                            text="Experience with Java and Go", terms=["Java"]),
        ]))
        after = service.fit("noisy")

    assert after.requirements == 1
    assert after.gaps == ["Java"]

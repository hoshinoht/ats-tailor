import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ats_tailor.index.loaders import (
    build_cache_key,
    load_cached_embeddings,
    parse_start_date,
    save_cached_embeddings,
    validate_index,
)
from ats_tailor.render.latex import build_project_context, derive_tagline, latex_escape


class CoreBehaviourTests(unittest.TestCase):
    def test_latex_escape_is_single_pass(self):
        self.assertEqual(latex_escape("\\"), r"\textbackslash{}")
        self.assertEqual(latex_escape("A&B_#{}~^%"), r"A\&B\_\#\{\}\textasciitilde{}\textasciicircum{}\%")

    def test_parse_start_date_accepts_full_month_names(self):
        self.assertEqual(parse_start_date("September 2024 – Present").month, 9)
        self.assertEqual(parse_start_date("Sept. 2024 – Dec 2024").month, 9)

    def test_cache_key_is_model_aware_and_path_safe(self):
        key_a = build_cache_key("role/../x", "org/model:v1", "a" * 64)
        key_b = build_cache_key("role/../x", "org/model:v2", "a" * 64)
        self.assertNotEqual(key_a, key_b)
        self.assertNotIn("/", key_a)
        self.assertNotIn("..", key_a)

    def test_cache_dimension_mismatch_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            key = build_cache_key("jd", "model", "b" * 64)
            save_cached_embeddings(cache_dir, key, np.zeros((2, 3)))
            self.assertIsNone(load_cached_embeddings(cache_dir, key, expected_dim=4))
            self.assertIsNone(load_cached_embeddings(cache_dir, key, expected_dim=3, expected_ndim=1))
            self.assertIsNone(load_cached_embeddings(cache_dir, key, expected_dim=3, expected_ndim=2, expected_rows=3))
            self.assertEqual(
                load_cached_embeddings(cache_dir, key, expected_dim=3, expected_ndim=2, expected_min_rows=2).shape,
                (2, 3),
            )

    def test_project_context_includes_impact_bullet(self):
        project = {
            "name": "Example",
            "period": "Jan 2025 – Present",
            "summary": "Example project with useful impact",
            "tech_stack": {"languages": ["Python"], "frameworks": [], "infrastructure": [], "patterns": [], "tools": []},
            "responsibilities": ["Built API", "Wrote tests", "Documented deployment"],
            "impact": ["Reduced latency by 40%"],
        }
        ctx = build_project_context(project)
        self.assertIn("Reduced latency by 40%", ctx["bullets"])

    def test_project_context_prefers_explicit_tagline(self):
        project = {
            "name": "Example",
            "period": "Jan 2025 - Present",
            "summary": "Semantic resume tailoring engine that matches a YAML index against postings",
            "tagline": "Resume tailoring engine",
            "tech_stack": {},
            "responsibilities": ["Built API"],
            "impact": [],
        }
        self.assertEqual(build_project_context(project)["tagline"], "Resume tailoring engine")

    def test_derive_tagline_cuts_at_clause_boundary(self):
        # A long summary without a "with/using/for" separator used to be chopped
        # mid-clause at 50 characters ("... engine that matches a").
        self.assertEqual(
            derive_tagline("Semantic resume tailoring engine that matches a YAML index against job descriptions"),
            "Semantic resume tailoring engine",
        )
        self.assertEqual(derive_tagline("Full-stack caregiver platform with event-driven backend"),
                         "Full-stack caregiver platform")
        self.assertEqual(derive_tagline("Real-time social platform, built for hawker centres"),
                         "Real-time social platform")
        # No separator at all: fall back to a word-boundary cut inside the window.
        long = "Distributed multi-region observability aggregation service dashboard thing"
        self.assertEqual(derive_tagline(long), "Distributed multi-region observability aggregation")
        self.assertEqual(derive_tagline("Short summary"), "Short summary")

    def test_validate_index_rejects_unknown_evidence_refs(self):
        skills = {
            "categories": [{
                "name": "Languages",
                "skills": [{
                    "name": "Python",
                    "ats_keywords": ["Python"],
                    "proficiency": "advanced",
                    "evidence": [{"ref": "missing", "type": "project"}],
                }],
            }]
        }
        projects = {"projects": []}
        exp = {"roles": [], "education": [{"institution": "U", "degree": "BSc", "field": "CS", "period": "2024 – 2027"}]}
        certs = {"certifications": []}
        with self.assertRaisesRegex(ValueError, "unknown evidence ref"):
            validate_index(skills, projects, exp, certs)


class LegacyEntryPointTests(unittest.TestCase):
    """``ats_tailor.tailor`` is a shim; the engine lives in ``engine/legacy``."""

    def test_the_shim_re_exports_main_from_where_the_code_went(self):
        import ats_tailor.engine.legacy.tailor as engine
        import ats_tailor.tailor as shim

        self.assertIs(shim.main, engine.main)
        self.assertIs(shim.run, engine.run)

    def test_python_dash_m_still_reaches_the_parser(self):
        result = subprocess.run(
            [sys.executable, "-m", "ats_tailor.tailor", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--company", result.stdout)


if __name__ == "__main__":
    unittest.main()

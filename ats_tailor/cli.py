"""``ats-tailor`` command line entry point.

Subcommands:

- ``tailor``   — the v1 generation pipeline (arguments passed through unchanged)
- ``validate`` — load a v1 or v2 index, print errors and warnings
- ``migrate``  — convert a v1 index directory into the v2 layout
- ``schema``   — emit JSON Schema files for editor validation
- ``check``    — compile a rendered resume and check that an ATS can read it
- ``labels``   — scaffold the per-JD ground truth files ``eval`` scores against
- ``eval``     — run the pipeline over every label and print the metric table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .index_io import SCHEMA_DIRNAME, write_index_schemas, write_schemas

USAGE = """usage: ats-tailor <command> [options]

commands:
  tailor    generate a tailored resume (see `ats-tailor tailor --help`)
  validate  validate an index directory (v1 or v2 layout)
  migrate   convert a v1 index directory to the v2 layout
  schema    write JSON Schema files for the v2 index
  check     ATS parse + keyword check on a rendered resume
  labels    manage per-JD ground truth (labels init ...)
  eval      score the pipeline against the labels
"""


def _cmd_tailor(argv: list[str]) -> int:
    from .tailor import run

    run(argv)
    return 0


def _cmd_validate(argv: list[str]) -> int:
    from .index_io import detect_layout, load_index_v2, to_legacy
    from .loaders import load_index, validate_index
    from .migrate import build_v2_index

    parser = argparse.ArgumentParser(prog="ats-tailor validate")
    parser.add_argument("index_dir", nargs="?", default="index",
                        help="index directory (default: index/)")
    args = parser.parse_args(argv)

    index_dir = Path(args.index_dir)
    layout = detect_layout(index_dir)
    print(f"{index_dir}: {layout} layout")

    try:
        if layout == "v2":
            index = load_index_v2(index_dir)
            validate_index(*to_legacy(index))
        else:
            load_index(index_dir)
            index, _report = build_v2_index(index_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"  {len(index.roles)} roles ({sum(1 for r in index.roles if r.hidden)} hidden), "
        f"{len(index.projects)} projects, {len(index.certifications)} certifications, "
        f"{len(index.all_skills())} skills in {len(index.skills)} categories"
    )

    warnings = index.warnings()
    if warnings:
        print(f"  {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print("  no warnings")
    return 0


def _cmd_migrate(argv: list[str]) -> int:
    from .migrate import migrate_v1_to_v2

    parser = argparse.ArgumentParser(prog="ats-tailor migrate")
    parser.add_argument("src", help="v1 index directory")
    parser.add_argument("dst", help="destination directory for the v2 layout")
    parser.add_argument("--force", action="store_true",
                        help="write into a non-empty destination")
    parser.add_argument("--profile", default=None,
                        help="profile.yaml to copy alongside the migrated index")
    args = parser.parse_args(argv)

    try:
        report = migrate_v1_to_v2(args.src, args.dst, args.profile, force=args.force)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report.render())
    return 0


def _cmd_schema(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ats-tailor schema")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--out", default=None,
                       help="output directory (default: schema/)")
    group.add_argument("--index", default=None,
                       help=f"index directory; writes into <dir>/{SCHEMA_DIRNAME}/")
    args = parser.parse_args(argv)

    if args.index:
        written = write_index_schemas(args.index)
    else:
        written = write_schemas(Path(args.out or "schema"))
    for path in written:
        print(f"wrote {path}")
    return 0


def _resolve_profile(explicit, index_dir):
    """``--profile`` if given, else ``profile.yaml`` beside the index directory."""
    return Path(explicit) if explicit else Path(index_dir).resolve().parent / "profile.yaml"


def _cmd_check(argv: list[str]) -> int:
    import json

    from .atscheck import (
        CompileError, auto_terms, compile_pdf, entries_from_tex, extract_text,
        index_term_forms, keyword_coverage, output_filename, page_count,
        parse_check, resolve_terms, split_regions, summarize, surface_form_set,
        write_reports,
    )
    from .loaders import load_profile
    from .tailor import load_index_for_terms

    parser = argparse.ArgumentParser(
        prog="ats-tailor check",
        description="Compile a rendered resume (if needed) and check ATS parseability "
                    "and keyword coverage against the job description.")
    parser.add_argument("output_dir", help="run directory holding resume.tex")
    parser.add_argument("--jd", default=None,
                        help="job description; without it only the parse checks run")
    parser.add_argument("--profile", default=None,
                        help="profile.yaml (default: next to the index directory)")
    parser.add_argument("--labels", default=None,
                        help="label file supplying must_terms/nice_terms "
                             "(default: derive them from the JD)")
    parser.add_argument("--index", default="index", help="index directory (default: index/)")
    parser.add_argument("--company", default=None,
                        help="company name for the output PDF filename")
    parser.add_argument("--tex", default="resume.tex", help="tex file name in the run directory")
    parser.add_argument("--recompile", action="store_true",
                        help="compile even when a PDF is already present")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    tex_path = out_dir / args.tex
    pdf_path = out_dir / f"{Path(args.tex).stem}.pdf"

    selection = {}
    selection_path = out_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    company = args.company or selection.get("company") or out_dir.resolve().name

    index_dir = Path(args.index)
    profile_path = _resolve_profile(args.profile, index_dir)
    try:
        profile = load_profile(profile_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    pages = None
    if args.recompile or not pdf_path.exists():
        if not tex_path.exists():
            print(f"error: {tex_path} not found", file=sys.stderr)
            return 1
        print(f"Compiling {tex_path}...")
        try:
            compiled = compile_pdf(tex_path, output_filename(profile.get("name", ""), company))
        except CompileError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        pdf_path, pages = compiled.pdf_path, compiled.pages
    else:
        pages = page_count(pdf_path)

    text = extract_text(pdf_path)
    entries = selection.get("entries") or (
        entries_from_tex(tex_path.read_text()) if tex_path.exists() else [])
    parse_report = parse_check(text, profile, entries, pages)

    coverage = None
    jd_path = Path(args.jd) if args.jd else None
    if jd_path or args.labels:
        index_obj = load_index_for_terms(index_dir)
        all_terms = index_term_forms(index_obj)
        must, nice = [], []
        if args.labels:
            from .labels import Label
            import yaml as _yaml

            label = Label.model_validate(_yaml.safe_load(Path(args.labels).read_text()))
            must, nice = list(label.must_terms), list(label.nice_terms)
            if jd_path is None and label.jd:
                jd_path = Path(label.jd)
        if not must and not nice:
            if jd_path is None or not jd_path.is_file():
                print("error: --jd is required unless the label file has term lists",
                      file=sys.stderr)
                return 1
            must, nice = auto_terms(jd_path.read_text(), index_obj)
        terms = resolve_terms([*must, *nice], all_terms)
        coverage = keyword_coverage(
            text, terms, surface_form_set(all_terms), split_regions(text), must=must)

    write_reports(out_dir, parse_report, coverage)
    print(summarize(parse_report, coverage))
    print(f"reports: {out_dir / 'ats_check.md'}, {out_dir / 'ats_check.json'}")
    return 0 if parse_report.ok else 1


def _cmd_labels(argv: list[str]) -> int:
    from .labels import init_label, load_labels
    from .tailor import load_index_for_terms

    parser = argparse.ArgumentParser(prog="ats-tailor labels")
    sub = parser.add_subparsers(dest="action", required=True)

    init = sub.add_parser("init", help="scaffold a label file for one job description")
    init.add_argument("jd", help="job description text file")
    init.add_argument("--company", required=True)
    init.add_argument("--role", required=True)
    init.add_argument("--labels-dir", default="labels", help="default: labels/")
    init.add_argument("--index", default="index", help="index directory (default: index/)")
    init.add_argument("--jd-ref", default=None,
                      help="path to record in the file (default: the --jd path as given)")
    init.add_argument("--force", action="store_true", help="overwrite an existing label file")

    listing = sub.add_parser("list", help="list the label files and their pins")
    listing.add_argument("--labels-dir", default="labels")

    args = parser.parse_args(argv)

    if args.action == "list":
        try:
            labels = load_labels(args.labels_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for slug, label in labels.items():
            print(f"{slug:<24} {label.company}/{label.role}  "
                  f"pins: {len(label.pin_projects) + len(label.pin_roles)}, "
                  f"excl: {len(label.exclude)}, "
                  f"must: {len(label.must_terms)}, nice: {len(label.nice_terms)}")
        return 0

    try:
        index = load_index_for_terms(args.index)
        path = init_label(args.jd, args.company, args.role, args.labels_dir,
                          index=index, jd_ref=args.jd_ref, force=args.force)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {path}")
    return 0


def _cmd_eval(argv: list[str]) -> int:
    from .evaluate import compare, evaluate, render_table

    parser = argparse.ArgumentParser(
        prog="ats-tailor eval",
        description="Run the pipeline over every label file and score the results.")
    parser.add_argument("--index", default="index", help="index directory (default: index/)")
    parser.add_argument("--labels-dir", default="labels", help="default: labels/")
    parser.add_argument("--jd-dir", default="jd",
                        help="where to look for a label's JD file (default: jd/)")
    parser.add_argument("--out", default="eval", help="output directory (default: eval/)")
    parser.add_argument("--profile", default=None,
                        help="profile.yaml (default: next to the index directory)")
    parser.add_argument("--llm", action="store_true", help="enable LLM keyword expansion")
    parser.add_argument("--rerank", action="store_true", help="enable cross-encoder reranking")
    parser.add_argument("--only", nargs="+", default=None, help="evaluate only these slugs")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), default=None,
                        help="print the per-metric delta between two eval results")
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(*args.compare))
        return 0

    try:
        payload = evaluate(
            args.index, args.labels_dir, args.jd_dir, args.out,
            profile_path=args.profile, llm=args.llm, rerank=args.rerank, only=args.only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(render_table(payload))
    print(f"\nwrote {payload['path']}  (config {payload['config_hash'][:12]})")
    return 0


COMMANDS = {
    "tailor": _cmd_tailor,
    "validate": _cmd_validate,
    "migrate": _cmd_migrate,
    "schema": _cmd_schema,
    "check": _cmd_check,
    "labels": _cmd_labels,
    "eval": _cmd_eval,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(USAGE)
        return 0 if argv else 1
    command = argv[0]
    if command not in COMMANDS:
        print(f"error: unknown command '{command}'\n\n{USAGE}", file=sys.stderr)
        return 2
    return COMMANDS[command](argv[1:])


if __name__ == "__main__":
    sys.exit(main())

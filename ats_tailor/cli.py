"""``ats-tailor`` command line entry point.

Subcommands:

- ``tailor``   — the v1 generation pipeline (arguments passed through unchanged)
- ``validate`` — load a v1 or v2 index, print errors and warnings
- ``migrate``  — convert a v1 index directory into the v2 layout
- ``schema``   — emit JSON Schema files for editor validation
- ``check``    — compile a rendered resume and check that an ATS can read it
- ``labels``   — scaffold the per-JD ground truth files ``eval`` scores against
- ``eval``     — run the pipeline over every label and print the metric table
- ``sync``     — bring the runtime store in line with the index
- ``override`` — edit a run's ``overrides.yaml``
- ``runs``     — the application tracker
- ``jd``       — read a posting into gates and requirements, and edit them
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .index_io import SCHEMA_DIRNAME, write_index_schemas, write_schemas

USAGE = """usage: ats-tailor <command> [options]

commands:
  tailor    generate a tailored resume (see `ats-tailor tailor --help`)
  validate  validate an index directory (v1 or v2 layout)
  migrate   convert a v1 index directory to the v2 layout
  schema    write JSON Schema files for the v2 index
  check     ATS parse + keyword check on a rendered resume
  labels    manage per-JD ground truth (labels init | list | from-run)
  eval      score the pipeline against the labels
  sync      update the runtime store from the index (--embed to pre-embed)
  override  edit the per-run overrides.yaml next to a run's output
  runs      application tracker (runs list | show | status | note | quiet)
  jd        inspect a posting's gates and requirements (jd show)
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


def _open_store(index_dir, data_dir=None):
    """The store for this invocation: ``--data-dir`` wins, else the index's."""
    from .store import Store, store_path

    if data_dir:
        return Store(Path(data_dir) / "ats.db")
    return Store(store_path(index_dir))


def _cmd_sync(argv: list[str]) -> int:
    from .sync import sync

    parser = argparse.ArgumentParser(
        prog="ats-tailor sync",
        description="Upsert every index entity into the runtime store, prune what "
                    "the index no longer has, and optionally pre-compute embeddings.")
    parser.add_argument("--index", default="index", help="index directory (default: index/)")
    parser.add_argument("--data-dir", default=None,
                        help="store directory (default: ATS_DATA_DIR, else <index>/../data/)")
    parser.add_argument("--embed", action="store_true",
                        help="pre-compute claim, skill and certification embeddings")
    parser.add_argument("--model", default=None,
                        help="embedding model to pre-compute for (default: ATS_EMBED_MODEL)")
    args = parser.parse_args(argv)

    try:
        store = _open_store(args.index, args.data_dir)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        report = sync(args.index, embed=args.embed, model=args.model, store=store)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()
    print(f"store: {store.path}")
    print(report.render())
    return 0


def _cmd_override(argv: list[str]) -> int:
    from . import overrides as overrides_mod

    parser = argparse.ArgumentParser(
        prog="ats-tailor override",
        description="Edit (or create) the overrides.yaml a v2 run reads from its "
                    "output directory. Re-run `tailor --engine v2 --output <dir>` "
                    "to apply it.")
    parser.add_argument("out_dir", help="run output directory")
    parser.add_argument("--pin", action="append", default=[], metavar="ID",
                        help="role/project id that must be shown")
    parser.add_argument("--exclude", action="append", default=[], metavar="ID",
                        help="role/project id that must never be shown")
    parser.add_argument("--pin-bullet", action="append", default=[], metavar="ID",
                        help="bullet id that must be shown (pins its item too)")
    parser.add_argument("--exclude-bullet", action="append", default=[], metavar="ID")
    parser.add_argument("--text", action="append", default=[], metavar="ID=TEXT",
                        help="replacement phrasing for one bullet, this run only")
    parser.add_argument("--summary", default=None,
                        help="replacement summary text ('' drops the section)")
    parser.add_argument("--budget-lines", type=int, default=None,
                        help="claim-line budget, replacing the computed one")
    parser.add_argument("--show", action="store_true", help="print the file and exit")
    args = parser.parse_args(argv)

    path = overrides_mod.overrides_path(args.out_dir)
    if args.show:
        if not path.is_file():
            print(f"no overrides file at {path}")
            return 1
        print(path.read_text().rstrip())
        return 0

    edits: dict[str, str] = {}
    for item in args.text:
        bullet_id, sep, replacement = item.partition("=")
        if not sep or not bullet_id.strip():
            print(f"error: --text expects ID=TEXT, got {item!r}", file=sys.stderr)
            return 1
        edits[bullet_id.strip()] = replacement

    try:
        current, changes = overrides_mod.edit(
            path, pin=args.pin, exclude=args.exclude, pin_bullet=args.pin_bullet,
            exclude_bullet=args.exclude_bullet, text=edits, summary=args.summary,
            budget_lines=args.budget_lines,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {path}")
    for change in changes or ["(no change)"]:
        print(f"  {change}")
    print(f"  now: {current.summary_line()}")
    return 0


def _cmd_runs(argv: list[str]) -> int:
    from .store import STATUSES, StatusError
    from .tracker import QUIET_DAYS, render_list, render_quiet, render_show, quiet_runs

    parser = argparse.ArgumentParser(prog="ats-tailor runs")
    parser.add_argument("--index", default="index", help="index directory (default: index/)")
    parser.add_argument("--data-dir", default=None, help="store directory")
    sub = parser.add_subparsers(dest="action", required=True)

    listing = sub.add_parser("list", help="every recorded run, newest first")
    listing.add_argument("--status", default=None, choices=list(STATUSES))
    listing.add_argument("--company", default=None)

    show = sub.add_parser("show", help="one run in full")
    show.add_argument("run_id")

    status = sub.add_parser("status", help="move a run's status")
    status.add_argument("run_id")
    status.add_argument("status", choices=list(STATUSES))
    status.add_argument("--note", default=None, help="why, in the user's words")

    note = sub.add_parser("note", help="add a dated note to a run")
    note.add_argument("run_id")
    note.add_argument("text")

    quiet = sub.add_parser("quiet", help="open applications nobody has heard from")
    quiet.add_argument("--days", type=int, default=QUIET_DAYS)

    args = parser.parse_args(argv)
    try:
        store = _open_store(args.index, args.data_dir)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        if args.action == "list":
            print(render_list(store.list_runs(status=args.status, company=args.company)))
            return 0
        if args.action == "quiet":
            print(render_quiet(quiet_runs(store, args.days), args.days))
            return 0

        run = store.get_run(args.run_id)
        if run is None:
            print(f"error: no such run: {args.run_id}", file=sys.stderr)
            return 1
        if args.action == "show":
            print(render_show(run, store.notes(run.id)))
            return 0
        if args.action == "note":
            added = store.add_note(run.id, args.text)
            print(f"{run.id}: noted at {added.created_at}")
            return 0
        if args.action == "status":
            try:
                updated = store.set_status(run.id, args.status, args.note)
            except StatusError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            print(f"{run.id}: {run.status} -> {updated.status}")
            if updated.data.get("applied_at"):
                print(f"  applied_at {updated.data['applied_at']}")
            return 0
    finally:
        store.close()
    return 2


def _cmd_jd(argv: list[str]) -> int:
    """``jd show``: what the pipeline will read out of a posting, before it runs.

    The point is the *order*: gates first, then the requirements and the terms
    they carry. Everything printed here is what the next ``tailor`` of the same
    posting will use — same provider, same cache key — so reviewing it is
    reviewing the run, not a preview of one.
    """
    import json
    import os
    import subprocess
    import tempfile

    from .jd.extract import (
        FALLBACK_PROVIDER, extract_requirements, save_edited, term_alias_map,
    )
    from .jd.hashing import jd_hash
    from .jd.models import JDProfile
    from .llm.provider import get_provider
    from .loaders import load_profile
    from .tailor import load_index_for_terms

    parser = argparse.ArgumentParser(
        prog="ats-tailor jd",
        description="Read a job description into gates and weighted requirements.")
    sub = parser.add_subparsers(dest="action", required=True)
    show = sub.add_parser("show", help="print a posting's profile")
    show.add_argument("jd", help="job description text file")
    show.add_argument("--index", default="index", help="index directory (default: index/)")
    show.add_argument("--profile", default=None, help="profile.yaml")
    show.add_argument("--data-dir", default=None, help="store directory")
    show.add_argument("--provider", default=None, choices=list(config.PROVIDERS),
                      help=f"who reads the posting (default: {config.LLM_PROVIDER})")
    show.add_argument("--reextract", action="store_true",
                      help="ignore any cached extraction and read it again")
    show.add_argument("--company", default=None,
                      help="company name, so the gap list matches the run's")
    show.add_argument("--role", default=None,
                      help="role name, so the gap list matches the run's")
    show.add_argument("--json", action="store_true", help="print the profile as JSON")
    show.add_argument("--edit", action="store_true",
                      help="open the profile in $EDITOR and save it as the edited one")
    args = parser.parse_args(argv)

    jd_path = Path(args.jd)
    if not jd_path.is_file():
        print(f"error: no such job description: {jd_path}", file=sys.stderr)
        return 1
    jd_text = jd_path.read_text().strip()
    if not jd_text:
        print(f"error: {jd_path} is empty", file=sys.stderr)
        return 1

    index_dir = Path(args.index)
    try:
        index = load_index_for_terms(index_dir)
        candidate = load_profile(_resolve_profile(args.profile, index_dir))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    provider = get_provider(args.provider, alias_map=term_alias_map(index))
    store = _open_store(index_dir, args.data_dir)
    try:
        profile = extract_requirements(
            jd_text, index, provider, store, jd_path=str(jd_path),
            reextract=args.reextract,
            # The company and role are `tailor`'s, not the posting's: they keep
            # a company's own name out of the gap list. Given here, this prints
            # exactly what that run will use; left out, one or two of the
            # posting's proper nouns show up as requirement terms.
            extra_known=[value for value in (args.company, args.role) if value],
            languages=candidate.get("languages", []) or [],
        )
        if args.json:
            print(json.dumps(profile.to_json(), indent=2))
        else:
            print(_render_jd_profile(profile, jd_path))
        if not args.edit:
            return 0

        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            print("error: set $EDITOR to edit a profile", file=sys.stderr)
            return 1
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
            handle.write(json.dumps(profile.to_json(), indent=2) + "\n")
            temp = Path(handle.name)
        try:
            subprocess.run([*editor.split(), str(temp)], check=False)
            edited = JDProfile.from_json(json.loads(temp.read_text()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        finally:
            temp.unlink(missing_ok=True)
        if edited is None:
            print("error: the edited file is not a valid profile; nothing saved",
                  file=sys.stderr)
            return 1
        save_edited(store, jd_text,
                    edited.model_copy(update={
                        "provider": profile.provider or FALLBACK_PROVIDER}),
                    jd_path=str(jd_path))
        print(f"saved the edited profile for {jd_hash(jd_text)[:12]} "
              f"(provider {profile.provider}); it now outranks any extraction")
        return 0
    finally:
        store.close()


def _render_jd_profile(profile, jd_path) -> str:
    """Gates first, then the requirements, then what the index cannot cover."""
    lines = [
        f"{jd_path}",
        f"  read by     {profile.provider}"
        + (f" (prompt {profile.prompt_version})" if profile.prompt_version else "")
        + ("  [cached]" if profile.cached else ""),
        f"  title       {profile.title or '-'}",
        f"  archetype   {profile.archetype or '-'}",
        "",
        "gates",
    ]
    lines += [f"  {line}" for line in profile.gates.render().splitlines()]
    lines += ["", f"requirements ({profile.summary_line()})", ""]
    header = f"  {'id':<14} {'kind':<14} {'w':>4}  {'terms':<34} text"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for spec in profile.requirements:
        terms = ", ".join(spec.terms)
        text = spec.text if len(spec.text) <= 70 else spec.text[:69] + "…"
        lines.append(
            f"  {spec.id:<14} {spec.kind:<14} {spec.weight:>4.2f}  "
            f"{(terms[:33] + '…') if len(terms) > 34 else terms:<34} {text}"
        )
    return "\n".join(lines)


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

    from_run = sub.add_parser(
        "from-run", help="merge a run's pins and exclusions into its label file")
    from_run.add_argument("run", help="run output directory or run id")
    from_run.add_argument("--labels-dir", default="labels", help="default: labels/")
    from_run.add_argument("--index", default="index", help="index directory (default: index/)")
    from_run.add_argument("--data-dir", default=None, help="store directory")

    args = parser.parse_args(argv)

    if args.action == "from-run":
        return _labels_from_run(args)

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


def _labels_from_run(args) -> int:
    """``labels from-run``: promote one run's overrides into its label file."""
    from .labels import merge_overrides, slug
    from .tailor import load_index_for_terms

    try:
        store = _open_store(args.index, args.data_dir)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        run = _resolve_run(store, args.run)
    finally:
        store.close()
    if run is None:
        print(f"error: no recorded run for {args.run!r}", file=sys.stderr)
        return 1

    overrides = run.data.get("overrides") or {}
    pins = list(overrides.get("pin_items") or [])
    excludes = list(overrides.get("exclude_items") or [])
    if not pins and not excludes:
        print(f"{run.id}: no pins or exclusions to promote")
        return 0

    jd_path = run.data.get("jd_path")
    if not jd_path:
        print(f"error: {run.id} has no recorded JD path, so its label file is unknown",
              file=sys.stderr)
        return 1

    try:
        index = load_index_for_terms(args.index)
        path, changes = merge_overrides(
            args.labels_dir, slug(jd_path), pins, excludes, index.item_kinds())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not changes:
        print(f"{path}: already up to date with {run.id}")
        return 0
    print(f"updated {path} from {run.id}")
    for change in changes:
        print(f"  {change}")
    return 0


def _resolve_run(store, reference: str):
    """A run id, or the newest run recorded for an output directory."""
    run = store.get_run(reference)
    if run is not None:
        return run
    path = Path(reference)
    if not path.exists():
        return None
    wanted = path.resolve()
    for candidate in store.list_runs():
        if candidate.out_dir and Path(candidate.out_dir).resolve() == wanted:
            return candidate
    return None


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
    parser.add_argument("--engine", choices=["v1", "v2"], default=config.ENGINE,
                        help=f"selection engine to evaluate (env: ATS_ENGINE, "
                             f"default: {config.ENGINE})")
    parser.add_argument("--provider", default="none", choices=list(config.PROVIDERS),
                        help="who reads the postings in the v2 engine (default: none). "
                             "Recorded in the config fingerprint; profiles are cached "
                             "per (jd, provider, prompt version), so a model sweep is "
                             "reproducible once it has run")
    parser.add_argument("--reextract", action="store_true",
                        help="ignore cached extractions and read every posting again")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"), default=None,
                        help="print the per-metric delta between two eval results")
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(*args.compare))
        return 0

    try:
        payload = evaluate(
            args.index, args.labels_dir, args.jd_dir, args.out,
            profile_path=args.profile, llm=args.llm, rerank=args.rerank,
            only=args.only, engine=args.engine, provider=args.provider,
            reextract=args.reextract,
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
    "sync": _cmd_sync,
    "override": _cmd_override,
    "runs": _cmd_runs,
    "validate": _cmd_validate,
    "migrate": _cmd_migrate,
    "schema": _cmd_schema,
    "check": _cmd_check,
    "labels": _cmd_labels,
    "eval": _cmd_eval,
    "jd": _cmd_jd,
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

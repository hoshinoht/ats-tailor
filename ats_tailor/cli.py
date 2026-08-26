"""``ats-tailor`` command line entry point.

Subcommands:

- ``tailor``   — the v1 generation pipeline (arguments passed through unchanged)
- ``validate`` — load a v1 or v2 index, print errors and warnings
- ``migrate``  — convert a v1 index directory into the v2 layout
- ``schema``   — emit JSON Schema files for editor validation
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
"""


def _cmd_tailor(argv: list[str]) -> int:
    from .tailor import main as tailor_main

    saved = sys.argv
    sys.argv = ["ats-tailor tailor", *argv]
    try:
        tailor_main()
    finally:
        sys.argv = saved
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


COMMANDS = {
    "tailor": _cmd_tailor,
    "validate": _cmd_validate,
    "migrate": _cmd_migrate,
    "schema": _cmd_schema,
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

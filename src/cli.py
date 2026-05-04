"""Command-line interface.

Subcommands:
    process    Extract + index any new PDFs.
    search     Run a hybrid search and print ranked results.
    gui        Launch the legacy Tkinter UI (to be retired in Stage 3).
    reindex    Wipe and rebuild the index (use after changing chunking
               or extraction; see docs/changes/0002 and 0003).
    info       Print resolved configuration.

Configuration comes from environment variables (`HR_*`) or a `.env` file.
See `.env.example`.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import settings


def _print(msg: str) -> None:
    print(msg, flush=True)


def cmd_process(_args: argparse.Namespace) -> int:
    from src.main import process_pdfs_headless

    summary = process_pdfs_headless(
        str(settings.pdf_directory), str(settings.persist_directory), log=_print
    )
    _print(json.dumps(summary, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from src.main import run_search_headless

    try:
        response = run_search_headless(
            args.query, str(settings.persist_directory), top_k=args.top_k
        )
    except FileNotFoundError as e:
        _print(str(e))
        return 1

    if args.json:
        # Pydantic model_dump_json gives us a stable, typed contract — same
        # shape the FastAPI handler will return in Stage 3.
        _print(response.model_dump_json(indent=2))
        return 0

    _print(f"{len(response.hits)} hits in {response.took_ms:.1f} ms")
    _print("")
    for i, hit in enumerate(response.hits, 1):
        section = f" [{hit.section}]" if hit.section else ""
        _print(f"[{i}] {hit.source}  p.{hit.page}{section}  (score={hit.score:.5f})")
        _print(f"    {hit.snippet}")
        _print("")
    return 0


def cmd_gui(_args: argparse.Namespace) -> int:
    from src.main import launch_gui

    launch_gui()
    return 0


def cmd_reindex(_args: argparse.Namespace) -> int:
    from src.main import reindex_headless

    summary = reindex_headless(
        str(settings.persist_directory), str(settings.pdf_directory), log=_print
    )
    _print(json.dumps(summary, indent=2))
    return 0


def cmd_info(_args: argparse.Namespace) -> int:
    info = {
        "pdf_directory": str(settings.pdf_directory),
        "persist_directory": str(settings.persist_directory),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "candidate_k": settings.candidate_k,
    }
    _print(json.dumps(info, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybrid-retrieval", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("process", help="Index any new PDFs.").set_defaults(func=cmd_process)

    s = sub.add_parser("search", help="Run a hybrid search.")
    s.add_argument("query", help="Search query.")
    s.add_argument("--top-k", type=int, default=None)
    s.add_argument("--json", action="store_true", help="Emit JSON instead of human output.")
    s.set_defaults(func=cmd_search)

    sub.add_parser("gui", help="Launch the Tkinter UI (legacy).").set_defaults(func=cmd_gui)
    sub.add_parser("reindex", help="Wipe and rebuild the index.").set_defaults(func=cmd_reindex)
    sub.add_parser("info", help="Print resolved configuration.").set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

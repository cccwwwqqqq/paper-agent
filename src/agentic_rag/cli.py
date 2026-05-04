from __future__ import annotations

import argparse
import subprocess
import sys

from agentic_rag.settings import get_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic RAG command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Start the Gradio application.")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF or Markdown files into a workspace.")
    ingest_parser.add_argument("--workspace", default=None, help="Workspace name.")
    ingest_parser.add_argument("--files", nargs="+", required=True, help="Files to ingest.")

    refine_parser = subparsers.add_parser("refine-formulas", help="Refine formula-heavy pages after fast ingestion.")
    refine_parser.add_argument("--workspace", default=None, help="Workspace name.")
    refine_parser.add_argument("--paper-id", default=None, help="Only refine one paper id.")
    refine_parser.add_argument("--max-pages", type=int, default=None, help="Maximum pages to refine per paper.")

    eval_parser = subparsers.add_parser("eval", help="Run the evaluation suite.")
    eval_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to eval/evaluate.py")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, forwarded_args = parser.parse_known_args(argv)

    if forwarded_args and args.command != "eval":
        parser.error(f"unrecognized arguments: {' '.join(forwarded_args)}")

    if args.command == "serve":
        from agentic_rag.app import main as serve_main

        serve_main()
        return 0

    if args.command == "ingest":
        from agentic_rag.bootstrap import build_runtime

        runtime = build_runtime(get_settings())
        workspace_id = args.workspace or runtime.settings.default_workspace_id
        added, skipped = runtime.ingestion_service.add_documents(args.files, workspace_id=workspace_id)
        print(f"Added: {added} | Skipped: {skipped}")
        return 0

    if args.command == "refine-formulas":
        from agentic_rag.bootstrap import build_runtime

        runtime = build_runtime(get_settings())
        workspace_id = args.workspace or runtime.settings.default_workspace_id
        report = runtime.formula_refinement_service.refine_workspace(
            workspace_id,
            paper_id=args.paper_id,
            max_pages=args.max_pages,
        )
        print(f"Formula refinement candidates: {report['candidate_count']}")
        for item in report["reports"]:
            print(
                f"- {item['paper_id']}: pages {item['pages_refined']} | "
                f"merged pages {item.get('merged_pages', [])} | "
                f"remaining markers after merge: {item.get('formula_markers_after_merge', item['formula_markers_in_refined_output'])}"
            )
        return 0

    if args.command == "eval":
        settings = get_settings()
        command = [sys.executable, "-m", "eval.evaluate", *args.args, *forwarded_args]
        return subprocess.call(command, cwd=settings.root_dir)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

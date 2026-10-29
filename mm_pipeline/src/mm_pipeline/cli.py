from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import apply_overrides, load_project_config, resolve_window
from .exceptions import PipelineError
from .pipeline import run_pipeline
from .registry import ADAPTERS


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    return logging.getLogger("mm_pipeline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monitor-pipeline",
        description="Run a configuration-driven media monitoring pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Collect and prepare a master dataset")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--start", help="Inclusive ISO-8601 UTC start")
    run.add_argument("--end", help="Exclusive ISO-8601 UTC end")
    run.add_argument("--lookback-days", type=int)
    run.add_argument("--source", action="append", dest="sources")
    run.add_argument("--output-uri")
    run.add_argument("--env-file", type=Path)
    run.add_argument("--log-level", default="INFO")

    validate = subparsers.add_parser("validate-config", help="Validate project YAML")
    validate.add_argument("--config", required=True, type=Path)

    subparsers.add_parser("list-adapters", help="List installed source adapters")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-adapters":
        print("\n".join(sorted(ADAPTERS)))
        return 0

    try:
        config = load_project_config(args.config)
        if args.command == "validate-config":
            print(
                json.dumps(
                    {
                        "project": config.project.id,
                        "enabled_sources": [source.id for source in config.enabled_sources],
                        "status": "valid",
                    },
                    indent=2,
                )
            )
            return 0

        if args.env_file:
            load_dotenv(args.env_file, override=False)
        config = apply_overrides(
            config,
            selected_sources=args.sources,
            output_uri=args.output_uri,
        )
        window = resolve_window(
            config,
            start=args.start,
            end=args.end,
            lookback_days=args.lookback_days,
        )
        logger = configure_logging(args.log_level)
        result = run_pipeline(config, window=window, logger=logger)
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "status": "succeeded" if result.succeeded else "failed",
                    "records": len(result.articles),
                    "failures": [failure.__dict__ for failure in result.failures],
                    "outputs": result.output_locations,
                    "fatal_errors": result.fatal_errors,
                },
                indent=2,
                default=str,
            )
        )
        return 0 if result.succeeded else 1
    except (PipelineError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

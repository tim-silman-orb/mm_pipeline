from __future__ import annotations

import argparse
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .collection import RunWindow
from .config import PipelineConfig, load_config
from .stages import ClusteringStage, CollectionStage, ExtractionStage, FullPipeline


def default_config_path() -> Path:
    configured = os.getenv("MM_PIPELINE_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    candidates = [
        Path(__file__).resolve().parents[2] / "config" / "ebola.yml",
        Path.cwd() / "config" / "ebola.yml",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm-pipeline",
        description="Run the existing media-monitoring backend without notebooks.",
    )
    parser.add_argument("--config", default=str(default_config_path()), help="Project YAML file")
    parser.add_argument("--env-file", help="Optional dotenv file; defaults to <project_root>/.env")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate-config", help="Load and validate configuration only")

    collect = commands.add_parser("collect", help="Collect and prepare master_data.jsonl")
    add_window_arguments(collect)
    add_collection_arguments(collect)

    extract = commands.add_parser("extract", help="Run Bedrock development extraction")
    extract.add_argument("--limit", type=int, help="Temporarily process only the first N articles")

    cluster = commands.add_parser("cluster", help="Run per-stream embedding and story clustering")
    cluster.add_argument("--stream", action="append", help="Run one configured stream; repeat as needed")

    full = commands.add_parser("run", help="Run collection, extraction and clustering in one task")
    add_window_arguments(full)
    add_collection_arguments(full)
    full.add_argument("--limit", type=int, help="Temporarily extract only the first N articles")
    full.add_argument("--stream", action="append", help="Run one configured stream; repeat as needed")
    return parser


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--lookback-days", type=int, default=7, help="Inclusive rolling window (default: 7)")
    group.add_argument("--start-date", help="Start date in YYYY-MM-DD format; requires --end-date")
    parser.add_argument("--end-date", help="End date in YYYY-MM-DD format")


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", action="append", help="Run one source; repeat as needed")
    parser.add_argument(
        "--topic",
        action="append",
        help="Replace configured topic terms for this run; repeat as needed",
    )


def resolve_window(args: argparse.Namespace) -> RunWindow:
    if args.start_date:
        if not args.end_date:
            raise ValueError("--start-date requires --end-date")
        return RunWindow(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date))
    if args.end_date:
        raise ValueError("--end-date requires --start-date")
    days = int(args.lookback_days)
    if days < 1:
        raise ValueError("--lookback-days must be at least 1")
    end = date.today()
    return RunWindow(end - timedelta(days=days - 1), end)


def load_environment(args: argparse.Namespace, config: PipelineConfig) -> None:
    env_file = Path(args.env_file).expanduser().resolve() if args.env_file else config.project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("mm_pipeline_2")
    try:
        config = load_config(args.config)
        if args.command == "validate-config":
            print(f"Configuration valid: {config.config_path}")
            print(f"Project root: {config.project_root}")
            print(f"Artifact URI: {config.artifact_uri}")
            return 0
        load_environment(args, config)

        if args.command == "collect":
            CollectionStage(config, logger=logger.getChild("collect")).run(
                window=resolve_window(args),
                selected_sources=set(args.source) if args.source else None,
                topic_terms=args.topic,
            )
        elif args.command == "extract":
            ExtractionStage(config, logger=logger.getChild("extract")).run(limit=args.limit)
        elif args.command == "cluster":
            ClusteringStage(config, logger=logger.getChild("cluster")).run(
                selected_streams=set(args.stream) if args.stream else None,
            )
        elif args.command == "run":
            FullPipeline(config, logger=logger).run(
                window=resolve_window(args),
                selected_sources=set(args.source) if args.source else None,
                topic_terms=args.topic,
                extraction_limit=args.limit,
                selected_streams=set(args.stream) if args.stream else None,
            )
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

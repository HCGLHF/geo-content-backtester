from __future__ import annotations

import argparse
import logging

from geo_backtester.chunking.chunker import chunk_article
from geo_backtester.config import config_from_env
from geo_backtester.demo import init_demo
from geo_backtester.ingestion.loader import load_article
from geo_backtester.pipeline import run_backtest


def inspect_chunks(article_path: str) -> None:
    config = config_from_env()
    article = load_article(article_path, "inspect")
    chunks = chunk_article(article, config.chunk_size, config.chunk_overlap)
    for chunk in chunks:
        print(f"\n{chunk.chunk_id} | tokens={chunk.token_count} | section={' > '.join(chunk.heading_path)}")
        print("-" * 80)
        print(chunk.text[:1000])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geo-backtester", description="GEO Content Backtester")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run old vs new article backtest")
    run_parser.add_argument("--old", required=True)
    run_parser.add_argument("--new", required=True)
    run_parser.add_argument("--queries", required=True)
    run_parser.add_argument("--entities")
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--mode", choices=["mvp", "realistic"], default="mvp")
    run_parser.add_argument("--labels")
    run_parser.add_argument("--background-corpus")
    run_parser.add_argument("--core-terms")

    subparsers.add_parser("init-demo", help="Create demo article, query, entity, label, and background files")

    inspect_parser = subparsers.add_parser("inspect-chunks", help="Print chunks for one article")
    inspect_parser.add_argument("--article", required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        result = run_backtest(
            args.old,
            args.new,
            args.queries,
            args.entities,
            args.output,
            mode=args.mode,
            labels_path=args.labels,
            background_corpus=args.background_corpus,
            core_terms_path=args.core_terms,
        )
        summary = result["score_summary"]
        print(f"Report written to: {result['report_path']}")
        print(
            f"Mode: {summary.get('mode', 'mvp')} | "
            f"Old score: {summary['old']['total_geo_score']} | "
            f"New score: {summary['new']['total_geo_score']} | "
            f"Winner: {summary['improvement']['winner']}"
        )
    elif args.command == "init-demo":
        init_demo()
    elif args.command == "inspect-chunks":
        inspect_chunks(args.article)


if __name__ == "__main__":
    main()

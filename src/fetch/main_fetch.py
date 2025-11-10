"""
Main fetch workflow for LLM Research Dashboard.

This module provides the main entry point for fetching papers from all sources.
It can be run as a script or imported as a module.

Usage:
    python -m src.fetch.main_fetch
    # or
    from src.fetch.main_fetch import main_fetch
    results = main_fetch(days=7)
"""

import argparse
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.fetch.fetch_manager import FetchManager
from src.storage.paper_db import PaperDB
from src.utils.logger import logger
from src.utils.config_loader import get_queries_config


def main_fetch(
    days: int = 7,
    sources: Optional[List[str]] = None,
    parallel: bool = True,
    store: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Main fetch workflow for all sources.

    Args:
        days: Number of days to look back for papers
        sources: List of sources to include ['arxiv', 'twitter', 'linkedin']
        parallel: Whether to fetch sources in parallel
        store: Whether to store papers in database
        verbose: Whether to print detailed progress

    Returns:
        Dictionary with fetch results
    """
    # Configure logger level
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

    logger.info("Starting main fetch workflow")
    logger.info(f"Configuration: days={days}, sources={sources}, parallel={parallel}, store={store}")

    # Initialize fetch manager
    manager = FetchManager()

    # Show fetcher status
    if verbose:
        status = manager.get_fetcher_status()
        logger.info("Fetcher Status:")
        for source, info in status.items():
            logger.info(f"  {source.capitalize()}: {info}")

    # Fetch papers
    if store:
        # Fetch and store in one step
        results = manager.fetch_and_store(
            days=days,
            parallel=parallel,
            include_sources=sources
        )
    else:
        # Just fetch without storing
        papers = manager.fetch_all_papers(
            days=days,
            parallel=parallel,
            include_sources=sources
        )
        results = {
            'papers_fetched': len(papers),
            'papers_stored': 0,
            'papers': papers,
            **manager.get_fetch_stats()
        }

    # Print summary
    if verbose:
        logger.info("=" * 60)
        logger.info("FETCH SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total papers fetched: {results['papers_fetched']}")
        logger.info(f"New papers stored: {results['papers_stored']}")
        logger.info(f"Duplicates removed: {results.get('duplicates_removed', 0)}")
        logger.info(f"Duration: {results.get('duration_seconds', 0):.2f} seconds")

        if 'source_counts' in results:
            logger.info("\nSource breakdown:")
            for source, count in results['source_counts'].items():
                logger.info(f"  {source.capitalize()}: {count} papers")

        if results.get('errors'):
            logger.warning(f"\nErrors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                logger.warning(f"  - {error}")

    logger.info("Main fetch workflow complete")
    return results


def update_papers_command():
    """Command-line interface for updating papers."""
    parser = argparse.ArgumentParser(description="Fetch LLM research papers from multiple sources")
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)"
    )
    parser.add_argument(
        "--sources", "-s",
        nargs="+",
        choices=["arxiv", "twitter", "linkedin"],
        help="Sources to fetch from (default: all)"
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Don't store papers in database"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Fetch sources sequentially instead of in parallel"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics after fetch"
    )

    args = parser.parse_args()

    # Run fetch
    results = main_fetch(
        days=args.days,
        sources=args.sources,
        parallel=not args.no_parallel,
        store=not args.no_store,
        verbose=not args.quiet
    )

    # Show stats if requested
    if args.stats and not args.quiet:
        manager = FetchManager()
        stats = manager.get_fetch_stats()

        print("\nDetailed Statistics:")
        print(f"  Fetch start time: {stats.get('fetch_start_time')}")
        print(f"  Fetch end time: {stats.get('fetch_end_time')}")

        if 'arxiv' in stats:
            print(f"\n  arXiv Fetcher:")
            print(f"    Last request: {stats['arxiv'].get('last_request')}")
            print(f"    Request count: {stats['arxiv'].get('request_count')}")

        if 'linkedin' in stats:
            print(f"\n  LinkedIn Fetcher:")
            print(f"    Mode: {stats['linkedin'].get('mode')}")
            print(f"    Daily count: {stats['linkedin'].get('daily_count')}")
            print(f"    Max daily: {stats['linkedin'].get('max_daily')}")

    return results


def daily_update():
    """
    Convenience function for daily scheduled updates.
    Fetches papers from the last day and stores them.
    """
    logger.info("Running daily scheduled update")
    results = main_fetch(days=1, parallel=True, store=True, verbose=True)

    # Check if we got any new papers
    if results['papers_stored'] > 0:
        logger.info(f"Daily update complete: {results['papers_stored']} new papers added")
    else:
        logger.info("Daily update complete: No new papers found")

    return results


def full_update():
    """
    Convenience function for full weekly updates.
    Fetches papers from the last 7 days and stores them.
    """
    logger.info("Running full weekly update")
    results = main_fetch(days=7, parallel=True, store=True, verbose=True)

    logger.info(f"Full update complete: {results['papers_stored']} new papers added out of {results['papers_fetched']} fetched")
    return results


def force_refresh():
    """
    Force refresh all papers, ignoring cache.
    """
    logger.warning("Running force refresh (ignoring cache)")
    manager = FetchManager()
    results = manager.force_refresh_all(days=7)

    logger.info(f"Force refresh complete: {results['papers_stored']} papers updated")
    return results


def check_status():
    """Check the status of all fetchers and recent fetch activity."""
    print("LLM Research Dashboard - Fetch Status")
    print("=" * 50)

    manager = FetchManager()
    status = manager.get_fetcher_status()

    for source, info in status.items():
        print(f"\n{source.upper()}:")
        print(f"  Initialized: {'Yes' if info['initialized'] else 'No'}")

        if source == 'arxiv':
            print(f"  Last request: {info['last_request'] or 'Never'}")
            print(f"  Request count: {info['request_count'] or 0}")
        elif source == 'twitter':
            print(f"  Config loaded: {'Yes' if info['config_loaded'] else 'No'}")
        elif source == 'linkedin':
            print(f"  Mode: {info['mode']}")
            print(f"  Papers fetched today: {info['daily_count']}")

    # Check database status
    print("\nDATABASE:")
    with PaperDB() as db:
        total_papers = db.get_paper_count()
        print(f"  Total papers: {total_papers}")

        # Recent papers
        recent = db.get_all_papers(limit=5, order_by="fetch_date DESC")
        if recent:
            print(f"\nMost recent papers:")
            for paper in recent[:3]:
                title = paper.get('title', 'No title')[:50]
                if len(title) == 50:
                    title += "..."
                print(f"  - {title} ({paper.get('source', 'unknown')})")


if __name__ == "__main__":
    # Run command-line interface
    results = update_papers_command()

    # Exit with error code if there were errors
    if results.get('errors'):
        sys.exit(1)
    else:
        sys.exit(0)
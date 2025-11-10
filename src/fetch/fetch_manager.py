"""
Fetch Manager - Coordinates all paper fetchers.

This module manages the coordinated fetching of papers from multiple sources:
- arXiv (primary research source)
- X/Twitter (social metrics)
- LinkedIn (professional metrics)

The fetch manager handles:
- Parallel fetching from all sources
- Rate limiting across sources
- Error handling and retries
- Progress tracking
- Integration with PaperDeduplicator
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from src.fetch.arxiv_fetcher import ArxivFetcher
from src.fetch.twitter_fetcher import TwitterFetcher
from src.fetch.linkedin_fetcher import LinkedinFetcher
from src.fetch.paper_deduplicator import PaperDeduplicator
from src.storage.paper_db import PaperDB
from src.utils.logger import logger
from src.utils.config_loader import get_queries_config


class FetchManager:
    """
    Coordinates fetching papers from all sources.

    Handles parallel fetching with rate limiting, error handling,
    and automatic deduplication across sources.
    """

    def __init__(self):
        """Initialize fetch manager with all fetchers."""
        self.logger = logger.bind(component="fetch_manager")

        # Load configuration
        self.config = get_queries_config()

        # Initialize fetchers
        self.arxiv_fetcher = ArxivFetcher()
        self.twitter_fetcher = TwitterFetcher()
        self.linkedin_fetcher = LinkedinFetcher()
        self.deduplicator = PaperDeduplicator()

        # Track statistics
        self.stats = {
            'fetch_start_time': None,
            'fetch_end_time': None,
            'arxiv_count': 0,
            'twitter_count': 0,
            'linkedin_count': 0,
            'total_before_dedup': 0,
            'total_after_dedup': 0,
            'duplicates_removed': 0,
            'errors': []
        }

        self.logger.info("Fetch manager initialized with all sources")

    def fetch_all_papers(
        self,
        days: int = 7,
        parallel: bool = True,
        include_sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch papers from all configured sources.

        Args:
            days: Number of days to look back for papers
            parallel: Whether to fetch sources in parallel
            include_sources: List of sources to include ['arxiv', 'twitter', 'linkedin']
                           If None, includes all configured sources

        Returns:
            List of deduplicated papers from all sources
        """
        self.stats['fetch_start_time'] = datetime.now(timezone.utc)
        self.logger.info(f"Starting coordinated fetch for last {days} days")

        # Determine which sources to fetch
        sources = include_sources or ['arxiv', 'twitter', 'linkedin']

        # Validate sources
        valid_sources = {'arxiv', 'twitter', 'linkedin'}
        sources = [s for s in sources if s in valid_sources]

        if not sources:
            self.logger.error("No valid sources to fetch")
            return []

        self.logger.info(f"Fetching from sources: {', '.join(sources)}")

        # Fetch from sources
        all_papers = []

        if parallel:
            # Parallel fetching with thread pool
            with ThreadPoolExecutor(max_workers=len(sources)) as executor:
                futures = {}

                # Submit fetch tasks
                if 'arxiv' in sources:
                    future = executor.submit(self._fetch_arxiv_safe, days)
                    futures[future] = 'arxiv'

                if 'twitter' in sources:
                    future = executor.submit(self._fetch_twitter_safe, days)
                    futures[future] = 'twitter'

                if 'linkedin' in sources:
                    future = executor.submit(self._fetch_linkedin_safe, days)
                    futures[future] = 'linkedin'

                # Collect results
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        papers = future.result()
                        self.logger.info(f"Received {len(papers)} papers from {source}")
                        all_papers.extend(papers)
                        self.stats[f'{source}_count'] = len(papers)
                    except Exception as e:
                        error_msg = f"Error fetching from {source}: {e}"
                        self.logger.error(error_msg)
                        self.stats['errors'].append(error_msg)
        else:
            # Sequential fetching
            if 'arxiv' in sources:
                papers = self._fetch_arxiv_safe(days)
                all_papers.extend(papers)
                self.stats['arxiv_count'] = len(papers)

            if 'twitter' in sources:
                papers = self._fetch_twitter_safe(days)
                all_papers.extend(papers)
                self.stats['twitter_count'] = len(papers)

            if 'linkedin' in sources:
                papers = self._fetch_linkedin_safe(days)
                all_papers.extend(papers)
                self.stats['linkedin_count'] = len(papers)

        # Update statistics
        self.stats['total_before_dedup'] = len(all_papers)
        self.logger.info(f"Fetched {len(all_papers)} total papers before deduplication")

        # Deduplicate papers
        self.logger.info("Deduplicating papers across sources")
        deduplicated_papers = self.deduplicator.deduplicate(all_papers)

        # Update final statistics
        self.stats['total_after_dedup'] = len(deduplicated_papers)
        self.stats['duplicates_removed'] = len(all_papers) - len(deduplicated_papers)
        self.stats['fetch_end_time'] = datetime.now(timezone.utc)

        # Log summary
        duration = (self.stats['fetch_end_time'] - self.stats['fetch_start_time']).total_seconds()
        self.logger.info(
            f"Fetch complete in {duration:.2f}s: "
            f"{self.stats['total_after_dedup']} unique papers "
            f"({self.stats['duplicates_removed']} duplicates removed)",
            extra={
                "duration_seconds": duration,
                "arxiv_count": self.stats['arxiv_count'],
                "twitter_count": self.stats['twitter_count'],
                "linkedin_count": self.stats['linkedin_count'],
                "duplicates_removed": self.stats['duplicates_removed'],
                "error_count": len(self.stats['errors'])
            }
        )

        return deduplicated_papers

    def _fetch_arxiv_safe(self, days: int) -> List[Dict[str, Any]]:
        """Fetch from arXiv with error handling."""
        try:
            self.logger.debug("Fetching from arXiv")
            papers = self.arxiv_fetcher.fetch_recent_papers(days=days)

            # Add source identifier if not present
            for paper in papers:
                if 'source' not in paper:
                    paper['source'] = 'arxiv'

            return papers
        except Exception as e:
            self.logger.error(f"arXiv fetch failed: {e}", exc_info=True)
            return []

    def _fetch_twitter_safe(self, days: int) -> List[Dict[str, Any]]:
        """Fetch from X/Twitter with error handling."""
        try:
            self.logger.debug("Fetching from X/Twitter")
            papers = self.twitter_fetcher.fetch_recent_papers(days=days)

            # Add source identifier if not present
            for paper in papers:
                if 'source' not in paper:
                    paper['source'] = 'twitter'

            return papers
        except Exception as e:
            self.logger.error(f"Twitter fetch failed: {e}", exc_info=True)
            return []

    def _fetch_linkedin_safe(self, days: int) -> List[Dict[str, Any]]:
        """Fetch from LinkedIn with error handling."""
        try:
            self.logger.debug("Fetching from LinkedIn")
            papers = self.linkedin_fetcher.fetch_papers()

            # Filter by days if needed
            if days < 7:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                filtered = []
                for paper in papers:
                    if paper.get('published_date'):
                        try:
                            pub_date = datetime.fromisoformat(paper['published_date'])
                            if pub_date >= cutoff_date:
                                filtered.append(paper)
                        except (ValueError, TypeError):
                            # Keep if date parsing fails
                            filtered.append(paper)
                papers = filtered

            # Add source identifier if not present
            for paper in papers:
                if 'source' not in paper:
                    paper['source'] = 'linkedin'

            return papers
        except Exception as e:
            self.logger.error(f"LinkedIn fetch failed: {e}", exc_info=True)
            return []

    def fetch_and_store(
        self,
        days: int = 7,
        parallel: bool = True,
        include_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch papers from all sources and store in database.

        Args:
            days: Number of days to look back
            parallel: Whether to fetch sources in parallel
            include_sources: List of sources to include

        Returns:
            Dictionary with fetch results and statistics
        """
        # Fetch papers
        papers = self.fetch_all_papers(days=days, parallel=parallel, include_sources=include_sources)

        # Store in database
        stored_count = 0
        if papers:
            self.logger.info(f"Storing {len(papers)} papers in database")

            with PaperDB() as db:
                for paper in papers:
                    try:
                        # Check if paper already exists
                        if not db.paper_exists(paper['id']):
                            db.insert_paper(paper)
                            stored_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to store paper {paper['id']}: {e}")

        # Return results
        results = {
            'papers_fetched': len(papers),
            'papers_stored': stored_count,
            'duplicates_removed': self.stats['duplicates_removed'],
            'source_counts': {
                'arxiv': self.stats['arxiv_count'],
                'twitter': self.stats['twitter_count'],
                'linkedin': self.stats['linkedin_count']
            },
            'duration_seconds': (
                self.stats['fetch_end_time'] - self.stats['fetch_start_time']
            ).total_seconds() if self.stats['fetch_end_time'] else 0,
            'errors': self.stats['errors']
        }

        self.logger.info(
            f"Fetch and store complete: {stored_count}/{len(papers)} new papers stored",
            extra=results
        )

        return results

    def get_fetch_stats(self) -> Dict[str, Any]:
        """Get statistics from the last fetch."""
        stats = self.stats.copy()

        # Add fetcher-specific stats
        stats['arxiv'] = self.arxiv_fetcher.get_stats()
        stats['twitter'] = self.twitter_fetcher.get_stats()
        stats['linkedin'] = self.linkedin_fetcher.get_stats()

        return stats

    def get_fetcher_status(self) -> Dict[str, Any]:
        """Get status of all fetchers."""
        return {
            'arxiv': {
                'initialized': self.arxiv_fetcher is not None,
                'last_request': getattr(self.arxiv_fetcher, 'last_request_time', None),
                'request_count': getattr(self.arxiv_fetcher, 'request_count', None)
            },
            'twitter': {
                'initialized': self.twitter_fetcher is not None,
                'config_loaded': hasattr(self.twitter_fetcher, 'tracked_accounts')
            },
            'linkedin': {
                'initialized': self.linkedin_fetcher is not None,
                'mode': getattr(self.linkedin_fetcher, 'mode', 'unknown'),
                'daily_count': getattr(self.linkedin_fetcher.cache, 'daily_fetch_count', 0) if hasattr(self.linkedin_fetcher, 'cache') else 0
            }
        }

    def update_daily_papers(self, days: int = 1) -> Dict[str, Any]:
        """
        Convenience method to fetch today's papers only.

        Args:
            days: Number of recent days to update (default: 1)

        Returns:
            Fetch results dictionary
        """
        self.logger.info(f"Updating papers for last {days} day(s)")
        return self.fetch_and_store(days=days, parallel=True)

    def force_refresh_all(self, days: int = 7) -> Dict[str, Any]:
        """
        Force refresh all papers, ignoring cache.

        Args:
            days: Number of days to refresh

        Returns:
            Fetch results dictionary
        """
        self.logger.warning(f"Force refreshing all papers for last {days} days")

        # Clear caches
        if hasattr(self.arxiv_fetcher, '_seen_ids'):
            self.arxiv_fetcher._seen_ids.clear()

        if hasattr(self.linkedin_fetcher, 'cache'):
            self.linkedin_fetcher.cache.seen_posts.clear()

        # Fetch and store
        return self.fetch_and_store(days=days, parallel=True)


# Convenience function for simple usage
def fetch_papers(days: int = 7, sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch papers from all sources.

    Args:
        days: Number of days to look back
        sources: List of sources to include

    Returns:
        List of deduplicated papers
    """
    manager = FetchManager()
    return manager.fetch_all_papers(days=days, include_sources=sources)


def fetch_and_store_papers(days: int = 7, sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to fetch and store papers.

    Args:
        days: Number of days to look back
        sources: List of sources to include

    Returns:
        Dictionary with fetch results
    """
    manager = FetchManager()
    return manager.fetch_and_store(days=days, include_sources=sources)
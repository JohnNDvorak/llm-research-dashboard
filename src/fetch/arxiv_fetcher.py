"""Fetch papers from arXiv API with rate limiting and metadata extraction."""

import arxiv
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Iterator
from urllib.parse import parse_qs, urlparse
import re

from src.utils.logger import logger
from src.utils.config_loader import load_config


class ArxivFetcher:
    """Fetch papers from arXiv using official API with rate limiting."""

    def __init__(self):
        """Initialize arXiv fetcher with configuration."""
        self.config = load_config('queries')
        self.arxiv_config = self.config['arxiv']
        self.last_request_time = 0
        self.request_count = 0

        # Initialize logger
        self.logger = logger.bind(component="arxiv_fetcher")

        # Cache for recently seen papers (avoid duplicates)
        self._seen_ids = set()

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between arXiv API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        delay = self.arxiv_config['rate_limit_delay']
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """Extract arXiv ID from URL or identifier."""
        if not url:
            return None

        # Handle different arXiv URL formats
        patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',
            r'arxiv\.org/pdf/(\d+\.\d+)',
            r'^(\d+\.\d+)$',  # Direct ID
            r'arxiv:(\d+\.\d+)',  # arxiv: prefix
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _format_authors(self, authors) -> List[str]:
        """Format arXiv author objects into list of strings."""
        formatted_authors = []
        for author in authors:
            name = str(author)
            # Clean up author names
            name = re.sub(r'\s+', ' ', name).strip()
            if name:
                formatted_authors.append(name)
        return formatted_authors

    def _parse_paper_metadata(self, result) -> Dict[str, Any]:
        """Parse arXiv result into standardized paper dictionary."""
        # Extract arXiv ID
        arxiv_id = self._extract_arxiv_id(result.entry_id)

        # Format authors
        authors = self._format_authors(result.authors)

        # Parse published date
        published_date = result.published.date() if result.published else None

        # Extract categories
        categories = list(result.categories) if result.categories else []

        # Build paper dictionary matching our database schema
        paper = {
            'id': f"arxiv:{arxiv_id}" if arxiv_id else result.entry_id,
            'title': result.title.strip(),
            'abstract': result.summary.strip(),
            'authors': authors,
            'published_date': published_date.isoformat() if published_date else None,

            # URLs
            'url': result.entry_id,
            'pdf_url': result.pdf_url,

            # Source information
            'source': 'arxiv',
            'fetch_date': datetime.now().isoformat(),

            # Note: arXiv-specific fields like categories, doi are not in current schema
            # Could be added to metadata field or new columns in future migration

            # Social metrics (initially 0 for arXiv)
            'social_score': 0,
            'professional_score': 0,

            # Analysis fields (initially empty)
            'analyzed': False,
            'stages': None,
            'summary': None,
            'key_insights': None,
            'metrics': None,
            'complexity_score': None,

            # LLM tracking
            'model_used': None,
            'analysis_cost': None,
        }

        return paper

    def search_papers(self, query: str, max_results: int = 100, sort_by: str = "submittedDate") -> Iterator[Dict[str, Any]]:
        """
        Search arXiv for papers matching query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            sort_by: Sort criteria ("submittedDate", "lastUpdatedDate", "relevance")

        Yields:
            Paper dictionaries
        """
        self.logger.info(f"Searching arXiv: query='{query}', max_results={max_results}")

        # Build search with category filters
        search_query = query
        categories = self.arxiv_config.get('categories', [])
        if categories:
            # Add category filters
            cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
            search_query = f"({query}) AND ({cat_filter})"

        self.logger.debug(f"Final search query: {search_query}")

        # Configure sort order
        sort_options = {
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "relevance": arxiv.SortCriterion.Relevance
        }
        sort_criterion = sort_options.get(sort_by, arxiv.SortCriterion.SubmittedDate)

        try:
            # Enforce rate limit
            self._enforce_rate_limit()

            # Create search client
            client = arxiv.Client(
                page_size=min(max_results, 100),  # arxiv limit
                delay_seconds=0,  # We handle our own rate limiting
                num_retries=3
            )

            # Build search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=sort_criterion,
                sort_order=arxiv.SortOrder.Descending
            )

            # Execute search
            results = list(client.results(search))

            self.logger.info(f"Found {len(results)} papers for query: {query}")

            # Process each result
            for result in results:
                try:
                    paper = self._parse_paper_metadata(result)

                    # Skip if we've already seen this paper
                    if paper['id'] in self._seen_ids:
                        self.logger.debug(f"Skipping duplicate paper: {paper['id']}")
                        continue

                    self._seen_ids.add(paper['id'])

                    # Log paper info
                    self.logger.debug(
                        f"Fetched paper: {paper['title'][:50]}... "
                        f"[{paper['id']}]"
                    )

                    yield paper

                except Exception as e:
                    self.logger.error(f"Error parsing paper {result.entry_id}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error searching arXiv with query '{query}': {e}")
            raise

    def fetch_by_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 500
    ) -> Iterator[Dict[str, Any]]:
        """
        Fetch papers published within date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_results: Maximum total results

        Yields:
            Paper dictionaries
        """
        # Use config defaults if not provided
        if not start_date:
            start_date = self.arxiv_config['date_range']['start']
        if not end_date:
            end_date = self.arxiv_config['date_range']['end']

        self.logger.info(f"Fetching papers from {start_date} to {end_date}")

        # Build date query
        date_query = ""
        if start_date:
            date_query += f"submittedDate:[{start_date}0000 TO "
        else:
            date_query += "submittedDate:[* TO "

        if end_date:
            date_query += f"{end_date}2359]"
        else:
            date_query += "*]"

        # Use all configured queries
        all_queries = self.arxiv_config['queries']
        total_papers = 0

        for query in all_queries:
            if total_papers >= max_results:
                break

            # Combine with date filter
            full_query = f"({query}) AND {date_query}"
            remaining = max_results - total_papers

            try:
                for paper in self.search_papers(full_query, remaining):
                    total_papers += 1
                    yield paper

                    if total_papers >= max_results:
                        break

            except Exception as e:
                self.logger.error(f"Error fetching papers for query '{query}': {e}")
                continue

    def fetch_recent_papers(self, days: int = 7, max_results: int = 200) -> Iterator[Dict[str, Any]]:
        """
        Fetch papers from the last N days.

        Args:
            days: Number of days to look back
            max_results: Maximum results to return

        Yields:
            Paper dictionaries
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        self.logger.info(f"Fetching papers from last {days} days ({start_str} to {end_str})")

        yield from self.fetch_by_date_range(start_str, end_str, max_results)

    def fetch_paper_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific paper by arXiv ID.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2401.00001" or "arxiv:2401.00001")

        Returns:
            Paper dictionary or None if not found
        """
        # Clean ID
        clean_id = self._extract_arxiv_id(arxiv_id)
        if not clean_id:
            clean_id = arxiv_id

        self.logger.info(f"Fetching paper by ID: {clean_id}")

        try:
            # Search for specific paper
            search_query = f"id:{clean_id}"

            for paper in self.search_papers(search_query, max_results=1):
                return paper

            self.logger.warning(f"Paper not found: {arxiv_id}")
            return None

        except Exception as e:
            self.logger.error(f"Error fetching paper {arxiv_id}: {e}")
            return None

    def get_categories(self) -> List[str]:
        """Get list of configured arXiv categories."""
        return self.arxiv_config.get('categories', [])

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        return {
            'requests_made': self.request_count,
            'papers_seen': len(self._seen_ids),
            'last_request_time': datetime.fromtimestamp(self.last_request_time).isoformat() if self.last_request_time else None,
            'configured_queries': len(self.arxiv_config['queries']),
            'configured_categories': self.arxiv_config['categories']
        }
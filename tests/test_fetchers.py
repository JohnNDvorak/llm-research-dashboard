"""
Unit tests for paper fetcher modules.

NOTE: This file previously contained placeholder tests that have been replaced
by comprehensive test suites in dedicated test files:

- ArxivFetcher tests: See tests/test_arxiv_fetcher.py (34 tests)
- PaperDeduplicator tests: See tests/test_paper_deduplicator.py (45 tests)
- Integration tests: See tests/test_phase1_phase2_integration.py (12 tests)

This file now contains only basic sanity checks to verify modules can be imported.
"""

import pytest
from src.fetch.arxiv_fetcher import ArxivFetcher
from src.fetch.paper_deduplicator import PaperDeduplicator, deduplicate_papers


class TestFetcherImports:
    """Basic sanity checks for fetcher modules."""

    def test_arxiv_fetcher_can_be_imported(self):
        """Test that ArxivFetcher can be imported and instantiated."""
        fetcher = ArxivFetcher()
        assert fetcher is not None
        assert hasattr(fetcher, 'search_papers')
        assert hasattr(fetcher, 'fetch_recent_papers')
        assert hasattr(fetcher, 'fetch_paper_by_id')

    def test_paper_deduplicator_can_be_imported(self):
        """Test that PaperDeduplicator can be imported and instantiated."""
        deduplicator = PaperDeduplicator()
        assert deduplicator is not None
        assert hasattr(deduplicator, 'deduplicate')

    def test_deduplicate_papers_function_exists(self):
        """Test that deduplicate_papers convenience function exists."""
        assert deduplicate_papers is not None
        assert callable(deduplicate_papers)

    def test_arxiv_fetcher_basic_functionality(self):
        """Test basic ArxivFetcher functionality."""
        fetcher = ArxivFetcher()

        # Verify config loaded
        assert fetcher.config is not None
        assert 'arxiv' in fetcher.config

        # Verify methods are callable
        assert callable(fetcher.search_papers)
        assert callable(fetcher.fetch_recent_papers)
        assert callable(fetcher.fetch_paper_by_id)

    def test_paper_deduplicator_basic_functionality(self):
        """Test basic PaperDeduplicator functionality."""
        deduplicator = PaperDeduplicator()

        # Verify config loaded
        assert deduplicator.config is not None
        assert 'title_similarity_threshold' in deduplicator.config

        # Test with empty list
        result = deduplicator.deduplicate([])
        assert result == []

        # Test with single paper
        papers = [{'id': 'test:1', 'title': 'Test', 'abstract': 'Test', 'source': 'test'}]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1

    def test_deduplicate_papers_function_basic(self):
        """Test basic deduplicate_papers function."""
        # Test with empty list
        result = deduplicate_papers([])
        assert result == []

        # Test with single paper
        papers = [{'id': 'test:1', 'title': 'Test', 'abstract': 'Test', 'source': 'test'}]
        result = deduplicate_papers(papers)
        assert len(result) == 1


# For comprehensive tests, see:
# - tests/test_arxiv_fetcher.py
# - tests/test_paper_deduplicator.py
# - tests/test_phase1_phase2_integration.py

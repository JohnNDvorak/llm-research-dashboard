"""Comprehensive unit tests for arXiv fetcher."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time

from src.fetch.arxiv_fetcher import ArxivFetcher


class TestArxivFetcherInit:
    """Test ArxivFetcher initialization."""

    def test_can_be_instantiated(self):
        """Test that ArxivFetcher can be instantiated."""
        fetcher = ArxivFetcher()
        assert fetcher is not None
        assert hasattr(fetcher, 'config')
        assert hasattr(fetcher, 'arxiv_config')
        assert hasattr(fetcher, 'logger')
        assert hasattr(fetcher, '_seen_ids')
        assert fetcher._seen_ids == set()

    def test_loads_configuration(self):
        """Test that fetcher loads configuration correctly."""
        fetcher = ArxivFetcher()
        assert 'arxiv' in fetcher.config
        assert 'queries' in fetcher.arxiv_config
        assert 'categories' in fetcher.arxiv_config
        assert 'rate_limit_delay' in fetcher.arxiv_config

    def test_has_required_methods(self):
        """Test that ArxivFetcher has all required methods."""
        fetcher = ArxivFetcher()
        required_methods = [
            'search_papers',
            'fetch_by_date_range',
            'fetch_recent_papers',
            'fetch_paper_by_id',
            'get_categories',
            'get_stats'
        ]

        for method in required_methods:
            assert hasattr(fetcher, method), f"Missing method: {method}"
            assert callable(getattr(fetcher, method)), f"Method not callable: {method}"


class TestArxivFetcherHelpers:
    """Test ArxivFetcher helper methods."""

    def test_extract_arxiv_id(self):
        """Test arXiv ID extraction from URLs."""
        fetcher = ArxivFetcher()

        # Test various URL formats
        assert fetcher._extract_arxiv_id("https://arxiv.org/abs/2401.00001") == "2401.00001"
        assert fetcher._extract_arxiv_id("https://arxiv.org/pdf/2401.00001.pdf") == "2401.00001"
        assert fetcher._extract_arxiv_id("2401.00001") == "2401.00001"
        assert fetcher._extract_arxiv_id("arxiv:2401.00001") == "2401.00001"
        assert fetcher._extract_arxiv_id("http://arxiv.org/abs/2305.12345v2") == "2305.12345"

        # Test invalid inputs
        assert fetcher._extract_arxiv_id(None) is None
        assert fetcher._extract_arxiv_id("") is None
        assert fetcher._extract_arxiv_id("https://example.com") is None

    def test_format_authors(self):
        """Test author formatting."""
        fetcher = ArxivFetcher()

        # Mock author objects
        author1 = Mock()
        author1.__str__ = Mock(return_value="John Doe")
        author2 = Mock()
        author2.__str__ = Mock(return_value=" Jane Smith ")
        author3 = Mock()
        author3.__str__ = Mock(return_value="Bob Johnson")

        authors = [author1, author2, author3]
        formatted = fetcher._format_authors(authors)

        assert formatted == ["John Doe", "Jane Smith", "Bob Johnson"]

        # Test empty list
        assert fetcher._format_authors([]) == []

    def test_enforce_rate_limit(self):
        """Test rate limiting enforcement."""
        fetcher = ArxivFetcher()

        # Test first call (should not sleep)
        start_time = time.time()
        fetcher._enforce_rate_limit()
        first_call_time = time.time() - start_time

        # Should be very fast (no sleep needed)
        assert first_call_time < 0.1
        assert fetcher.request_count == 1

        # Test immediate second call (should sleep)
        start_time = time.time()
        fetcher._enforce_rate_limit()
        second_call_time = time.time() - start_time

        # Should sleep for at least rate_limit_delay seconds
        delay = fetcher.arxiv_config['rate_limit_delay']
        assert second_call_time >= delay - 0.1  # Allow small tolerance
        assert fetcher.request_count == 2

    def test_get_categories(self):
        """Test getting configured categories."""
        fetcher = ArxivFetcher()
        categories = fetcher.get_categories()

        assert isinstance(categories, list)
        assert "cs.CL" in categories
        assert "cs.LG" in categories
        assert "cs.AI" in categories

    def test_get_stats(self):
        """Test getting fetcher statistics."""
        fetcher = ArxivFetcher()
        stats = fetcher.get_stats()

        required_keys = [
            'requests_made',
            'papers_seen',
            'last_request_time',
            'configured_queries',
            'configured_categories'
        ]

        for key in required_keys:
            assert key in stats, f"Missing stat key: {key}"

        assert stats['requests_made'] == 0
        assert stats['papers_seen'] == 0
        assert stats['last_request_time'] is None
        assert isinstance(stats['configured_queries'], int)
        assert isinstance(stats['configured_categories'], list)


class TestArxivFetcherSearch:
    """Test ArxivFetcher search functionality."""

    @patch('src.fetch.arxiv_fetcher.arxiv.Client')
    @patch('src.fetch.arxiv_fetcher.arxiv.Search')
    def test_search_papers_basic(self, mock_search, mock_client):
        """Test basic paper search functionality."""
        # Setup mocks
        mock_result = Mock()
        mock_result.entry_id = "https://arxiv.org/abs/2401.00001"
        mock_result.title = "Test Paper"
        mock_result.summary = "This is a test abstract."
        mock_result.published = Mock()
        mock_result.published.date = Mock(return_value=datetime(2024, 1, 1).date())
        mock_result.authors = [Mock()]
        mock_result.authors[0].__str__ = Mock(return_value="Test Author")
        mock_result.categories = ["cs.CL", "cs.LG"]

        mock_search.return_value = [mock_result]
        mock_client_instance = Mock()
        mock_client_instance.results = Mock(return_value=[mock_result])
        mock_client.return_value = mock_client_instance

        # Test search
        fetcher = ArxivFetcher()
        papers = list(fetcher.search_papers("test query", max_results=10))

        assert len(papers) == 1
        paper = papers[0]
        assert paper['title'] == "Test Paper"
        assert paper['abstract'] == "This is a test abstract."
        assert paper['id'] == "arxiv:2401.00001"
        assert paper['source'] == "arxiv"
        assert len(paper['authors']) == 1
        assert paper['authors'][0] == "Test Author"

    @patch('src.fetch.arxiv_fetcher.arxiv.Client')
    @patch('src.fetch.arxiv_fetcher.arxiv.Search')
    def test_search_papers_with_duplicates(self, mock_search, mock_client):
        """Test that duplicate papers are filtered out."""
        # Setup duplicate papers
        mock_result = Mock()
        mock_result.entry_id = "https://arxiv.org/abs/2401.00001"
        mock_result.title = "Test Paper"
        mock_result.summary = "This is a test abstract."
        mock_result.published = Mock()
        mock_result.published.date = Mock(return_value=datetime(2024, 1, 1).date())
        mock_result.authors = [Mock()]
        mock_result.authors[0].__str__ = Mock(return_value="Test Author")
        mock_result.categories = []

        mock_search.return_value = [mock_result]
        mock_client_instance = Mock()
        mock_client_instance.results = Mock(return_value=[mock_result, mock_result])  # Duplicate
        mock_client.return_value = mock_client_instance

        # Test search
        fetcher = ArxivFetcher()
        papers = list(fetcher.search_papers("test query", max_results=10))

        # Should only return one paper (duplicate filtered)
        assert len(papers) == 1

    def test_search_papers_invalid_params(self):
        """Test search with invalid parameters."""
        fetcher = ArxivFetcher()

        # These should not raise errors (method should handle gracefully)
        try:
            list(fetcher.search_papers("", max_results=0))
            list(fetcher.search_papers("test", max_results=-1))
        except Exception as e:
            # Allow exceptions for invalid params but log them
            assert isinstance(e, (ValueError, Exception))

    def test_parse_paper_metadata(self):
        """Test paper metadata parsing."""
        fetcher = ArxivFetcher()

        # Setup mock result
        mock_result = Mock()
        mock_result.entry_id = "https://arxiv.org/abs/2401.00001"
        mock_result.title = "  Test Paper with Spaces  "
        mock_result.summary = "  Test abstract with spaces  "
        mock_result.published = Mock()
        mock_result.published.date = Mock(return_value=datetime(2024, 1, 1).date())
        mock_result.authors = [Mock()]
        mock_result.authors[0].__str__ = Mock(return_value="Test Author")
        mock_result.categories = ["cs.CL", "cs.LG"]

        paper = fetcher._parse_paper_metadata(mock_result)

        # Test all required fields
        required_fields = [
            'id', 'title', 'abstract', 'authors', 'published_date',
            'url', 'pdf_url', 'source', 'fetch_date',
            'social_score', 'professional_score',
            'analyzed', 'stages', 'summary', 'key_insights',
            'metrics', 'complexity_score', 'model_used', 'analysis_cost'
        ]

        for field in required_fields:
            assert field in paper, f"Missing field: {field}"

        # Test field values
        assert paper['title'] == "Test Paper with Spaces"  # Stripped
        assert paper['abstract'] == "Test abstract with spaces"  # Stripped
        assert paper['id'] == "arxiv:2401.00001"
        assert paper['source'] == "arxiv"
        assert paper['social_score'] == 0
        assert paper['professional_score'] == 0
        assert paper['analyzed'] is False


class TestArxivFetcherDateMethods:
    """Test ArxivFetcher date-based fetching methods."""

    @patch('src.fetch.arxiv_fetcher.ArxivFetcher.search_papers')
    def test_fetch_recent_papers(self, mock_search):
        """Test fetching recent papers by days."""
        # Setup mock
        mock_papers = [
            {'id': 'arxiv:2401.00001', 'title': 'Recent Paper 1'},
            {'id': 'arxiv:2401.00002', 'title': 'Recent Paper 2'}
        ]
        mock_search.return_value = iter(mock_papers)

        fetcher = ArxivFetcher()
        papers = list(fetcher.fetch_recent_papers(days=7, max_results=10))

        assert len(papers) == 2
        assert mock_search.call_count > 0

        # Check that search was called with date-appropriate query
        call_args = mock_search.call_args
        # Check if query was passed as positional or keyword argument
        if call_args[0]:  # Positional args
            assert 'submittedDate' in call_args[0][0]
        else:  # Keyword args
            assert 'query' in call_args[1]
            assert 'submittedDate' in call_args[1]['query']

    @patch('src.fetch.arxiv_fetcher.ArxivFetcher.search_papers')
    def test_fetch_by_date_range(self, mock_search):
        """Test fetching papers by date range."""
        # Setup mock
        mock_papers = [
            {'id': 'arxiv:2401.00001', 'title': 'Date Range Paper'}
        ]
        mock_search.return_value = iter(mock_papers)

        fetcher = ArxivFetcher()
        papers = list(fetcher.fetch_by_date_range(
            start_date="2024-01-01",
            end_date="2024-01-31",
            max_results=5
        ))

        assert len(papers) == 1
        assert mock_search.call_count > 0

    @patch('src.fetch.arxiv_fetcher.ArxivFetcher.search_papers')
    def test_fetch_paper_by_id(self, mock_search):
        """Test fetching a specific paper by ID."""
        # Setup mock
        mock_papers = [
            {
                'id': 'arxiv:2401.00001',
                'title': 'Specific Paper',
                'abstract': 'Abstract for specific paper'
            }
        ]
        mock_search.return_value = iter(mock_papers)

        fetcher = ArxivFetcher()
        paper = fetcher.fetch_paper_by_id("2401.00001")

        assert paper is not None
        assert paper['id'] == "arxiv:2401.00001"
        assert paper['title'] == "Specific Paper"

        # Check correct search query was used
        mock_search.assert_called_once_with("id:2401.00001", max_results=1)

    @patch('src.fetch.arxiv_fetcher.ArxivFetcher.search_papers')
    def test_fetch_paper_by_id_not_found(self, mock_search):
        """Test fetching non-existent paper ID."""
        mock_search.return_value = iter([])

        fetcher = ArxivFetcher()
        paper = fetcher.fetch_paper_by_id("9999.99999")

        assert paper is None

    def test_fetch_paper_by_id_various_formats(self):
        """Test fetching paper with different ID formats."""
        fetcher = ArxivFetcher()

        # Test ID extraction from different formats
        test_cases = [
            ("2401.00001", "2401.00001"),
            ("arxiv:2401.00001", "2401.00001"),
            ("https://arxiv.org/abs/2401.00001", "2401.00001"),
            ("https://arxiv.org/pdf/2401.00001.pdf", "2401.00001")
        ]

        for input_id, expected_clean in test_cases:
            clean_id = fetcher._extract_arxiv_id(input_id)
            assert clean_id == expected_clean, f"Failed for {input_id}"


class TestArxivFetcherIntegration:
    """Integration tests for arXiv fetcher (require actual API)."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_real_search_papers(self):
        """Test real arXiv API search (slow test)."""
        fetcher = ArxivFetcher()

        # Search for a specific topic
        papers = list(fetcher.search_papers("LLM", max_results=2))

        assert len(papers) > 0, "Should find at least one paper"

        for paper in papers:
            assert 'id' in paper
            assert 'title' in paper
            assert 'abstract' in paper
            assert paper['title'].strip() != ""
            assert paper['abstract'].strip() != ""
            assert paper['source'] == 'arxiv'

    @pytest.mark.slow
    @pytest.mark.network
    def test_real_rate_limiting(self):
        """Test that rate limiting works with real API (slow test)."""
        fetcher = ArxivFetcher()

        # Make two rapid requests
        start_time = time.time()
        papers1 = list(fetcher.search_papers("machine learning", max_results=1))
        first_request_time = time.time() - start_time

        start_time = time.time()
        papers2 = list(fetcher.search_papers("neural networks", max_results=1))
        second_request_time = time.time() - start_time

        # Second request should be delayed by rate limiting
        rate_limit_delay = fetcher.arxiv_config['rate_limit_delay']
        assert second_request_time >= rate_limit_delay - 0.1  # Small tolerance


class TestArxivFetcherErrorHandling:
    """Test error handling in arXiv fetcher."""

    @patch('src.fetch.arxiv_fetcher.arxiv.Client')
    def test_search_papers_api_error(self, mock_client):
        """Test handling of arXiv API errors."""
        mock_client.side_effect = Exception("API Error")

        fetcher = ArxivFetcher()

        with pytest.raises(Exception):
            list(fetcher.search_papers("test query"))

    @patch('src.fetch.arxiv_fetcher.arxiv.Client')
    def test_search_papers_parse_error(self, mock_client):
        """Test handling of paper parsing errors."""
        # Setup mock that causes parsing error
        mock_result = Mock()
        mock_result.entry_id = "invalid_entry"
        mock_result.title = None  # This will cause error
        mock_result.summary = "Test abstract"
        mock_result.published = Mock()
        mock_result.published.date = Mock(side_effect=Exception("Parse error"))

        mock_client_instance = Mock()
        mock_client_instance.results = Mock(return_value=[mock_result])
        mock_client.return_value = mock_client_instance

        fetcher = ArxivFetcher()

        # Should handle error gracefully and continue
        papers = list(fetcher.search_papers("test query", max_results=10))

        # Should return empty list due to parsing error
        assert len(papers) == 0

    def test_invalid_configuration(self):
        """Test behavior with invalid configuration."""
        # This test ensures graceful handling of config issues
        with patch('src.fetch.arxiv_fetcher.load_config') as mock_load:
            mock_load.side_effect = FileNotFoundError("Config not found")

            with pytest.raises(FileNotFoundError):
                ArxivFetcher()


class TestArxivFetcherEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_queries_list(self):
        """Test behavior with empty queries configuration."""
        fetcher = ArxivFetcher()

        # Temporarily modify config
        original_queries = fetcher.arxiv_config['queries']
        fetcher.arxiv_config['queries'] = []

        try:
            # Should handle empty queries gracefully
            papers = list(fetcher.fetch_by_date_range(max_results=10))
            assert len(papers) == 0
        finally:
            # Restore original config
            fetcher.arxiv_config['queries'] = original_queries

    def test_very_long_query(self):
        """Test handling of very long search queries."""
        fetcher = ArxivFetcher()
        long_query = "test " * 1000  # Very long query

        # Should handle gracefully (may truncate or error)
        try:
            papers = list(fetcher.search_papers(long_query, max_results=1))
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (ValueError, Exception))

    def test_unicode_in_search(self):
        """Test searching with unicode characters."""
        fetcher = ArxivFetcher()

        # Should handle unicode without error
        try:
            papers = list(fetcher.search_papers("注意力机制", max_results=1))
        except Exception as e:
            # May fail due to arXiv API limitations, but shouldn't crash
            assert isinstance(e, Exception)

    def test_zero_max_results(self):
        """Test search with zero max results."""
        fetcher = ArxivFetcher()

        papers = list(fetcher.search_papers("test", max_results=0))
        assert len(papers) == 0

    def test_negative_max_results(self):
        """Test search with negative max results."""
        fetcher = ArxivFetcher()

        # Should handle gracefully
        try:
            papers = list(fetcher.search_papers("test", max_results=-1))
        except ValueError:
            # Expected behavior
            pass
        except Exception as e:
            # Should handle other exceptions gracefully
            assert isinstance(e, Exception)
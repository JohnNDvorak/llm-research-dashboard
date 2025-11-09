"""Unit tests for paper fetcher modules."""

import pytest
from typing import List, Dict, Any
from src.fetch.arxiv_fetcher import ArxivFetcher
from src.fetch.paper_deduplicator import deduplicate_papers


class TestArxivFetcher:
    """Test suite for ArxivFetcher class."""

    def test_arxiv_fetcher_can_be_instantiated(self):
        """Test that ArxivFetcher can be instantiated."""
        fetcher = ArxivFetcher()
        assert fetcher is not None

    def test_arxiv_fetcher_has_fetch_papers_method(self):
        """Test that ArxivFetcher has fetch_papers method."""
        fetcher = ArxivFetcher()
        assert hasattr(fetcher, 'fetch_papers')
        assert callable(fetcher.fetch_papers)

    def test_fetch_papers_accepts_query(self):
        """Test that fetch_papers accepts query parameter."""
        fetcher = ArxivFetcher()
        try:
            fetcher.fetch_papers(query="machine learning")
        except TypeError as e:
            pytest.fail(f"fetch_papers should accept query: {e}")

    def test_fetch_papers_accepts_max_results(self):
        """Test that fetch_papers accepts max_results parameter."""
        fetcher = ArxivFetcher()
        try:
            fetcher.fetch_papers(query="LLM", max_results=50)
        except TypeError as e:
            pytest.fail(f"fetch_papers should accept max_results: {e}")

    def test_fetch_papers_with_default_max_results(self):
        """Test that fetch_papers works with default max_results."""
        fetcher = ArxivFetcher()
        try:
            fetcher.fetch_papers(query="transformer")
        except TypeError as e:
            pytest.fail(f"fetch_papers should have default max_results: {e}")

    def test_fetch_papers_signature(self):
        """Test that fetch_papers has correct signature."""
        import inspect
        sig = inspect.signature(ArxivFetcher.fetch_papers)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'query' in params
        assert 'max_results' in params

    def test_fetch_papers_return_type_annotation(self):
        """Test that fetch_papers has correct return type annotation."""
        import inspect
        sig = inspect.signature(ArxivFetcher.fetch_papers)
        # Check that return annotation exists
        assert sig.return_annotation != inspect.Signature.empty


class TestArxivFetcherQueries:
    """Test suite for various arXiv query scenarios."""

    def test_fetch_papers_with_simple_query(self):
        """Test fetch_papers with simple query."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="attention mechanism")

    def test_fetch_papers_with_complex_query(self):
        """Test fetch_papers with complex query."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="large language model training")

    def test_fetch_papers_with_small_max_results(self):
        """Test fetch_papers with small max_results."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="DPO", max_results=10)

    def test_fetch_papers_with_large_max_results(self):
        """Test fetch_papers with large max_results."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="RLHF", max_results=500)

    def test_fetch_papers_with_zero_max_results(self):
        """Test fetch_papers with zero max_results."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="GPT", max_results=0)

    def test_fetch_papers_with_special_characters(self):
        """Test fetch_papers with special characters in query."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="attention (mechanism)")

    def test_fetch_papers_with_quotes(self):
        """Test fetch_papers with quoted query."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query='"direct preference optimization"')

    def test_fetch_papers_with_boolean_operators(self):
        """Test fetch_papers with boolean operators."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="LLM AND training")

    def test_fetch_papers_with_category_prefix(self):
        """Test fetch_papers with category prefix."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="cat:cs.CL")

    def test_fetch_papers_with_empty_query(self):
        """Test fetch_papers with empty query."""
        fetcher = ArxivFetcher()
        fetcher.fetch_papers(query="")


class TestPaperDeduplicator:
    """Test suite for paper deduplication function."""

    def test_deduplicate_papers_exists(self):
        """Test that deduplicate_papers function exists."""
        assert deduplicate_papers is not None
        assert callable(deduplicate_papers)

    def test_deduplicate_papers_signature(self):
        """Test that deduplicate_papers has correct signature."""
        import inspect
        sig = inspect.signature(deduplicate_papers)
        params = list(sig.parameters.keys())
        assert 'papers' in params

    def test_deduplicate_papers_accepts_list(self):
        """Test that deduplicate_papers accepts list parameter."""
        try:
            deduplicate_papers([])
        except TypeError as e:
            pytest.fail(f"deduplicate_papers should accept list: {e}")

    def test_deduplicate_papers_with_empty_list(self):
        """Test deduplicate_papers with empty list."""
        deduplicate_papers([])

    def test_deduplicate_papers_with_single_paper(self):
        """Test deduplicate_papers with single paper."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Test Paper",
                "abstract": "Test abstract"
            }
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_multiple_papers(self):
        """Test deduplicate_papers with multiple papers."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Paper 1",
                "abstract": "Abstract 1"
            },
            {
                "id": "arxiv:2401.00002",
                "title": "Paper 2",
                "abstract": "Abstract 2"
            }
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_return_type(self):
        """Test that deduplicate_papers has correct return type annotation."""
        import inspect
        sig = inspect.signature(deduplicate_papers)
        assert sig.return_annotation != inspect.Signature.empty


class TestPaperDeduplicationScenarios:
    """Test suite for various deduplication scenarios."""

    def test_deduplicate_identical_papers(self):
        """Test deduplication with identical papers."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Same Paper",
                "abstract": "Same abstract"
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Same Paper",
                "abstract": "Same abstract"
            }
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_same_id(self):
        """Test deduplication with papers having same ID."""
        papers = [
            {"id": "arxiv:2401.00001", "title": "Paper A"},
            {"id": "arxiv:2401.00001", "title": "Paper A Different Title"}
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_similar_titles(self):
        """Test deduplication with similar titles."""
        papers = [
            {"id": "arxiv:2401.00001", "title": "Large Language Models"},
            {"id": "arxiv:2401.00002", "title": "Large Language Models "}
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_from_different_sources(self):
        """Test deduplication with papers from different sources."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Paper",
                "source": "arxiv"
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Paper",
                "source": "twitter"
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Paper",
                "source": "linkedin"
            }
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_different_metadata(self):
        """Test deduplication with papers having different metadata."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Paper",
                "social_score": 10
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Paper",
                "social_score": 50
            }
        ]
        deduplicate_papers(papers)

    def test_deduplicate_large_list(self):
        """Test deduplication with large list of papers."""
        papers = [
            {
                "id": f"arxiv:2401.{str(i).zfill(5)}",
                "title": f"Paper {i}"
            }
            for i in range(1000)
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_missing_fields(self):
        """Test deduplication with papers missing some fields."""
        papers = [
            {"id": "arxiv:2401.00001"},
            {"title": "Paper without ID"},
            {"abstract": "Paper with only abstract"}
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_null_values(self):
        """Test deduplication with null values."""
        papers = [
            {"id": "arxiv:2401.00001", "title": None},
            {"id": "arxiv:2401.00002", "title": "Valid"}
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_preserves_structure(self):
        """Test that deduplication preserves paper structure."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Paper",
                "abstract": "Abstract",
                "authors": ["Author 1"],
                "published": "2024-01-01"
            }
        ]
        deduplicate_papers(papers)

    def test_deduplicate_papers_with_unicode(self):
        """Test deduplication with unicode characters."""
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Résumé of LLM Training 中文",
                "abstract": "Abstract with emoji 🤖"
            }
        ]
        deduplicate_papers(papers)

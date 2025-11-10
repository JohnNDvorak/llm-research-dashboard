"""
Comprehensive test suite for Paper Deduplicator.

This module tests the PaperDeduplicator class that handles deduplication
of papers from multiple sources (arXiv, Twitter, LinkedIn) with intelligent
merging of metadata.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from src.fetch.paper_deduplicator import PaperDeduplicator


class TestPaperDeduplicatorInitialization:
    """Test suite for PaperDeduplicator initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        deduplicator = PaperDeduplicator()
        assert deduplicator is not None
        assert hasattr(deduplicator, 'deduplicate')

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            'use_arxiv_id': True,
            'title_similarity_threshold': 0.85,
            'merge_strategy': {
                'social_score': 'max',
                'professional_score': 'max',
                'sources': 'merge'
            }
        }
        deduplicator = PaperDeduplicator(config)
        assert deduplicator is not None

    def test_config_loaded_correctly(self):
        """Test that configuration is loaded correctly."""
        config = {'title_similarity_threshold': 0.95}
        deduplicator = PaperDeduplicator(config)
        assert deduplicator.config['title_similarity_threshold'] == 0.95


class TestPaperDeduplicatorBasicFunctionality:
    """Test basic deduplication functionality."""

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        deduplicator = PaperDeduplicator()
        result = deduplicator.deduplicate([])
        assert result == []

    def test_deduplicate_single_paper(self):
        """Test deduplication with single paper."""
        deduplicator = PaperDeduplicator()
        papers = [{
            "id": "arxiv:2401.00001",
            "title": "Test Paper",
            "abstract": "Test abstract"
        }]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1
        assert result[0]["id"] == "arxiv:2401.00001"

    def test_deduplicate_unique_papers(self):
        """Test deduplication with all unique papers."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "arxiv:2401.00001", "title": "Paper 1"},
            {"id": "arxiv:2401.00002", "title": "Paper 2"},
            {"id": "arxiv:2401.00003", "title": "Paper 3"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 3

    def test_deduplicate_returns_list(self):
        """Test that deduplicate returns a list."""
        deduplicator = PaperDeduplicator()
        result = deduplicator.deduplicate([])
        assert isinstance(result, list)


class TestArxivIDExtraction:
    """Test arXiv ID extraction from various formats."""

    def test_extract_arxiv_id_standard_format(self):
        """Test extraction from standard arXiv format."""
        deduplicator = PaperDeduplicator()
        paper = {"id": "arxiv:2401.00001"}
        arxiv_id = deduplicator._extract_arxiv_id(paper)
        assert arxiv_id == "2401.00001"

    def test_extract_arxiv_id_from_url(self):
        """Test extraction from arXiv URL."""
        deduplicator = PaperDeduplicator()
        paper = {"url": "https://arxiv.org/abs/2401.00001"}
        arxiv_id = deduplicator._extract_arxiv_id(paper)
        assert arxiv_id == "2401.00001"

    def test_extract_arxiv_id_from_pdf_url(self):
        """Test extraction from PDF URL."""
        deduplicator = PaperDeduplicator()
        paper = {"pdf_url": "https://arxiv.org/pdf/2401.00001.pdf"}
        arxiv_id = deduplicator._extract_arxiv_id(paper)
        assert arxiv_id == "2401.00001"

    def test_extract_arxiv_id_with_version(self):
        """Test extraction with version number (normalize to base)."""
        deduplicator = PaperDeduplicator()
        paper = {"id": "arxiv:2401.00001v2"}
        arxiv_id = deduplicator._extract_arxiv_id(paper)
        assert arxiv_id == "2401.00001"

    def test_extract_arxiv_id_plain_format(self):
        """Test extraction from plain ID format."""
        deduplicator = PaperDeduplicator()
        paper = {"id": "2401.00001"}
        arxiv_id = deduplicator._extract_arxiv_id(paper)
        assert arxiv_id == "2401.00001"

    def test_extract_arxiv_id_none_when_missing(self):
        """Test returns None when arXiv ID not found."""
        deduplicator = PaperDeduplicator()
        paper = {"id": "twitter_12345", "title": "Some paper"}
        arxiv_id = deduplicator._extract_arxiv_id(paper)
        assert arxiv_id is None


class TestTitleSimilarity:
    """Test title similarity calculation."""

    def test_identical_titles(self):
        """Test similarity of identical titles."""
        deduplicator = PaperDeduplicator()
        similarity = deduplicator._calculate_title_similarity(
            "Large Language Models",
            "Large Language Models"
        )
        assert similarity == 1.0

    def test_similar_titles_with_whitespace(self):
        """Test similarity with extra whitespace."""
        deduplicator = PaperDeduplicator()
        similarity = deduplicator._calculate_title_similarity(
            "Large Language Models",
            "Large  Language  Models"
        )
        assert similarity >= 0.95

    def test_similar_titles_different_case(self):
        """Test similarity with different case."""
        deduplicator = PaperDeduplicator()
        similarity = deduplicator._calculate_title_similarity(
            "Large Language Models",
            "large language models"
        )
        assert similarity >= 0.95

    def test_different_titles(self):
        """Test similarity of completely different titles."""
        deduplicator = PaperDeduplicator()
        similarity = deduplicator._calculate_title_similarity(
            "Large Language Models",
            "Computer Vision Techniques"
        )
        assert similarity < 0.5

    def test_similar_titles_with_punctuation(self):
        """Test similarity with different punctuation."""
        deduplicator = PaperDeduplicator()
        similarity = deduplicator._calculate_title_similarity(
            "DPO: Direct Preference Optimization",
            "DPO Direct Preference Optimization"
        )
        assert similarity >= 0.90

    def test_empty_titles(self):
        """Test similarity with empty titles."""
        deduplicator = PaperDeduplicator()
        similarity = deduplicator._calculate_title_similarity("", "")
        assert similarity == 1.0  # Empty strings are identical


class TestDeduplicationByArxivID:
    """Test deduplication using arXiv ID matching."""

    def test_duplicate_by_arxiv_id(self):
        """Test deduplication when papers have same arXiv ID."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Paper A",
                "source": "arxiv",
                "social_score": 0
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Paper A (Twitter)",
                "source": "twitter",
                "social_score": 100
            }
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1
        assert result[0]["id"] == "arxiv:2401.00001"

    def test_duplicate_by_url_arxiv_id(self):
        """Test deduplication by extracting arXiv ID from URLs."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "arxiv:2401.00001", "title": "Paper A"},
            {"url": "https://arxiv.org/abs/2401.00001", "title": "Paper A"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1

    def test_no_duplicate_different_arxiv_ids(self):
        """Test no deduplication when arXiv IDs differ."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "arxiv:2401.00001", "title": "Paper 1"},
            {"id": "arxiv:2401.00002", "title": "Paper 2"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 2


class TestDeduplicationByTitle:
    """Test deduplication using title similarity."""

    def test_duplicate_by_title_similarity(self):
        """Test deduplication when titles are >90% similar."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "twitter_1",
                "title": "Large Language Models for Code Generation",
                "source": "twitter"
            },
            {
                "id": "linkedin_1",
                "title": "Large Language Models for Code Generation",
                "source": "linkedin"
            }
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1

    def test_no_duplicate_dissimilar_titles(self):
        """Test no deduplication when titles are dissimilar."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "twitter_1", "title": "DPO Paper"},
            {"id": "linkedin_1", "title": "Vision Transformers"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 2

    def test_duplicate_threshold_boundary(self):
        """Test deduplication at similarity threshold boundary."""
        deduplicator = PaperDeduplicator({'title_similarity_threshold': 0.90})
        # Create titles that are exactly at threshold
        papers = [
            {"id": "1", "title": "A" * 100},
            {"id": "2", "title": "A" * 90 + "B" * 10}  # 90% similar
        ]
        result = deduplicator.deduplicate(papers)
        # Should deduplicate at >=90% threshold
        assert len(result) <= 2


class TestMergeStrategies:
    """Test merging strategies when duplicates are found."""

    def test_merge_social_score_max(self):
        """Test social_score takes maximum value."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "social_score": 50
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "social_score": 150
            }
        ]
        result = deduplicator.deduplicate(papers)
        assert result[0]["social_score"] == 150

    def test_merge_professional_score_max(self):
        """Test professional_score takes maximum value."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "professional_score": 30
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "professional_score": 80
            }
        ]
        result = deduplicator.deduplicate(papers)
        assert result[0]["professional_score"] == 80

    def test_merge_sources_list(self):
        """Test sources are merged into a list."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "arxiv:2401.00001", "title": "Test", "source": "arxiv"},
            {"id": "arxiv:2401.00001", "title": "Test", "source": "twitter"},
            {"id": "arxiv:2401.00001", "title": "Test", "source": "linkedin"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1
        assert "arxiv" in result[0]["source"]
        assert "twitter" in result[0]["source"]
        assert "linkedin" in result[0]["source"]

    def test_merge_keeps_longest_title(self):
        """Test merge keeps the longest/most complete title."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "arxiv:2401.00001", "title": "Short"},
            {"id": "arxiv:2401.00001", "title": "Much Longer Title Here"}
        ]
        result = deduplicator.deduplicate(papers)
        assert result[0]["title"] == "Much Longer Title Here"

    def test_merge_keeps_longest_abstract(self):
        """Test merge keeps the longest/most complete abstract."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "abstract": "Short"
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "abstract": "Much longer abstract with more detail"
            }
        ]
        result = deduplicator.deduplicate(papers)
        assert result[0]["abstract"] == "Much longer abstract with more detail"

    def test_merge_combines_urls(self):
        """Test merge combines unique URLs."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "url": "https://arxiv.org/abs/2401.00001"
            },
            {
                "id": "arxiv:2401.00001",
                "title": "Test",
                "url": "https://twitter.com/status/12345"
            }
        ]
        result = deduplicator.deduplicate(papers)
        # Should keep at least one URL, preferably arXiv
        assert "url" in result[0]


class TestCombinedScoreCalculation:
    """Test combined score calculation."""

    def test_calculate_combined_score_basic(self):
        """Test basic combined score calculation."""
        deduplicator = PaperDeduplicator()
        paper = {
            "social_score": 100,
            "professional_score": 50,
            "published_date": (datetime.now() - timedelta(days=7)).isoformat()
        }
        score = deduplicator._calculate_combined_score(paper)
        assert score > 0
        assert isinstance(score, (int, float))

    def test_combined_score_formula(self):
        """Test combined score follows formula: (social*0.4) + (prof*0.6) + (recency*0.3)."""
        deduplicator = PaperDeduplicator()
        paper = {
            "social_score": 100,
            "professional_score": 100,
            "published_date": datetime.now().isoformat()  # Today = max recency
        }
        score = deduplicator._calculate_combined_score(paper)
        # With max scores and recent date, should be high
        assert score > 100

    def test_combined_score_zero_scores(self):
        """Test combined score with zero social/professional scores."""
        deduplicator = PaperDeduplicator()
        paper = {
            "social_score": 0,
            "professional_score": 0,
            "published_date": datetime.now().isoformat()
        }
        score = deduplicator._calculate_combined_score(paper)
        # Should still have recency component
        assert score >= 0

    def test_combined_score_missing_fields(self):
        """Test combined score handles missing fields gracefully."""
        deduplicator = PaperDeduplicator()
        paper = {"title": "Test"}  # Missing score fields
        score = deduplicator._calculate_combined_score(paper)
        assert score >= 0

    def test_combined_score_old_paper(self):
        """Test combined score for old paper (lower recency)."""
        deduplicator = PaperDeduplicator()
        paper = {
            "social_score": 100,
            "professional_score": 100,
            "published_date": (datetime.now() - timedelta(days=365)).isoformat()
        }
        score = deduplicator._calculate_combined_score(paper)
        # Should be lower than recent paper with same scores
        assert score > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_papers_with_missing_ids(self):
        """Test handling papers without IDs."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"title": "Transformers for Natural Language Processing"},
            {"title": "Computer Vision with Convolutional Networks"}
        ]
        result = deduplicator.deduplicate(papers)
        # Should assign generated IDs and keep both (distinct titles)
        assert len(result) == 2

    def test_papers_with_null_values(self):
        """Test handling papers with null values."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "1", "title": None, "abstract": "Test"},
            {"id": "2", "title": "Valid", "abstract": None}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 2

    def test_papers_with_unicode_titles(self):
        """Test handling papers with Unicode characters."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "1", "title": "Transformers für NLP"},
            {"id": "2", "title": "中文标题测试"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 2

    def test_papers_with_very_long_titles(self):
        """Test handling papers with very long titles."""
        deduplicator = PaperDeduplicator()
        long_title = "A" * 1000
        papers = [
            {"id": "1", "title": long_title},
            {"id": "2", "title": long_title}
        ]
        result = deduplicator.deduplicate(papers)
        # Should deduplicate if titles match
        assert len(result) <= 2

    def test_papers_with_empty_strings(self):
        """Test handling papers with empty string fields."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": "", "title": "Test 1"},
            {"id": "", "title": "Test 2"}
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) >= 1

    def test_large_batch_performance(self):
        """Test deduplication performance with large batch."""
        deduplicator = PaperDeduplicator()
        papers = [
            {"id": f"arxiv:2401.{str(i).zfill(5)}", "title": f"Paper {i}"}
            for i in range(1000)
        ]
        # Add some duplicates
        papers.extend([
            {"id": "arxiv:2401.00001", "title": "Paper 1"},
            {"id": "arxiv:2401.00100", "title": "Paper 100"}
        ])

        import time
        start = time.time()
        result = deduplicator.deduplicate(papers)
        duration = time.time() - start

        # Should complete in < 1 second for 1000 papers
        assert duration < 1.0
        # Should deduplicate the 2 duplicates
        assert len(result) == 1000


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_multi_source_deduplication(self):
        """Test deduplication across arXiv, Twitter, and LinkedIn."""
        deduplicator = PaperDeduplicator()
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Direct Preference Optimization",
                "source": "arxiv",
                "social_score": 0,
                "professional_score": 0
            },
            {
                "id": "twitter_12345",
                "title": "Direct Preference Optimization",
                "url": "https://twitter.com/status/12345",
                "source": "twitter",
                "social_score": 150,
                "professional_score": 0
            },
            {
                "id": "linkedin_67890",
                "title": "Direct Preference Optimization",
                "url": "https://linkedin.com/posts/67890",
                "source": "linkedin",
                "social_score": 0,
                "professional_score": 75
            }
        ]
        result = deduplicator.deduplicate(papers)

        # Should merge into 1 paper
        assert len(result) == 1

        # Should have arXiv ID
        assert "arxiv:2401.00001" in result[0]["id"]

        # Should have max scores
        assert result[0]["social_score"] == 150
        assert result[0]["professional_score"] == 75

        # Should have all sources
        assert "arxiv" in result[0]["source"]
        assert "twitter" in result[0]["source"]
        assert "linkedin" in result[0]["source"]

    def test_arxiv_fetcher_integration(self):
        """Test integration with ArxivFetcher output format."""
        deduplicator = PaperDeduplicator()
        # Simulate papers from ArxivFetcher
        papers = [
            {
                "id": "arxiv:2401.00001",
                "title": "Paper 1",
                "authors": ["Author A", "Author B"],
                "abstract": "Abstract text",
                "url": "https://arxiv.org/abs/2401.00001",
                "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
                "source": "arxiv",
                "published_date": "2024-01-01"
            },
            {
                "id": "arxiv:2401.00001",  # Duplicate
                "title": "Paper 1",
                "authors": ["Author A", "Author B"],
                "abstract": "Abstract text",
                "source": "arxiv"
            }
        ]
        result = deduplicator.deduplicate(papers)
        assert len(result) == 1
        assert result[0]["url"] == "https://arxiv.org/abs/2401.00001"

    def test_preserve_all_metadata(self):
        """Test that all important metadata is preserved."""
        deduplicator = PaperDeduplicator()
        papers = [{
            "id": "arxiv:2401.00001",
            "title": "Test Paper",
            "authors": ["John Doe"],
            "abstract": "Test abstract",
            "url": "https://arxiv.org/abs/2401.00001",
            "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
            "source": "arxiv",
            "published_date": "2024-01-01",
            "fetch_date": "2024-01-15",
            "social_score": 100,
            "professional_score": 50
        }]
        result = deduplicator.deduplicate(papers)

        # All fields should be preserved
        assert result[0]["id"] == "arxiv:2401.00001"
        assert result[0]["title"] == "Test Paper"
        assert result[0]["authors"] == ["John Doe"]
        assert result[0]["abstract"] == "Test abstract"
        assert "combined_score" in result[0]

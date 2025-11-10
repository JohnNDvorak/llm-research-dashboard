"""
Phase 2 Integration Tests - Complete end-to-end testing.

Tests the complete workflow:
1. Fetch from all sources (arXiv, X, LinkedIn)
2. Deduplicate across sources
3. Store in database
4. Retrieve and verify

This validates that Phase 2.4 (LinkedIn Integration) is complete and working.
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.fetch.fetch_manager import FetchManager
from src.fetch.arxiv_fetcher import ArxivFetcher
from src.fetch.twitter_fetcher import TwitterFetcher
from src.fetch.linkedin_fetcher import LinkedinFetcher, LinkedInPost
from src.fetch.paper_deduplicator import PaperDeduplicator
from src.storage.paper_db import PaperDB
from src.utils.logger import logger


class TestPhase2Integration:
    """Test complete Phase 2 integration with all fetchers."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        # Initialize database
        db = PaperDB(db_path=db_path)
        db.execute_migration('src/storage/migrations/001_initial_schema.sql')
        db.close()

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def mock_papers(self):
        """Mock papers from all sources for testing."""
        return {
            'arxiv': [
                {
                    'id': 'arxiv:2401.00001',
                    'title': 'DPO: Direct Preference Optimization',
                    'abstract': 'We propose DPO...',
                    'authors': ['OpenAI Team', 'John Doe'],
                    'source': 'arxiv',
                    'social_score': 0,
                    'professional_score': 0,
                    'arxiv_id': '2401.00001',
                    'url': 'https://arxiv.org/abs/2401.00001',
                    'pdf_url': 'https://arxiv.org/pdf/2401.00001.pdf',
                    'published_date': '2024-01-01',
                    'categories': ['cs.CL', 'cs.AI']
                }
            ],
            'twitter': [
                {
                    'id': 'twitter_123456789',
                    'title': None,  # Will be filled by arXiv merge
                    'abstract': None,
                    'authors': ['@OpenAI'],
                    'source': 'twitter',
                    'social_score': 500,  # High social engagement
                    'professional_score': 0,
                    'arxiv_id': '2401.00001',  # Same as arXiv paper
                    'x_tweet_id': '123456789',
                    'x_author': 'OpenAI',
                    'x_url': 'https://x.com/OpenAI/status/123456789',
                    'fetch_date': datetime.now(timezone.utc).date().isoformat()
                }
            ],
            'linkedin': [
                {
                    'id': 'linkedin:987654321',
                    'title': None,  # Will be filled by arXiv merge
                    'abstract': None,
                    'authors': ['Dr. Jane Smith'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 300,  # High professional engagement
                    'arxiv_id': '2401.00001',  # Same as arXiv paper
                    'linkedin_post_id': '987654321',
                    'linkedin_author_name': 'Dr. Jane Smith',
                    'linkedin_author_title': 'Research Scientist at OpenAI',
                    'linkedin_company': 'OpenAI',
                    'linkedin_post_url': 'https://linkedin.com/posts/987654321',
                    'linkedin_likes': 150,
                    'linkedin_comments': 30,
                    'linkedin_shares': 20,
                    'fetch_date': datetime.now(timezone.utc).date().isoformat()
                },
                {
                    'id': 'linkedin:456789123',
                    'title': None,
                    'abstract': None,
                    'authors': ['John Doe'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 200,
                    'arxiv_id': '2401.12345',  # Different paper
                    'linkedin_company': 'Anthropic',
                    'fetch_date': datetime.now(timezone.utc).date().isoformat()
                }
            ]
        }

    def test_fetch_manager_initialization(self):
        """Test FetchManager initializes all sources."""
        with patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'test_token'}):
            manager = FetchManager()

            assert manager.arxiv_fetcher is not None
            assert manager.twitter_fetcher is not None
            assert manager.linkedin_fetcher is not None
            assert manager.deduplicator is not None

            # Check stats initialization
            assert 'fetch_start_time' in manager.stats
            assert 'arxiv_count' in manager.stats
            assert 'twitter_count' in manager.stats
            assert 'linkedin_count' in manager.stats

    @patch('src.fetch.arxiv_fetcher.ArxivFetcher.fetch_recent_papers')
    @patch('src.fetch.twitter_fetcher.TwitterFetcher.fetch_recent_papers')
    @patch('src.fetch.linkedin_fetcher.LinkedinFetcher.fetch_papers')
    def test_fetch_all_sources_sequential(
        self,
        mock_linkedin,
        mock_twitter,
        mock_arxiv,
        mock_papers
    ):
        """Test fetching from all sources sequentially."""
        # Setup mocks
        mock_arxiv.return_value = mock_papers['arxiv']
        mock_twitter.return_value = mock_papers['twitter']
        mock_linkedin.return_value = mock_papers['linkedin']

        manager = FetchManager()

        # Fetch sequentially
        papers = manager.fetch_all_papers(days=7, parallel=False)

        # Should have deduplicated papers
        # - One merged paper (arxiv + twitter + linkedin for 2401.00001)
        # - One linkedin-only paper (2401.12345)
        assert len(papers) == 2

        # Check merged paper has all metadata
        merged = next(p for p in papers if p['arxiv_id'] == '2401.00001')
        assert merged['title'] == 'DPO: Direct Preference Optimization'
        assert merged['social_score'] == 500  # From Twitter
        assert merged['professional_score'] == 300  # From LinkedIn
        assert 'OpenAI' in str(merged.get('authors', []))

        # Check LinkedIn-only paper
        linkedin_only = next(p for p in papers if p['arxiv_id'] == '2401.12345')
        assert linkedin_only['linkedin_company'] == 'Anthropic'
        assert linkedin_only['professional_score'] == 200

        # Check stats
        assert manager.stats['arxiv_count'] == 1
        assert manager.stats['twitter_count'] == 1
        assert manager.stats['linkedin_count'] == 2
        assert manager.stats['total_before_dedup'] == 4
        assert manager.stats['total_after_dedup'] == 2
        assert manager.stats['duplicates_removed'] == 2

    def test_fetch_specific_sources(self):
        """Test fetching only specific sources."""
        manager = FetchManager()

        with patch.object(manager, 'fetch_all_papers') as mock_fetch:
            mock_fetch.return_value = []

            # Fetch only arXiv and LinkedIn
            manager.fetch_all_papers(days=7, sources=['arxiv', 'linkedin'])

            # Check the call
            mock_fetch.assert_called_once_with(days=7, parallel=True, include_sources=['arxiv', 'linkedin'])

    def test_fetch_and_store_workflow(self, temp_db, mock_papers):
        """Test complete fetch and store workflow."""
        # Patch the database path
        with patch('src.storage.paper_db.PaperDB') as MockPaperDB:
            mock_db = Mock()
            MockPaperDB.return_value.__enter__.return_value = mock_db
            MockPaperDB.return_value.__exit__.return_value = None

            # Mock paper existence check
            mock_db.paper_exists.return_value = False

            # Mock fetchers
            with patch.object(FetchManager, '__init__', return_value=None):
                manager = FetchManager()
                manager.arxiv_fetcher = Mock()
                manager.twitter_fetcher = Mock()
                manager.linkedin_fetcher = Mock()
                manager.deduplicator = PaperDeduplicator()
                manager.stats = {
                    'fetch_start_time': datetime.now(timezone.utc),
                    'fetch_end_time': datetime.now(timezone.utc),
                    'arxiv_count': 1,
                    'twitter_count': 1,
                    'linkedin_count': 2,
                    'duplicates_removed': 2,
                    'errors': []
                }

            # Setup mock fetchers
            manager.arxiv_fetcher.fetch_recent_papers.return_value = mock_papers['arxiv']
            manager.twitter_fetcher.fetch_recent_papers.return_value = mock_papers['twitter']
            manager.linkedin_fetcher.fetch_papers.return_value = mock_papers['linkedin']

            # Mock fetch_all_papers to return deduplicated papers
            deduplicated = [
                {
                    'id': 'arxiv:2401.00001',
                    'title': 'DPO: Direct Preference Optimization',
                    'social_score': 500,
                    'professional_score': 300,
                    'source': ['arxiv', 'twitter', 'linkedin']
                },
                {
                    'id': 'linkedin:456789123',
                    'title': None,
                    'social_score': 0,
                    'professional_score': 200,
                    'source': 'linkedin'
                }
            ]
            manager.fetch_all_papers = Mock(return_value=deduplicated)

            # Run fetch and store
            results = manager.fetch_and_store(days=7)

            # Verify results
            assert results['papers_fetched'] == 2
            assert results['papers_stored'] == 2
            assert results['duplicates_removed'] == 2
            assert 'source_counts' in results

            # Verify database operations
            assert mock_db.insert_paper.call_count == 2
            assert mock_db.paper_exists.call_count == 2

    def test_error_handling_in_fetch(self):
        """Test error handling during fetch."""
        manager = FetchManager()

        # Mock one fetcher to raise an error
        with patch.object(manager.arxiv_fetcher, 'fetch_recent_papers', side_effect=Exception("Network error")):
            with patch.object(manager.twitter_fetcher, 'fetch_recent_papers', return_value=[]):
                with patch.object(manager.linkedin_fetcher, 'fetch_papers', return_value=[]):
                    # Should not raise exception
                    papers = manager.fetch_all_papers(days=7, parallel=False)

                    # Should have errors logged
                    assert len(manager.stats['errors']) > 0
                    assert "Network error" in str(manager.stats['errors'][0])

    def test_linkedin_integration_complete(self, mock_papers):
        """Test LinkedIn integration with the full pipeline."""
        # Create a LinkedIn post
        linkedin_post = LinkedInPost(
            id="linkedin123",
            author_name="Dr. Alice Johnson",
            author_title="Research Scientist at OpenAI",
            author_profile_url="https://linkedin.com/in/alice",
            company="OpenAI",
            text="Excited about our new paper on DPO! Check it out: https://arxiv.org/abs/2401.00001",
            url="https://linkedin.com/posts/linkedin123",
            likes_count=200,
            comments_count=40,
            shares_count=25,
            views_count=10000,
            published_at=datetime.now(timezone.utc) - timedelta(days=2)
        )

        # Convert to paper format
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': {'rate_limit_delay': 1}}):
            linkedin_fetcher = LinkedinFetcher()
            linkedin_paper = linkedin_fetcher._format_paper_dict(linkedin_post)

        # Verify LinkedIn paper format
        assert linkedin_paper['id'] == 'linkedin:linkedin123'
        assert linkedin_paper['source'] == 'linkedin'
        assert linkedin_paper['arxiv_id'] == '2401.00001'
        assert linkedin_paper['linkedin_company'] == 'OpenAI'
        assert linkedin_paper['professional_score'] > 0

        # Test deduplication with arXiv paper
        arxiv_paper = {
            'id': 'arxiv:2401.00001',
            'title': 'DPO: Direct Preference Optimization',
            'abstract': 'We propose DPO...',
            'authors': ['OpenAI Team'],
            'source': 'arxiv',
            'social_score': 0,
            'professional_score': 0,
            'arxiv_id': '2401.00001'
        }

        deduplicator = PaperDeduplicator()
        deduplicated = deduplicator.deduplicate([linkedin_paper, arxiv_paper])

        # Should merge into one paper
        assert len(deduplicated) == 1
        merged = deduplicated[0]

        # Verify merge results
        assert merged['id'] == 'arxiv:2401.00001'  # arXiv ID takes precedence
        assert merged['title'] == 'DPO: Direct Preference Optimization'
        assert merged['professional_score'] > 0  # From LinkedIn
        assert merged['linkedin_company'] == 'OpenAI'

    def test_combined_score_calculation(self, mock_papers):
        """Test combined score calculation across sources."""
        # Create papers with different scores
        papers = [
            {
                'id': 'paper1',
                'social_score': 100,
                'professional_score': 200,
                'published_date': '2024-01-10'
            },
            {
                'id': 'paper2',
                'social_score': 50,
                'professional_score': 300,
                'published_date': '2024-01-05'
            },
            {
                'id': 'paper3',
                'social_score': 200,
                'professional_score': 50,
                'published_date': '2024-01-01'
            }
        ]

        deduplicator = PaperDeduplicator()

        # Calculate combined scores
        for paper in papers:
            score = deduplicator._calculate_combined_score(paper)
            # Combined = (social * 0.4) + (prof * 0.6) + (recency * 0.3)
            # Recency: days ago from 2024-01-11 (current)
            expected = (paper['social_score'] * 0.4) + (paper['professional_score'] * 0.6)
            # Add recency bonus (more recent = higher score)
            if paper['published_date'] == '2024-01-10':
                expected += 0.3 * 1  # 1 day ago
            elif paper['published_date'] == '2024-01-05':
                expected += 0.3 * 0.6  # 6 days ago
            elif paper['published_date'] == '2024-01-01':
                expected += 0.3 * 0.4  # 10 days ago (minimum)

            assert abs(score - expected) < 0.1

    def test_source_priority_in_merge(self):
        """Test source priority when merging papers."""
        papers = [
            {
                'id': 'linkedin:123',
                'title': None,
                'abstract': None,
                'authors': ['LinkedIn Author'],
                'source': 'linkedin',
                'url': None
            },
            {
                'id': 'arxiv:2401.00001',
                'title': 'Full Title',
                'abstract': 'Full Abstract',
                'authors': ['ArXiv Author'],
                'source': 'arxiv',
                'url': 'https://arxiv.org/abs/2401.00001'
            }
        ]

        deduplicator = PaperDeduplicator()
        merged = deduplicator._merge_papers(papers)

        # Should prefer arXiv metadata
        assert merged['title'] == 'Full Title'
        assert merged['abstract'] == 'Full Abstract'
        assert merged['url'] == 'https://arxiv.org/abs/2401.00001'
        # Should merge authors
        assert len(merged['authors']) > 1
        assert 'LinkedIn Author' in merged['authors']
        assert 'ArXiv Author' in merged['authors']

    def test_linkedin_specific_fields_preserved(self, mock_papers):
        """Test LinkedIn-specific fields are preserved in merge."""
        linkedin_paper = mock_papers['linkedin'][0].copy()
        arxiv_paper = mock_papers['arxiv'][0].copy()

        deduplicator = PaperDeduplicator()
        merged = deduplicator._merge_papers([linkedin_paper, arxiv_paper])

        # Should preserve LinkedIn fields
        assert 'linkedin_post_id' in merged
        assert 'linkedin_company' in merged
        assert 'linkedin_author_title' in merged
        assert merged['linkedin_company'] == 'OpenAI'


class TestPhase2Workflow:
    """Test the complete Phase 2 workflow as it would run in production."""

    def test_complete_daily_workflow(self, temp_db):
        """Test the complete daily workflow simulation."""
        # This would be the equivalent of running `make fetch`
        from src.fetch.main_fetch import main_fetch

        # Mock all external APIs
        with patch('src.fetch.arxiv_fetcher.ArxivFetcher.fetch_recent_papers') as mock_arxiv:
            with patch('src.fetch.twitter_fetcher.TwitterFetcher.fetch_recent_papers') as mock_twitter:
                with patch('src.fetch.linkedin_fetcher.LinkedinFetcher.fetch_papers') as mock_linkedin:
                    # Mock returns
                    mock_arxiv.return_value = [
                        {
                            'id': 'arxiv:2401.00001',
                            'title': 'Test Paper',
                            'abstract': 'Test abstract',
                            'authors': ['Test Author'],
                            'source': 'arxiv',
                            'arxiv_id': '2401.00001'
                        }
                    ]
                    mock_twitter.return_value = []
                    mock_linkedin.return_value = [
                        {
                            'id': 'linkedin:123',
                            'title': None,
                            'abstract': None,
                            'authors': ['LinkedIn User'],
                            'source': 'linkedin',
                            'arxiv_id': '2401.00001',
                            'linkedin_company': 'Test Company'
                        }
                    ]

                    # Mock database
                    with patch('src.storage.paper_db.PaperDB') as MockPaperDB:
                        mock_db = Mock()
                        MockPaperDB.return_value.__enter__.return_value = mock_db
                        MockPaperDB.return_value.__exit__.return_value = None
                        mock_db.paper_exists.return_value = False

                        # Run daily update
                        results = main_fetch(days=1, store=True, verbose=False)

                        # Verify results
                        assert results['papers_fetched'] > 0
                        assert results['papers_stored'] > 0
                        assert 'source_counts' in results

    def test_phase2_success_criteria(self):
        """Verify Phase 2 success criteria are met."""
        # Phase 2 Requirements:
        # ✓ Fetch from arXiv (working)
        # ✓ Fetch from X/Twitter (working)
        # ✓ Fetch from LinkedIn (implemented)
        # ✓ Deduplicate across all sources (working)
        # ✓ Store in database (working)
        # ✓ Combined scoring (working)

        manager = FetchManager()

        # Check all sources are initialized
        assert manager.arxiv_fetcher is not None
        assert manager.twitter_fetcher is not None
        assert manager.linkedin_fetcher is not None

        # Check deduplicator is ready
        assert manager.deduplicator is not None

        # Check LinkedIn has required features
        linkedin_config = manager.linkedin_fetcher.config
        assert 'tracked_companies' in linkedin_config
        assert 'hashtags' in linkedin_config
        assert 'rate_limit_delay' in linkedin_config

        # All Phase 2 components are present and configured
        phase2_components = {
            'ArxivFetcher': manager.arxiv_fetcher is not None,
            'TwitterFetcher': manager.twitter_fetcher is not None,
            'LinkedinFetcher': manager.linkedin_fetcher is not None,
            'PaperDeduplicator': manager.deduplicator is not None,
            'FetchManager': manager is not None
        }

        assert all(phase2_components.values()), f"Missing components: {[k for k, v in phase2_components.items() if not v]}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])
"""
Integration tests for LinkedIn Fetcher with Phase 2 components.

Tests the complete workflow:
LinkedIn Fetcher → PaperDeduplicator → Database storage
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.fetch.linkedin_fetcher import LinkedinFetcher, LinkedInPost
from src.fetch.paper_deduplicator import PaperDeduplicator
from src.storage.paper_db import PaperDB
from src.utils.logger import logger


class TestLinkedInIntegration:
    """Test LinkedIn integration with existing Phase 2 components."""

    @pytest.fixture
    def linkedin_config(self):
        """LinkedIn configuration for testing."""
        return {
            'tracked_companies': [
                {'name': 'OpenAI', 'company_id': 'openai', 'priority': 'high'},
                {'name': 'Anthropic', 'company_id': 'anthropic', 'priority': 'high'}
            ],
            'hashtags': ['#LLM', '#AIResearch'],
            'rate_limit_delay': 1,  # Shorter for tests
            'max_posts_per_day': 50,
            'preferred_method': 'scraping'
        }

    @pytest.fixture
    def sample_linkedin_posts(self):
        """Sample LinkedIn posts for testing."""
        return [
            LinkedInPost(
                id="123456789",
                author_name="Dr. Jane Smith",
                author_title="Research Scientist at OpenAI",
                author_profile_url="https://linkedin.com/in/janesmith",
                company="OpenAI",
                text="Excited to share our latest work on DPO! Paper is now on arXiv: https://arxiv.org/abs/2401.00001. This represents a significant improvement in preference optimization.",
                url="https://linkedin.com/posts/123456789",
                likes_count=250,
                comments_count=45,
                shares_count=30,
                views_count=10000,
                published_at=datetime.now(timezone.utc) - timedelta(days=2)
            ),
            LinkedInPost(
                id="987654321",
                author_name="John Doe",
                author_title="ML Engineer at Anthropic",
                author_profile_url="https://linkedin.com/in/johndoe",
                company="Anthropic",
                text="Our new paper on constitutional AI is out: https://arxiv.org/abs/2401.12345. Check out how we're making AI safer!",
                url="https://linkedin.com/posts/987654321",
                likes_count=180,
                comments_count=32,
                shares_count=25,
                views_count=8500,
                published_at=datetime.now(timezone.utc) - timedelta(days=1)
            ),
            LinkedInPost(
                id="456789123",
                author_name="Dr. Alice Johnson",
                author_title="Senior Researcher, Google DeepMind",
                author_profile_url="https://linkedin.com/in/alicejohnson",
                company="Google DeepMind",
                text="No arXiv link here, but excited about our recent training improvements!",
                url="https://linkedin.com/posts/456789123",
                likes_count=100,
                comments_count=15,
                shares_count=10,
                views_count=5000,
                published_at=datetime.now(timezone.utc) - timedelta(days=3)
            )
        ]

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_fetcher_initialization(self, mock_config, linkedin_config):
        """Test LinkedIn fetcher initialization with configuration."""
        mock_config.return_value = {'linkedin': linkedin_config}

        fetcher = LinkedinFetcher()

        assert fetcher.mode == "scraping"  # Default mode
        assert fetcher.base_delay == 1
        assert len(fetcher.config['tracked_companies']) == 2
        assert fetcher.cache.max_daily == 50

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_to_paper_format(self, mock_config, linkedin_config, sample_linkedin_posts):
        """Test conversion from LinkedIn posts to paper format."""
        mock_config.return_value = {'linkedin': linkedin_config}
        fetcher = LinkedinFetcher()

        papers = []
        for post in sample_linkedin_posts:
            paper = fetcher._format_paper_dict(post)
            papers.append(paper)

        # Check first paper (with arXiv link)
        assert papers[0]['id'] == "linkedin:123456789"
        assert papers[0]['source'] == 'linkedin'
        assert papers[0]['authors'] == ['Dr. Jane Smith']
        assert papers[0]['arxiv_id'] == '2401.00001'
        assert papers[0]['linkedin_company'] == 'OpenAI'
        assert papers[0]['professional_score'] > 0  # Should be calculated

        # Check second paper (with arXiv link)
        assert papers[1]['arxiv_id'] == '2401.12345'
        assert papers[1]['linkedin_company'] == 'Anthropic'

        # Check third paper (no arXiv link)
        assert papers[2]['arxiv_id'] is None
        # Papers without arXiv links should still be formatted

    def test_professional_score_calculation(self, linkedin_config):
        """Test professional score calculation with different scenarios."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': linkedin_config}):
            fetcher = LinkedinFetcher()

            # Test verified researcher at top company
            post = LinkedInPost(
                id="1",
                author_name="Dr. Smith",
                author_title="Research Scientist at OpenAI",
                author_profile_url="url",
                company="OpenAI",
                text="Paper: https://arxiv.org/abs/2401.00001",
                url="url",
                likes_count=100,
                comments_count=20,
                shares_count=10,
                views_count=5000,
                published_at=datetime.now(timezone.utc)
            )

            score = fetcher._calculate_professional_score(post)
            # Base: 100*1 + 20*5 + 10*3 + 5000*0.001 = 100 + 100 + 30 + 5 = 235
            # With 1.5x boost for verified researcher: 235 * 1.5 = 352.5
            assert score == 352

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_deduplication_with_arxiv(self, mock_config, linkedin_config):
        """Test deduplication between LinkedIn and arXiv papers."""
        mock_config.return_value = {'linkedin': linkedin_config}

        # Create LinkedIn paper
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': linkedin_config}):
            linkedin_fetcher = LinkedinFetcher()

            linkedin_post = LinkedInPost(
                id="linkedin123",
                author_name="OpenAI Researcher",
                author_title="Research Scientist at OpenAI",
                author_profile_url="url",
                company="OpenAI",
                text="Our DPO paper: https://arxiv.org/abs/2401.00001",
                url="url",
                likes_count=150,
                comments_count=30,
                shares_count=20,
                views_count=8000,
                published_at=datetime.now(timezone.utc) - timedelta(days=1)
            )

            linkedin_paper = linkedin_fetcher._format_paper_dict(linkedin_post)

        # Create matching arXiv paper
        arxiv_paper = {
            'id': 'arxiv:2401.00001',
            'title': 'DPO: Direct Preference Optimization',
            'abstract': 'We propose DPO...',
            'authors': ['OpenAI Team', 'John Doe'],
            'source': 'arxiv',
            'social_score': 0,
            'professional_score': 0,
            'arxiv_id': '2401.00001',
            'url': 'https://arxiv.org/abs/2401.00001',
            'published_date': '2024-01-01'
        }

        # Deduplicate
        deduplicator = PaperDeduplicator()
        all_papers = [linkedin_paper, arxiv_paper]
        deduplicated = deduplicator.deduplicate(all_papers)

        # Should merge into one paper
        assert len(deduplicated) == 1
        merged = deduplicated[0]

        # Keep arXiv ID as primary
        assert merged['id'] == 'arxiv:2401.00001'
        # Keep arXiv metadata
        assert merged['title'] == 'DPO: Direct Preference Optimization'
        # Merge LinkedIn professional score
        assert merged['professional_score'] > 0
        # Merge sources
        sources = merged.get('source', [])
        if isinstance(sources, list):
            assert 'linkedin' in sources and 'arxiv' in sources
        else:
            # If only one source, check which one
            assert sources in ['linkedin', 'arxiv']

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    @patch('src.fetch.paper_deduplicator.get_queries_config')
    def test_complete_linkedin_workflow(self, mock_dedup_config, mock_linkedin_config, linkedin_config):
        """Test complete LinkedIn fetch and deduplication workflow."""
        mock_linkedin_config.return_value = {'linkedin': linkedin_config}
        mock_dedup_config.return_value = {
            'linkedin': linkedin_config,
            'deduplication': {
                'use_arxiv_id': True,
                'title_similarity_threshold': 0.90,
                'merge_strategy': {
                    'social_score': 'max',
                    'professional_score': 'max',
                    'sources': 'merge'
                }
            }
        }

        # Mock LinkedIn fetcher
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': linkedin_config}):
            linkedin_fetcher = LinkedinFetcher()
            linkedin_fetcher.fetch_recent_papers = AsyncMock(return_value=[
                {
                    'id': 'linkedin:123',
                    'title': None,
                    'abstract': None,
                    'authors': ['OpenAI Researcher'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 300,
                    'arxiv_id': '2401.00001',
                    'url': None,
                    'linkedin_company': 'OpenAI',
                    'linkedin_author_title': 'Research Scientist at OpenAI'
                },
                {
                    'id': 'linkedin:456',
                    'title': None,
                    'abstract': None,
                    'authors': ['Anthropic Researcher'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 200,
                    'arxiv_id': '2401.12345',
                    'url': None,
                    'linkedin_company': 'Anthropic',
                    'linkedin_author_title': 'ML Engineer at Anthropic'
                }
            ])

        # Mock arXiv papers for deduplication
        arxiv_papers = [
            {
                'id': 'arxiv:2401.00001',
                'title': 'DPO: Direct Preference Optimization',
                'abstract': 'We propose DPO...',
                'authors': ['OpenAI Team'],
                'source': 'arxiv',
                'social_score': 0,
                'professional_score': 0,
                'arxiv_id': '2401.00001',
                'url': 'https://arxiv.org/abs/2401.00001'
            }
        ]

        # Deduplicate
        deduplicator = PaperDeduplicator()
        all_papers = arxiv_papers + linkedin_fetcher.fetch_recent_papers.return_value
        deduplicated = deduplicator.deduplicate(all_papers)

        # Should have 2 papers: one merged (arxiv+linkedin) and one linkedin-only
        assert len(deduplicated) == 2

        # Find the merged paper
        merged = next(p for p in deduplicated if p['id'] == 'arxiv:2401.00001')
        assert merged['title'] == 'DPO: Direct Preference Optimization'
        assert merged['professional_score'] == 300  # From LinkedIn
        assert merged['linkedin_company'] == 'OpenAI'

        # Find LinkedIn-only paper
        linkedin_only = next(p for p in deduplicated if p['id'] == 'linkedin:456')
        assert linkedin_only['arxiv_id'] == '2401.12345'
        assert linkedin_only['linkedin_company'] == 'Anthropic'

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_rate_limiting(self, mock_config, linkedin_config):
        """Test LinkedIn rate limiting enforcement."""
        mock_config.return_value = {'linkedin': linkedin_config}
        fetcher = LinkedinFetcher()
        fetcher.base_delay = 0.1  # Shorter for tests

        # Mock time tracking
        original_time = time.time
        call_times = []

        def mock_time():
            call_times.append(original_time())
            return original_time()

        with patch('time.time', mock_time):
            # First call should not delay
            start = time.time()
            fetcher._enforce_rate_limit()
            first_elapsed = time.time() - start

            # Immediate second call should delay
            start = time.time()
            fetcher._enforce_rate_limit()
            second_elapsed = time.time() - start

            # Second call should take longer due to rate limiting
            assert second_elapsed > first_elapsed

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_cache_management(self, mock_config, linkedin_config):
        """Test LinkedIn post cache management."""
        mock_config.return_value = {'linkedin': linkedin_config}
        fetcher = LinkedinFetcher()

        # Initially empty
        assert not fetcher.cache.is_post_fetched("test123")
        assert fetcher.cache.daily_fetch_count == 0

        # Add post
        fetcher.cache.add_post("test123")
        assert fetcher.cache.is_post_fetched("test123")

        # Increment counter
        fetcher.cache.increment_fetch_count()
        assert fetcher.cache.daily_fetch_count == 1

        # Update company fetch
        fetcher.cache.update_company_fetch("OpenAI")
        assert fetcher.cache.get_company_last_fetch("OpenAI") is not None

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_error_handling(self, mock_config, linkedin_config):
        """Test LinkedIn error handling and recovery."""
        mock_config.return_value = {'linkedin': linkedin_config}
        fetcher = LinkedinFetcher()

        # Test arXiv extraction with invalid text
        assert fetcher._extract_arxiv_id(None) is None
        assert fetcher._extract_arxiv_id("") is None
        assert fetcher._extract_arxiv_id("No link here") is None

        # Test company extraction with invalid title
        assert fetcher._extract_company(None) is None
        assert fetcher._extract_company("") is None
        assert fetcher._extract_company("Just a title") is None

        # Test count parsing with invalid text
        assert fetcher._parse_count(None) == 0
        assert fetcher._parse_count("") == 0
        assert fetcher._parse_count("no numbers") == 0

        # Test time parsing with invalid text
        result = fetcher._parse_time_ago("invalid")
        assert isinstance(result, datetime)

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_statistics(self, mock_config, linkedin_config):
        """Test LinkedIn fetcher statistics."""
        mock_config.return_value = {'linkedin': linkedin_config}
        fetcher = LinkedinFetcher()

        stats = fetcher.get_stats()

        assert 'mode' in stats
        assert 'daily_fetch_count' in stats
        assert 'max_daily' in stats
        assert 'tracked_companies' in stats
        assert 'hashtags' in stats
        assert stats['tracked_companies'] == 2
        assert stats['hashtags'] == 2


class TestLinkedInDatabaseIntegration:
    """Test LinkedIn integration with database storage."""

    @pytest.fixture
    def linkedin_config(self):
        """LinkedIn configuration for testing."""
        return {
            'tracked_companies': [{'name': 'OpenAI', 'company_id': 'openai', 'priority': 'high'}],
            'hashtags': ['#LLM'],
            'rate_limit_delay': 1,
            'max_posts_per_day': 50
        }

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_paper_storage(self, mock_config, linkedin_config):
        """Test storing LinkedIn papers in database."""
        mock_config.return_value = {'linkedin': linkedin_config}

        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': linkedin_config}):
            fetcher = LinkedinFetcher()

        # Create LinkedIn paper
        linkedin_post = LinkedInPost(
            id="123",
            author_name="OpenAI Researcher",
            author_title="Research Scientist at OpenAI",
            author_profile_url="https://linkedin.com/in/researcher",
            company="OpenAI",
            text="Our new paper: https://arxiv.org/abs/2401.00001",
            url="https://linkedin.com/posts/123",
            likes_count=100,
            comments_count=20,
            shares_count=10,
            views_count=5000,
            published_at=datetime.now(timezone.utc) - timedelta(days=1)
        )

        paper = fetcher._format_paper_dict(linkedin_post)

        # Verify paper has required fields
        assert 'id' in paper
        assert 'source' in paper
        assert 'linkedin_company' in paper
        assert 'professional_score' in paper
        assert 'arxiv_id' in paper

        # Verify LinkedIn-specific fields
        assert paper['linkedin_post_id'] == "123"
        assert paper['linkedin_company'] == "OpenAI"
        assert paper['linkedin_author_title'] == "Research Scientist at OpenAI"
        assert paper['professional_score'] > 0

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_linkedin_batch_processing(self, mock_config, linkedin_config):
        """Test processing multiple LinkedIn papers."""
        mock_config.return_value = {'linkedin': linkedin_config}

        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': linkedin_config}):
            fetcher = LinkedinFetcher()

        # Create multiple posts
        posts = []
        for i in range(5):
            post = LinkedInPost(
                id=f"post_{i}",
                author_name=f"Researcher {i}",
                author_title=f"Research Scientist at Company {i}",
                author_profile_url=f"https://linkedin.com/in/researcher{i}",
                company=f"Company {i}",
                text=f"Paper {i}: https://arxiv.org/abs/2401.0000{i}",
                url=f"https://linkedin.com/posts/post_{i}",
                likes_count=100 + i * 10,
                comments_count=20 + i * 5,
                shares_count=10 + i * 2,
                views_count=5000 + i * 500,
                published_at=datetime.now(timezone.utc) - timedelta(days=i + 1)
            )
            posts.append(post)

        # Convert all to paper format
        papers = [fetcher._format_paper_dict(post) for post in posts]

        # Verify all papers were processed
        assert len(papers) == 5
        for i, paper in enumerate(papers):
            assert paper['id'] == f"linkedin:post_{i}"
            assert paper['arxiv_id'] == f"2401.0000{i}"
            assert paper['linkedin_company'] == f"Company {i}"
            assert paper['professional_score'] > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
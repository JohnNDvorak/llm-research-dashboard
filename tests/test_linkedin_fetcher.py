"""
Tests for LinkedIn fetcher module.

Comprehensive test suite covering:
- Initialization and mode selection
- API and scraping modes
- Data extraction and parsing
- Rate limiting and anti-detection
- Professional scoring
- Error handling and resilience
- Integration with existing fetchers
"""

import pytest
import asyncio
import time
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from src.fetch.linkedin_fetcher import (
    LinkedinFetcher,
    LinkedInPost,
    LinkedInError,
    LinkedInRateLimitError,
    LinkedInBlockedError,
    AntiDetectionManager,
    LinkedInCache
)
from src.utils.logger import logger


class TestAntiDetectionManager:
    """Test anti-detection measures for LinkedIn scraping."""

    def test_init(self):
        """Test AntiDetectionManager initialization."""
        manager = AntiDetectionManager()
        assert len(manager.user_agents) == 5
        assert manager.current_agent_index == 0

    def test_get_random_user_agent(self):
        """Test getting random user agent."""
        manager = AntiDetectionManager()
        agent = manager.get_random_user_agent()
        assert agent in manager.user_agents
        assert "Mozilla" in agent

    def test_rotate_user_agent(self):
        """Test user agent rotation."""
        manager = AntiDetectionManager()
        initial_index = manager.current_agent_index
        agent1 = manager.rotate_user_agent()
        assert manager.current_agent_index == (initial_index + 1) % len(manager.user_agents)
        assert agent1 in manager.user_agents

    @pytest.mark.asyncio
    async def test_simulate_human_behavior(self):
        """Test human behavior simulation."""
        manager = AntiDetectionManager()
        page = AsyncMock()

        await manager.simulate_human_behavior(page)

        # Verify scroll was called
        page.evaluate.assert_called()
        # Verify mouse movement
        page.mouse.move.assert_called()
        # Verify multiple calls happened (behavior simulation)
        assert page.evaluate.call_count + page.mouse.move.call_count >= 2


class TestLinkedInCache:
    """Test LinkedIn-specific caching and state management."""

    def test_init(self):
        """Test cache initialization."""
        cache = LinkedInCache(max_daily=50)
        assert cache.max_daily == 50
        assert cache.daily_fetch_count == 0
        assert len(cache.seen_posts) == 0

    def test_is_post_fetched(self):
        """Test post fetching tracking."""
        cache = LinkedInCache()
        post_id = "test_post_123"

        assert not cache.is_post_fetched(post_id)
        cache.add_post(post_id)
        assert cache.is_post_fetched(post_id)

    def test_should_pause(self):
        """Test pause logic based on daily limits."""
        cache = LinkedInCache(max_daily=100)

        # Should not pause initially
        assert not cache.should_pause()

        # Add posts up to limit
        for _ in range(80):  # Conservative limit
            cache.increment_fetch_count()

        # Should pause after 80 posts
        assert cache.should_pause()

    def test_daily_reset(self):
        """Test daily counter reset."""
        cache = LinkedInCache()
        cache.daily_fetch_count = 50
        cache.last_reset_date = datetime.now() - timedelta(days=1)

        # Should reset when checking pause
        cache.should_pause()
        assert cache.daily_fetch_count == 0

    def test_company_fetch_tracking(self):
        """Test company-specific fetch tracking."""
        cache = LinkedInCache()
        company = "OpenAI"

        assert cache.get_company_last_fetch(company) is None
        cache.update_company_fetch(company)
        assert cache.get_company_last_fetch(company) is not None


class TestLinkedinFetcher:
    """Test main LinkedIn fetcher functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock LinkedIn configuration."""
        return {
            'tracked_companies': [
                {'name': 'OpenAI', 'company_id': 'openai', 'priority': 'high'},
                {'name': 'Anthropic', 'company_id': 'anthropic', 'priority': 'high'}
            ],
            'hashtags': ['#LLM', '#MachineLearning', '#AIResearch'],
            'rate_limit_delay': 5,
            'max_posts_per_day': 100,
            'preferred_method': 'scraping'
        }

    @pytest.fixture
    def sample_linkedin_post(self):
        """Sample LinkedIn post for testing."""
        return LinkedInPost(
            id="123456789",
            author_name="John Doe",
            author_title="Research Scientist at OpenAI",
            author_profile_url="https://linkedin.com/in/johndoe",
            company="OpenAI",
            text="Excited to share our new paper on DPO: https://arxiv.org/abs/2401.00001",
            url="https://linkedin.com/posts/123456789",
            likes_count=150,
            comments_count=25,
            shares_count=10,
            views_count=5000,
            published_at=datetime.now(timezone.utc) - timedelta(days=1)
        )

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_init_scraping_mode(self, mock_get_config, mock_config):
        """Test initialization in scraping mode."""
        mock_get_config.return_value = {'linkedin': mock_config}

        with patch.dict(os.environ, {'LINKEDIN_EMAIL': 'test@example.com', 'LINKEDIN_PASSWORD': 'password'}):
            fetcher = LinkedinFetcher()
            assert fetcher.mode == "scraping"
            assert fetcher.base_delay == 5
            assert fetcher.api_client is None

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_init_api_mode(self, mock_get_config, mock_config):
        """Test initialization in API mode."""
        mock_get_config.return_value = {'linkedin': mock_config}

        with patch.dict(os.environ, {'LINKEDIN_ACCESS_TOKEN': 'token123'}):
            fetcher = LinkedinFetcher()
            assert fetcher.mode == "api"

    @patch('src.fetch.linkedin_fetcher.get_queries_config')
    def test_init_no_credentials(self, mock_get_config, mock_config):
        """Test initialization with no credentials (defaults to scraping)."""
        mock_get_config.return_value = {'linkedin': mock_config}

        # Remove all LinkedIn credentials
        with patch.dict(os.environ, {}, clear=True):
            fetcher = LinkedinFetcher()
            assert fetcher.mode == "scraping"

    def test_extract_arxiv_id(self, mock_config):
        """Test arXiv ID extraction from text."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            test_cases = [
                ("Check out our paper: https://arxiv.org/abs/2401.00001", "2401.00001"),
                ("Paper at arXiv:2401.12345", "2401.12345"),
                ("PDF: https://arxiv.org/pdf/2401.54321.pdf", "2401.54321"),
                ("No arXiv link here", None),
                ("", None)
            ]

            for text, expected in test_cases:
                assert fetcher._extract_arxiv_id(text) == expected

    def test_extract_company(self, mock_config):
        """Test company extraction from author title."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            test_cases = [
                ("Research Scientist at OpenAI", "OpenAI"),
                ("ML Engineer at Anthropic", "Anthropic"),
                ("Senior Researcher, Google DeepMind", "Google DeepMind"),
                ("No company here", None),
                ("", None)
            ]

            for title, expected in test_cases:
                assert fetcher._extract_company(title) == expected

    def test_is_verified_researcher(self, mock_config):
        """Test verified researcher detection."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            # Research titles
            assert fetcher._is_verified_researcher("Research Scientist at OpenAI", "OpenAI")
            assert fetcher._is_verified_researcher("AI Researcher", None)
            assert fetcher._is_verified_researcher("PhD Student", "Google DeepMind")

            # Non-research titles
            assert not fetcher._is_verified_researcher("Software Engineer", None)
            assert not fetcher._is_verified_researcher("Product Manager", "Random Corp")

    def test_calculate_professional_score(self, mock_config, sample_linkedin_post):
        """Test professional score calculation."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            # Base calculation: (150 * 1) + (25 * 5) + (10 * 3) + (5000 * 0.001)
            # = 150 + 125 + 30 + 5 = 310
            # With researcher boost: 310 * 1.5 = 465
            score = fetcher._calculate_professional_score(sample_linkedin_post)
            assert score == 465

    def test_professional_score_no_boost(self, mock_config):
        """Test professional score without researcher boost."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            post = LinkedInPost(
                id="123",
                author_name="Jane Doe",
                author_title="Software Engineer",
                author_profile_url="https://linkedin.com/in/jane",
                company=None,
                text="Paper link: https://arxiv.org/abs/2401.00001",
                url="https://linkedin.com/posts/123",
                likes_count=50,
                comments_count=5,
                shares_count=2,
                views_count=1000,
                published_at=datetime.now(timezone.utc)
            )

            # Base calculation: (50 * 1) + (5 * 5) + (2 * 3) + (1000 * 0.001)
            # = 50 + 25 + 6 + 1 = 82
            score = fetcher._calculate_professional_score(post)
            assert score == 82

    def test_format_paper_dict(self, mock_config, sample_linkedin_post):
        """Test paper dictionary formatting."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            paper = fetcher._format_paper_dict(sample_linkedin_post)

            assert paper['id'] == "linkedin:123456789"
            assert paper['source'] == 'linkedin'
            assert paper['authors'] == ["John Doe"]
            assert paper['social_score'] == 0
            assert paper['professional_score'] == 465
            assert paper['arxiv_id'] == "2401.00001"
            assert paper['linkedin_company'] == "OpenAI"
            assert paper['linkedin_author_title'] == "Research Scientist at OpenAI"

    def test_parse_count(self, mock_config):
        """Test count parsing from text."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            test_cases = [
                ("15 likes", 15),
                ("1.2K reactions", 1200),
                ("3.5M views", 3500000),
                ("no numbers", 0),
                ("", 0)
            ]

            for text, expected in test_cases:
                assert fetcher._parse_count(text) == expected

    def test_parse_time_ago(self, mock_config):
        """Test time parsing from 'time ago' text."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            now = datetime.now(timezone.utc)

            test_cases = [
                ("2d", now - timedelta(days=2)),
                ("3h", now - timedelta(hours=3)),
                ("1w", now - timedelta(weeks=1)),
                ("5m", now - timedelta(minutes=5)),
                ("invalid", now)
            ]

            for text, expected in test_cases:
                result = fetcher._parse_time_ago(text)
                # Allow small time differences
                assert abs((result - expected).total_seconds()) < 60

    def test_enforce_rate_limit(self, mock_config):
        """Test rate limiting enforcement."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher.last_request_time = time.time() - 10  # 10 seconds ago

            # Should not sleep if enough time passed
            start = time.time()
            fetcher._enforce_rate_limit()
            elapsed = time.time() - start
            assert elapsed < 1  # Should not sleep significantly

    @pytest.mark.asyncio
    async def test_switch_mode(self, mock_config):
        """Test mode switching."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher.mode = "api"
            fetcher.api_client = Mock()

            await fetcher._switch_mode()

            assert fetcher.mode == "scraping"
            assert fetcher.api_client is None

    @pytest.mark.asyncio
    @patch('src.fetch.linkedin_fetcher.async_playwright')
    async def test_init_browser(self, mock_playwright, mock_config):
        """Test browser initialization for scraping."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            # Mock playwright
            mock_pw_instance = AsyncMock()
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_playwright.return_value.start.return_value = mock_pw_instance
            mock_pw_instance.chromium.launch.return_value = mock_browser
            mock_browser.new_context.return_value = mock_context

            browser, context = await fetcher._init_browser()

            assert browser == mock_browser
            assert context == mock_context
            mock_browser.new_context.assert_called_once()
            mock_context.add_init_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_post_from_element(self, mock_config):
        """Test post extraction from HTML element."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            # Mock element and page
            element = AsyncMock()
            page = AsyncMock()

            # Setup mock returns
            element.get_attribute.side_effect = [
                "urn:li:activity:123456789",  # post_urn
                "https://linkedin.com/posts/123456789"  # post_url
            ]

            # Mock sub-elements
            text_element = AsyncMock()
            text_element.inner_text.return_value = "Check our new paper: https://arxiv.org/abs/2401.00001"

            author_element = AsyncMock()
            author_element.inner_text.return_value = "John Doe"

            title_element = AsyncMock()
            title_element.inner_text.return_value = "Research Scientist at OpenAI"

            profile_element = AsyncMock()
            profile_element.get_attribute.return_value = "https://linkedin.com/in/johndoe"

            likes_element = AsyncMock()
            likes_element.get_attribute.return_value = "150 likes"

            comments_element = AsyncMock()
            comments_element.get_attribute.return_value = "25 comments"

            shares_element = AsyncMock()
            shares_element.get_attribute.return_value = "10 shares"

            time_element = AsyncMock()
            time_element.inner_text.return_value = "1d"

            # Setup element.query_selector mocks
            element.query_selector.side_effect = lambda selector: {
                '.feed-shared-text__text': text_element,
                '.feed-shared-actor__name': author_element,
                '.feed-shared-actor__description': title_element,
                'a.feed-shared-actor__container-link': profile_element,
                '[aria-label*="like"]': likes_element,
                '[aria-label*="comment"]': comments_element,
                '[aria-label*="share"]': shares_element,
                '.feed-shared-timeago': time_element,
                'a[href*="/activity/"]': profile_element
            }.get(selector)

            post = await fetcher._extract_post_from_element(element, page)

            assert post is not None
            assert post.id == "123456789"
            assert post.author_name == "John Doe"
            assert post.author_title == "Research Scientist at OpenAI"
            assert post.company == "OpenAI"
            assert "2401.00001" in post.text
            assert post.likes_count == 150
            assert post.comments_count == 25
            assert post.shares_count == 10

    @pytest.mark.asyncio
    async def test_fetch_from_companies(self, mock_config):
        """Test fetching from company pages."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher._scrape_company_posts = AsyncMock(return_value=[])

            papers = await fetcher.fetch_from_companies(days=7)

            assert isinstance(papers, list)
            # Should call for each company
            assert fetcher._scrape_company_posts.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_by_hashtags(self, mock_config):
        """Test fetching by hashtags."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher._scrape_hashtag_posts = AsyncMock(return_value=[])

            papers = await fetcher.fetch_by_hashtags(days=7)

            assert isinstance(papers, list)
            # Should call for each hashtag
            assert fetcher._scrape_hashtag_posts.call_count == 3

    @pytest.mark.asyncio
    async def test_fetch_recent_papers(self, mock_config):
        """Test comprehensive paper fetching."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher.fetch_from_companies = AsyncMock(return_value=[{'id': 'paper1'}])
            fetcher.fetch_by_hashtags = AsyncMock(return_value=[{'id': 'paper2'}])

            papers = await fetcher.fetch_recent_papers(days=7)

            assert len(papers) == 2
            assert papers[0]['id'] == 'paper1'
            assert papers[1]['id'] == 'paper2'

    def test_get_stats(self, mock_config):
        """Test statistics retrieval."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            stats = fetcher.get_stats()

            assert 'mode' in stats
            assert 'daily_fetch_count' in stats
            assert 'tracked_companies' in stats
            assert stats['tracked_companies'] == 2
            assert stats['hashtags'] == 3

    def test_fetch_papers_sync_wrapper(self, mock_config):
        """Test synchronous wrapper for async method."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher.fetch_recent_papers = AsyncMock(return_value=[{'id': 'paper1'}, {'id': 'paper2'}])

            papers = fetcher.fetch_papers(max_results=1)

            assert len(papers) == 1
            assert papers[0]['id'] == 'paper1'

    @pytest.mark.asyncio
    async def test_fetch_with_retry_success(self, mock_config):
        """Test retry mechanism on success."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            async def mock_func():
                return "success"

            result = await fetcher._fetch_with_retry(mock_func)
            assert result == "success"

    @pytest.mark.asyncio
    async def test_fetch_with_retry_rate_limit(self, mock_config):
        """Test retry mechanism on rate limit."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            call_count = 0
            async def mock_func():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise LinkedInRateLimitError("Rate limited")
                return "success"

            with patch('asyncio.sleep') as mock_sleep:
                result = await fetcher._fetch_with_retry(mock_func)
                assert result == "success"
                assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_fetch_with_retry_blocked(self, mock_config):
        """Test retry mechanism on block."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher._switch_mode = AsyncMock()

            async def mock_func():
                raise LinkedInBlockedError("Blocked")

            with pytest.raises(LinkedInBlockedError):
                await fetcher._fetch_with_retry(mock_func)

            fetcher._switch_mode.assert_called_once()


class TestLinkedInIntegration:
    """Integration tests for LinkedIn fetcher with other components."""

    @pytest.mark.asyncio
    async def test_integration_with_paper_deduplicator(self, mock_config):
        """Test integration with PaperDeduplicator."""
        from src.fetch.paper_deduplicator import PaperDeduplicator

        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            # Create fetcher and deduplicator
            fetcher = LinkedinFetcher()
            deduplicator = PaperDeduplicator()

            # Mock LinkedIn post
            linkedin_papers = [{
                'id': 'linkedin:123',
                'title': None,
                'abstract': None,
                'authors': ['John Doe'],
                'source': 'linkedin',
                'social_score': 0,
                'professional_score': 100,
                'arxiv_id': '2401.00001',
                'url': None
            }]

            # Mock arXiv papers
            arxiv_papers = [{
                'id': 'arxiv:2401.00001',
                'title': 'DPO: Direct Preference Optimization',
                'abstract': 'We propose DPO...',
                'authors': ['John Doe', 'Jane Smith'],
                'source': 'arxiv',
                'social_score': 0,
                'professional_score': 0,
                'arxiv_id': '2401.00001',
                'url': 'https://arxiv.org/abs/2401.00001'
            }]

            # Deduplicate
            all_papers = linkedin_papers + arxiv_papers
            deduplicated = deduplicator.deduplicate(all_papers)

            # Should merge into one paper with combined scores
            assert len(deduplicated) == 1
            merged = deduplicated[0]
            assert merged['id'] == 'arxiv:2401.00001'
            assert merged['title'] == 'DPO: Direct Preference Optimization'
            assert merged['professional_score'] == 100  # From LinkedIn
            assert 'linkedin' in merged.get('sources', [])


class TestLinkedInErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_no_companies_configured(self, mock_config):
        """Test behavior with no companies configured."""
        empty_config = {**mock_config, 'tracked_companies': []}

        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': empty_config}):
            fetcher = LinkedinFetcher()
            papers = await fetcher.fetch_from_companies(days=7)
            assert papers == []

    @pytest.mark.asyncio
    async def test_no_hashtags_configured(self, mock_config):
        """Test behavior with no hashtags configured."""
        empty_config = {**mock_config, 'hashtags': []}

        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': empty_config}):
            fetcher = LinkedinFetcher()
            papers = await fetcher.fetch_by_hashtags(days=7)
            assert papers == []

    def test_extract_post_from_element_error(self, mock_config):
        """Test handling of extraction errors."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            # Mock element that raises exception
            element = AsyncMock()
            element.get_attribute.side_effect = Exception("DOM error")
            page = AsyncMock()

            # Should return None on error
            result = asyncio.run(fetcher._extract_post_from_element(element, page))
            assert result is None

    @pytest.mark.asyncio
    async def test_company_fetch_error(self, mock_config):
        """Test handling of company fetch errors."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()
            fetcher._scrape_company_posts = AsyncMock(side_effect=Exception("Network error"))

            # Should continue despite errors
            papers = await fetcher.fetch_from_companies(days=7)
            assert papers == []  # No papers due to error

    def test_parse_invalid_time(self, mock_config):
        """Test parsing invalid time strings."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            result = fetcher._parse_time_ago("invalid time string")
            # Should return current time for invalid input
            assert isinstance(result, datetime)

    def test_calculate_engagement_rate_no_views(self, mock_config):
        """Test engagement rate calculation without views."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            post = LinkedInPost(
                id="123",
                author_name="Test",
                author_title="Title",
                author_profile_url="url",
                company=None,
                text="text",
                url="url",
                likes_count=100,
                comments_count=10,
                shares_count=5,
                views_count=None,  # No views
                published_at=datetime.now(timezone.utc)
            )

            rate = fetcher._calculate_engagement_rate(post)
            assert rate == 0.0  # Should be 0 without views

    @pytest.mark.asyncio
    async def test_close_browser(self, mock_config):
        """Test browser resource cleanup."""
        with patch('src.fetch.linkedin_fetcher.get_queries_config', return_value={'linkedin': mock_config}):
            fetcher = LinkedinFetcher()

            # Mock browser
            mock_browser = AsyncMock()
            fetcher.browser = mock_browser
            fetcher.browser_context = AsyncMock()

            await fetcher.close()

            mock_browser.close.assert_called_once()
            assert fetcher.browser is None
            assert fetcher.browser_context is None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
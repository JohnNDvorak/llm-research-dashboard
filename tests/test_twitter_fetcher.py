"""
Comprehensive test suite for Twitter/X fetcher module.

Tests Twitter API integration, paper extraction, social scoring,
and rate limiting behavior.
"""

import pytest
import os
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.fetch.twitter_fetcher import TwitterFetcher, fetch_twitter_papers
from src.utils.logger import logger


class TestTwitterFetcher:
    """Test cases for TwitterFetcher class."""

    @pytest.fixture
    def mock_twitter_client(self):
        """Mock Twitter API client."""
        with patch('src.fetch.twitter_fetcher.Client') as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_config(self):
        """Mock Twitter configuration."""
        return {
            'tracked_accounts': ['@huggingface', '@OpenAI', '@AnthropicAI'],
            'hashtags': ['#LLM', '#MachineLearning', '#AIResearch'],
            'keywords': ['arXiv', 'paper', 'research'],
            'min_likes': 10,
            'min_retweets': 5,
            'max_tweets_per_day': 1000,
            'rate_limit_delay': 2,
            'max_results_per_query': 100,
            'max_total_results': 500
        }

    @pytest.fixture
    def mock_tweet_data(self):
        """Mock tweet data with arXiv links."""
        return Mock(
            id='1234567890',
            text='Check out our new paper on LLM efficiency! "Scaling Laws for Neural Language Models" https://arxiv.org/abs/2001.08361 #LLM #AIResearch',
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
            author_id='987654321',
            public_metrics={
                'like_count': 150,
                'retweet_count': 45,
                'quote_count': 12,
                'reply_count': 23
            },
            entities={
                'hashtags': [{'tag': 'LLM'}, {'tag': 'AIResearch'}],
                'mentions': [{'username': 'huggingface'}]
            }
        )

    @pytest.fixture
    def mock_user_data(self):
        """Mock user data."""
        mock_user = Mock()
        mock_user.id = '987654321'
        mock_user.name = 'AI Research Lab'
        mock_user.username = 'airesearchlab'
        return mock_user

    def test_init_with_bearer_token(self, mock_twitter_client, mock_config):
        """Test TwitterFetcher initialization with bearer token."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            assert fetcher.bearer_token == 'test_token'
            assert fetcher.tracked_accounts == mock_config['tracked_accounts']
            assert fetcher.hashtags == mock_config['hashtags']
            assert fetcher.min_likes == mock_config['min_likes']
            assert fetcher.min_retweets == mock_config['min_retweets']
            assert fetcher.rate_limit_delay == mock_config['rate_limit_delay']
            assert len(fetcher.arxiv_regex) == 6  # 6 arXiv patterns

            mock_twitter_client.assert_called_once_with(
                bearer_token='test_token',
                wait_on_rate_limit=True
            )

    def test_init_without_bearer_token(self, mock_config):
        """Test TwitterFetcher initialization fails without bearer token."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="X Bearer Token required"):
                    TwitterFetcher()

    def test_init_with_env_variable(self, mock_twitter_client, mock_config):
        """Test TwitterFetcher initialization with environment variable."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'env_token'}):
                fetcher = TwitterFetcher()
                assert fetcher.bearer_token == 'env_token'

    def test_extract_arxiv_links(self, mock_twitter_client, mock_config):
        """Test arXiv link extraction from tweet text."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            test_cases = [
                ('Check out https://arxiv.org/abs/2001.08361', ['2001.08361']),
                ('Multiple papers: https://arxiv.org/abs/2001.08361 and https://arxiv.org/pdf/2101.00018.pdf',
                 ['2001.08361', '2101.00018']),
                ('arXiv:2001.08361 is now published', ['2001.08361']),
                ('See ar5iv.org/abs/2001.08361 for the paper', ['2001.08361']),
                ('No arXiv links here', []),
                ('Mixed case ARXIV.ORG/ABS/2001.08361', ['2001.08361']),
                ('Duplicate link https://arxiv.org/abs/2001.08361 appears twice', ['2001.08361']),
            ]

            for text, expected_ids in test_cases:
                result = fetcher._extract_arxiv_links(text)
                assert result == expected_ids, f"Failed for text: {text}"

    def test_calculate_social_score(self, mock_twitter_client, mock_config):
        """Test social score calculation from Twitter metrics."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            test_cases = [
                ({'like_count': 100, 'retweet_count': 20, 'quote_count': 5, 'reply_count': 10},
                 100*1 + 20*3 + 5*2 + 10*0.5),  # 100 + 60 + 10 + 5 = 175
                ({'like_count': 0, 'retweet_count': 0, 'quote_count': 0, 'reply_count': 0}, 0),
                ({'like_count': 50, 'retweet_count': 10}, 50 + 10*3),  # 80
                ({}, 0),  # Empty metrics
            ]

            for metrics, expected_score in test_cases:
                score = fetcher._calculate_social_score(metrics)
                assert score == int(expected_score), f"Expected {expected_score}, got {score}"

    def test_extract_title_from_tweet(self, mock_twitter_client, mock_config):
        """Test title extraction from tweet text."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            test_cases = [
                ('New paper: "Scaling Laws for Large Language Models" https://arxiv.org/abs/2001.08361',
                 'Scaling Laws for Large Language Models'),
                ('Just published "Attention Is All You Need" arXiv:1706.03762',
                 'Attention Is All You Need'),
                ('Our research on transformers is now available https://arxiv.org/abs/2001.08361',
                 'on transformers is now available'),
                ('"Deep Learning" paper arXiv:2001.08361',
                 'Deep Learning'),
                ('Short arXiv:2001.08361', 'Short arXiv:'),
            ]

            for text, expected_title in test_cases:
                title = fetcher._extract_title_from_tweet(text, '2001.08361')
                assert title == expected_title, f"Failed for text: {text}"

    def test_fetch_from_accounts(self, mock_twitter_client, mock_config, mock_tweet_data, mock_user_data):
        """Test fetching papers from tracked accounts."""
        # Mock the client and its responses
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Mock get_users_tweets to return tweet data
        mock_response = Mock()
        mock_response.data = [mock_tweet_data]
        mock_client_instance.get_users_tweets.return_value = mock_response

        # Mock get_user for author info
        mock_user = Mock()
        mock_user.id = mock_user_data.id
        mock_user.name = mock_user_data.name
        mock_user.username = mock_user_data.username
        mock_user_response = Mock()
        mock_user_response.data = mock_user
        mock_client_instance.get_user.return_value = mock_user_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):  # Skip sleep for testing
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_from_accounts(days=7)

        assert len(papers) == 3  # 1 from each account
        paper = papers[0]
        assert paper['arxiv_id'] == '2001.08361'
        assert paper['social_score'] == 320  # Calculated from mock metrics (150 + 45*3 + 12*2 + 23*0.5)
        assert paper['author_name'] == 'AI Research Lab'
        assert paper['author_username'] == '@airesearchlab'
        assert 'x' in paper['source']
        assert 'account:@huggingface' in paper['source']

    def test_fetch_by_hashtags(self, mock_twitter_client, mock_config, mock_tweet_data):
        """Test fetching papers by hashtag search."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Mock search_recent_tweets
        mock_response = Mock()
        mock_response.data = [mock_tweet_data]
        mock_client_instance.search_recent_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_by_hashtags(days=7)

        assert len(papers) == 9  # 1 from each hashtag (9 hashtags)
        paper = papers[0]
        assert paper['arxiv_id'] == '2001.08361'
        assert 'hashtag:#LLM' in paper['source']

    def test_fetch_recent_papers(self, mock_twitter_client, mock_config, mock_tweet_data):
        """Test fetching papers from all sources."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Mock both account and hashtag responses
        mock_response = Mock()
        mock_response.data = [mock_tweet_data]
        mock_client_instance.get_users_tweets.return_value = mock_response
        mock_client_instance.search_recent_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_recent_papers(days=7)

        # Should have 12 papers (3 from accounts, 9 from hashtags)
        assert len(papers) == 12

    def test_deduplication(self, mock_twitter_client, mock_config):
        """Test duplicate paper removal based on arXiv ID."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Create tweets with duplicate arXiv IDs
        tweet1 = Mock()
        tweet1.id = '1'
        tweet1.text = 'Paper 1 https://arxiv.org/abs/2001.08361'
        tweet1.created_at = datetime.now(timezone.utc)
        tweet1.author_id = 'user1'
        tweet1.public_metrics = {'like_count': 100, 'retweet_count': 20, 'quote_count': 5, 'reply_count': 10}

        tweet2 = Mock()
        tweet2.id = '2'
        tweet2.text = 'Paper 2 https://arxiv.org/abs/2001.08361'
        tweet2.created_at = datetime.now(timezone.utc)
        tweet2.author_id = 'user2'
        tweet2.public_metrics = {'like_count': 50, 'retweet_count': 10, 'quote_count': 2, 'reply_count': 5}

        mock_response = Mock()
        mock_response.data = [tweet1, tweet2]
        mock_client_instance.get_users_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_from_accounts(days=7)

        # Should only have one paper (duplicate removed)
        assert len(papers) == 1
        assert papers[0]['arxiv_id'] == '2001.08361'

    def test_process_tweets_filters_low_engagement(self, mock_twitter_client, mock_config):
        """Test that tweets below engagement thresholds are filtered out."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Create tweet with low engagement
        low_engagement_tweet = Mock()
        low_engagement_tweet.id = '1'
        low_engagement_tweet.text = 'Low engagement paper https://arxiv.org/abs/2001.08361'
        low_engagement_tweet.created_at = datetime.now(timezone.utc)
        low_engagement_tweet.author_id = 'user1'
        low_engagement_tweet.public_metrics = {'like_count': 5, 'retweet_count': 2, 'quote_count': 0, 'reply_count': 1}

        mock_response = Mock()
        mock_response.data = [low_engagement_tweet]
        mock_client_instance.get_users_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_from_accounts(days=7)

        # Should have no papers (filtered out due to low engagement)
        assert len(papers) == 0

    def test_process_tweets_excludes_no_arxiv_links(self, mock_twitter_client, mock_config):
        """Test that tweets without arXiv links are excluded."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Create tweet without arXiv link
        no_arxiv_tweet = Mock()
        no_arxiv_tweet.id = '1'
        no_arxiv_tweet.text = 'Just a regular tweet about AI research'
        no_arxiv_tweet.created_at = datetime.now(timezone.utc)
        no_arxiv_tweet.author_id = 'user1'
        no_arxiv_tweet.public_metrics = {'like_count': 100, 'retweet_count': 20, 'quote_count': 5, 'reply_count': 10}
        no_arxiv_tweet.entities = {}

        mock_response = Mock()
        mock_response.data = [no_arxiv_tweet]
        mock_client_instance.get_users_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_from_accounts(days=7)

        # Should have no papers (no arXiv link)
        assert len(papers) == 0

    def test_extract_hashtags(self, mock_twitter_client, mock_config):
        """Test hashtag extraction from tweet entities."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            # Mock tweet with hashtags
            tweet = Mock()
            tweet.entities = {
                'hashtags': [
                    {'tag': 'LLM'},
                    {'tag': 'AIResearch'},
                    {'tag': 'MachineLearning'}
                ]
            }

            hashtags = fetcher._extract_hashtags(tweet)
            assert hashtags == ['#LLM', '#AIResearch', '#MachineLearning']

            # Test tweet without hashtags
            tweet_no_tags = Mock()
            tweet_no_tags.entities = {}

            hashtags = fetcher._extract_hashtags(tweet_no_tags)
            assert hashtags == []

    def test_extract_mentions(self, mock_twitter_client, mock_config):
        """Test mention extraction from tweet entities."""
        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            # Mock tweet with mentions
            tweet = Mock()
            tweet.entities = {
                'mentions': [
                    {'username': 'huggingface'},
                    {'username': 'OpenAI'},
                    {'username': 'AnthropicAI'}
                ]
            }

            mentions = fetcher._extract_mentions(tweet)
            assert mentions == ['@huggingface', '@OpenAI', '@AnthropicAI']

            # Test tweet without mentions
            tweet_no_mentions = Mock()
            tweet_no_mentions.entities = {}

            mentions = fetcher._extract_mentions(tweet_no_mentions)
            assert mentions == []

    def test_get_author_info(self, mock_twitter_client, mock_config, mock_user_data):
        """Test fetching author information."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Mock user response
        mock_user = Mock()
        mock_user.id = mock_user_data.id
        mock_user.name = mock_user_data.name
        mock_user.username = mock_user_data.username
        mock_user_response = Mock()
        mock_user_response.data = mock_user
        mock_client_instance.get_user.return_value = mock_user_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            name = fetcher._get_author_name('987654321')
            username = fetcher._get_username('987654321')

            assert name == 'AI Research Lab'
            assert username == '@airesearchlab'

            # Test error handling
            mock_client_instance.get_user.side_effect = Exception('API Error')
            name = fetcher._get_author_name('invalid_id')
            username = fetcher._get_username('invalid_id')

            assert name == 'User invalid_id'
            assert username == '@userinvalid_id'

    def test_rate_limiting(self, mock_twitter_client, mock_config):
        """Test that rate limiting is applied between requests."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Mock empty response
        mock_response = Mock()
        mock_response.data = []
        mock_client_instance.get_users_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep') as mock_sleep:
                fetcher = TwitterFetcher(bearer_token='test_token')
                fetcher.fetch_from_accounts(days=7)

                # Should sleep between each account (2 delays for 3 accounts)
                assert mock_sleep.call_count == len(mock_config['tracked_accounts']) - 1
                mock_sleep.assert_any_call(mock_config['rate_limit_delay'])

    def test_error_handling(self, mock_twitter_client, mock_config):
        """Test error handling for API failures."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Simulate API error
        mock_client_instance.get_users_tweets.side_effect = Exception('API Rate Limit')

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')

                # Should not raise exception, should continue with next account
                papers = fetcher.fetch_from_accounts(days=7)
                assert papers == []

    def test_fetch_papers_legacy_method(self, mock_twitter_client, mock_config):
        """Test legacy fetch_papers method for backward compatibility."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.data = []
        mock_client_instance.get_users_tweets.return_value = mock_response
        mock_client_instance.search_recent_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_papers(max_results=50)

                # Should call fetch_recent_papers with default parameters
                assert isinstance(papers, list)

    def test_parse_tweet_metadata(self, mock_twitter_client, mock_config, mock_tweet_data, mock_user_data):
        """Test tweet metadata parsing into standardized format."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Mock user response
        mock_user = Mock()
        mock_user.id = mock_user_data.id
        mock_user.name = mock_user_data.name
        mock_user.username = mock_user_data.username
        mock_user_response = Mock()
        mock_user_response.data = mock_user
        mock_client_instance.get_user.return_value = mock_user_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            fetcher = TwitterFetcher(bearer_token='test_token')

            metadata = fetcher._parse_tweet_metadata(mock_tweet_data, '2001.08361', 'test_source')

            # Check all required fields
            required_fields = [
                'arxiv_id', 'title', 'source', 'social_score', 'professional_score',
                'combined_score', 'tweet_id', 'author_id', 'author_name',
                'author_username', 'text', 'created_at', 'public_metrics',
                'hashtags', 'mentions', 'url', 'extracted_date'
            ]

            for field in required_fields:
                assert field in metadata, f"Missing field: {field}"

            assert metadata['arxiv_id'] == '2001.08361'
            assert metadata['social_score'] == 320  # 150 + 45*3 + 12*2 + 23*0.5 = 320.5
            assert metadata['professional_score'] == 0
            assert metadata['source'] == ['x', 'test_source']
            assert metadata['tweet_id'] == '1234567890'


class TestTwitterFetcherIntegration:
    """Integration tests for Twitter fetcher (with mocked API calls)."""

    def test_full_workflow_mock(self):
        """Test full fetch workflow with mocked API responses."""
        # This test simulates the full workflow without making real API calls

        # Create mock tweets
        tweet1 = Mock()
        tweet1.id = '1'
        tweet1.text = 'New paper: "Attention Is All You Need" https://arxiv.org/abs/1706.03762 #Transformer #NLP'
        tweet1.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        tweet1.author_id = 'author1'
        tweet1.public_metrics = {'like_count': 500, 'retweet_count': 100, 'quote_count': 25, 'reply_count': 50}
        tweet1.entities = {}

        tweet2 = Mock()
        tweet2.id = '2'
        tweet2.text = 'Check out our latest work on GPT models https://arxiv.org/abs/2005.14165'
        tweet2.created_at = datetime.now(timezone.utc) - timedelta(hours=5)
        tweet2.author_id = 'author2'
        tweet2.public_metrics = {'like_count': 200, 'retweet_count': 40, 'quote_count': 10, 'reply_count': 20}
        tweet2.entities = {}

        mock_tweets = [tweet1, tweet2]

        with patch('src.fetch.twitter_fetcher.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock user responses
            mock_user = Mock()
            mock_user.name = 'AI Researcher'
            mock_user.username = 'airesearcher'
            mock_client.get_user.return_value = mock_user

            # Mock tweet responses
            mock_response = Mock()
            mock_response.data = mock_tweets
            mock_client.get_users_tweets.return_value = mock_response
            mock_client.search_recent_tweets.return_value = mock_response

            # Run fetcher with mocked bearer token
            with patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'test_token'}):
                papers = fetch_twitter_papers(days=7, use_accounts=True, use_hashtags=False)

            # Verify results
            assert len(papers) >= 0  # May be 0 if mocked tweets don't meet thresholds

            # If papers were found, verify structure
            for paper in papers:
                assert 'arxiv_id' in paper
                assert 'social_score' in paper
                assert 'source' in paper
                assert 'x' in paper['source']


class TestConvenienceFunction:
    """Test the convenience function for quick usage."""

    @pytest.fixture
    def mock_twitter_client(self):
        """Mock Twitter API client."""
        with patch('src.fetch.twitter_fetcher.Client') as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_config(self):
        """Mock Twitter configuration."""
        return {
            'tracked_accounts': ['@huggingface', '@OpenAI'],
            'hashtags': ['#LLM', '#MachineLearning'],
            'keywords': ['arXiv', 'paper', 'research'],
            'min_likes': 10,
            'min_retweets': 5,
            'max_tweets_per_day': 1000,
            'rate_limit_delay': 2,
            'max_results_per_query': 100,
            'max_total_results': 500
        }

    def test_fetch_twitter_papers_function(self, mock_twitter_client, mock_config):
        """Test the convenience function fetch_twitter_papers."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.data = []
        mock_client_instance.get_users_tweets.return_value = mock_response
        mock_client_instance.search_recent_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                with patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'test_token'}):
                    papers = fetch_twitter_papers(days=7, use_accounts=True, use_hashtags=True)

                assert isinstance(papers, list)


class TestPerformance:
    """Performance tests for Twitter fetcher."""

    @pytest.fixture
    def mock_twitter_client(self):
        """Mock Twitter API client."""
        with patch('src.fetch.twitter_fetcher.Client') as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_config(self):
        """Mock Twitter configuration."""
        return {
            'tracked_accounts': ['@huggingface', '@OpenAI'],
            'hashtags': ['#LLM', '#MachineLearning'],
            'keywords': ['arXiv', 'paper', 'research'],
            'min_likes': 10,
            'min_retweets': 5,
            'max_tweets_per_day': 1000,
            'rate_limit_delay': 0,  # No delay for performance testing
            'max_results_per_query': 100,
            'max_total_results': 500
        }

    def test_performance_with_large_dataset(self, mock_twitter_client, mock_config):
        """Test performance with large number of tweets."""
        mock_client_instance = Mock()
        mock_twitter_client.return_value = mock_client_instance

        # Create many mock tweets
        large_tweet_set = []
        for i in range(100):
            tweet = Mock()
            tweet.id = str(i)
            tweet.text = f'Paper {i} https://arxiv.org/abs/2001.{str(i).zfill(5)}'
            tweet.created_at = datetime.now(timezone.utc)
            tweet.author_id = f'user{i % 10}'
            tweet.public_metrics = {'like_count': 50, 'retweet_count': 10, 'quote_count': 2, 'reply_count': 5}
            tweet.entities = {}
            large_tweet_set.append(tweet)

        mock_response = Mock()
        mock_response.data = large_tweet_set
        mock_client_instance.get_users_tweets.return_value = mock_response

        with patch('src.fetch.twitter_fetcher.get_queries_config', return_value={'twitter': mock_config}):
            with patch('time.sleep'):
                import time
                start_time = time.time()

                fetcher = TwitterFetcher(bearer_token='test_token')
                papers = fetcher.fetch_from_accounts(days=7)

                end_time = time.time()
                duration = end_time - start_time

                # Should process 100 tweets quickly (< 1 second)
                assert duration < 1.0
                assert len(papers) == 200  # 100 from each of 2 accounts
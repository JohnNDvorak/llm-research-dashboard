"""
X (formerly Twitter) fetcher for LLM research papers.

This module fetches papers mentioned on X, extracts social metrics,
and identifies arXiv links shared by researchers and AI labs.
"""

import re
import time
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set
import tweepy
from tweepy import Client
import logging

from src.utils.logger import logger
from src.utils.config_loader import get_queries_config


class TwitterFetcher:
    """Fetches LLM papers from X (formerly Twitter) with social metrics."""

    def __init__(self, bearer_token: Optional[str] = None):
        """
        Initialize X fetcher with API credentials.

        Args:
            bearer_token: X API Bearer Token (defaults to environment variable)
        """
        self.bearer_token = bearer_token or os.getenv('TWITTER_BEARER_TOKEN')

        if not self.bearer_token:
            raise ValueError(
                "X Bearer Token required. Set TWITTER_BEARER_TOKEN environment variable "
                "or pass bearer_token parameter"
            )

        # Initialize X client
        self.client = Client(
            bearer_token=self.bearer_token,
            wait_on_rate_limit=True
        )

        # Load configuration
        self.config = get_queries_config().get('twitter', {})
        self.tracked_accounts = self.config.get('tracked_accounts', [])
        self.hashtags = self.config.get('hashtags', [])
        self.keywords = self.config.get('keywords', [])
        self.min_likes = self.config.get('min_likes', 10)
        self.min_retweets = self.config.get('min_retweets', 5)
        self.max_tweets_per_day = self.config.get('max_tweets_per_day', 1000)
        self.rate_limit_delay = self.config.get('rate_limit_delay', 2)

        # arXiv URL patterns
        self.arxiv_patterns = [
            r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'https?://arxiv\.org/pdf/(\d{4}\.\d{4,5})\.pdf',
            r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'arXiv:(\d{4}\.\d{4,5})',
            r'https?://ar5iv\.org/abs/(\d{4}\.\d{4,5})',
            r'ar5iv\.org/abs/(\d{4}\.\d{4,5})',
        ]

        # Compile regex patterns
        self.arxiv_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.arxiv_patterns]

        logger.info(
            "Twitter fetcher initialized",
            extra={
                "tracked_accounts": len(self.tracked_accounts),
                "hashtags": len(self.hashtags),
                "rate_limit_delay": self.rate_limit_delay
            }
        )

    def fetch_from_accounts(self, days: int = 7, max_tweets: int = None) -> List[Dict[str, Any]]:
        """
        Fetch recent tweets from tracked AI lab and researcher accounts.

        Args:
            days: Number of days to look back
            max_tweets: Maximum tweets to fetch per account (default from config)

        Returns:
            List of papers found in tweets
        """
        logger.info(f"Fetching tweets from {len(self.tracked_accounts)} tracked accounts")

        papers = []
        max_tweets = max_tweets or self.config.get('max_results_per_query', 100)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        for i, account in enumerate(self.tracked_accounts):
            try:
                logger.info(f"Fetching tweets from {account} ({i+1}/{len(self.tracked_accounts)})")

                # Get user tweets
                tweets = self.client.get_users_tweets(
                    username=account.replace('@', ''),
                    max_results=max_tweets,
                    start_time=start_time,
                    end_time=end_time,
                    tweet_fields=[
                        'created_at', 'public_metrics', 'context_annotations',
                        'entities', 'author_id', 'conversation_id'
                    ],
                    exclude=['retweets', 'replies']
                )

                if tweets and tweets.data:
                    account_papers = self._process_tweets(tweets.data, source=f"account:{account}")
                    papers.extend(account_papers)
                    logger.info(f"Found {len(account_papers)} papers from {account}")

                # Rate limiting
                if i < len(self.tracked_accounts) - 1:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching from {account}: {e}")
                continue

        logger.info(f"Total papers from accounts: {len(papers)}")
        return papers

    def fetch_by_hashtags(self, days: int = 7, max_tweets: int = None) -> List[Dict[str, Any]]:
        """
        Fetch tweets containing AI/ML research hashtags.

        Args:
            days: Number of days to look back
            max_tweets: Maximum total tweets to fetch

        Returns:
            List of papers found in tweets
        """
        logger.info(f"Fetching tweets by hashtags ({len(self.hashtags)} tags)")

        papers = []
        max_tweets = max_tweets or self.config.get('max_total_results', 500)
        tweets_per_hashtag = max(max_tweets // len(self.hashtags), 10)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        for i, hashtag in enumerate(self.hashtags):
            try:
                logger.info(f"Searching hashtag {hashtag} ({i+1}/{len(self.hashtags)})")

                # Search tweets with hashtag
                query = f"{hashtag} -is:retweet lang:en"
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=tweets_per_hashtag,
                    start_time=start_time,
                    end_time=end_time,
                    tweet_fields=[
                        'created_at', 'public_metrics', 'context_annotations',
                        'entities', 'author_id', 'conversation_id'
                    ]
                )

                if tweets and tweets.data:
                    hashtag_papers = self._process_tweets(tweets.data, source=f"hashtag:{hashtag}")
                    papers.extend(hashtag_papers)
                    logger.info(f"Found {len(hashtag_papers)} papers from {hashtag}")

                # Rate limiting
                if i < len(self.hashtags) - 1:
                    time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error searching {hashtag}: {e}")
                continue

        logger.info(f"Total papers from hashtags: {len(papers)}")
        return papers

    def fetch_recent_papers(self, days: int = 7, use_accounts: bool = True,
                           use_hashtags: bool = True) -> List[Dict[str, Any]]:
        """
        Fetch papers from all Twitter sources.

        Args:
            days: Number of days to look back
            use_accounts: Whether to fetch from tracked accounts
            use_hashtags: Whether to search by hashtags

        Returns:
            List of all papers found
        """
        logger.info(f"Starting Twitter paper fetch for last {days} days")

        all_papers = []

        if use_accounts:
            account_papers = self.fetch_from_accounts(days=days)
            all_papers.extend(account_papers)

        if use_hashtags:
            hashtag_papers = self.fetch_by_hashtags(days=days)
            all_papers.extend(hashtag_papers)

        # Remove duplicates based on arXiv ID
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            arxiv_id = paper.get('arxiv_id')
            if arxiv_id and arxiv_id not in seen_ids:
                seen_ids.add(arxiv_id)
                unique_papers.append(paper)

        logger.info(f"Found {len(all_papers)} total papers, {len(unique_papers)} unique after deduplication")
        return unique_papers

    def _process_tweets(self, tweets: List, source: str) -> List[Dict[str, Any]]:
        """
        Process a list of tweets to extract paper information.

        Args:
            tweets: List of tweet objects from Twitter API
            source: Source description for logging

        Returns:
            List of papers found in tweets
        """
        papers = []

        for tweet in tweets:
            try:
                # Check if tweet contains arXiv link
                arxiv_ids = self._extract_arxiv_links(tweet.text)

                if not arxiv_ids:
                    continue

                # Calculate social score
                social_score = self._calculate_social_score(tweet.public_metrics)

                # Check if meets thresholds
                if (tweet.public_metrics.get('like_count', 0) < self.min_likes or
                    tweet.public_metrics.get('retweet_count', 0) < self.min_retweets):
                    continue

                # For multiple arXiv IDs in one tweet, create separate entries
                for arxiv_id in arxiv_ids:
                    paper = self._parse_tweet_metadata(tweet, arxiv_id, source)
                    papers.append(paper)

            except Exception as e:
                logger.error(f"Error processing tweet {tweet.id}: {e}")
                continue

        return papers

    def _extract_arxiv_links(self, text: str) -> List[str]:
        """
        Extract arXiv IDs from tweet text.

        Args:
            text: Tweet text to search

        Returns:
            List of arXiv IDs found
        """
        found_ids = []

        for pattern in self.arxiv_regex:
            matches = pattern.findall(text)
            for match in matches:
                # Normalize arXiv ID
                arxiv_id = match.strip()
                if arxiv_id not in found_ids:
                    found_ids.append(arxiv_id)

        return found_ids

    def _calculate_social_score(self, metrics: Dict[str, int]) -> int:
        """
        Calculate social score based on Twitter metrics.

        Args:
            metrics: Public metrics from tweet

        Returns:
            Social score value
        """
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        quote_tweets = metrics.get('quote_count', 0)
        replies = metrics.get('reply_count', 0)

        # Weighted formula: retweets are more valuable than likes
        social_score = (likes * 1) + (retweets * 3) + (quote_tweets * 2) + (replies * 0.5)

        return int(social_score)

    def _parse_tweet_metadata(self, tweet, arxiv_id: str, source: str) -> Dict[str, Any]:
        """
        Parse tweet into standardized paper metadata format.

        Args:
            tweet: Tweet object from Twitter API
            arxiv_id: Extracted arXiv ID
            source: Source description

        Returns:
            Dictionary with paper metadata
        """
        # Extract paper title from tweet if possible
        title = self._extract_title_from_tweet(tweet.text, arxiv_id)

        # Get author information
        author_name = self._get_author_name(tweet.author_id)

        return {
            'arxiv_id': arxiv_id,
            'title': title,
            'source': ['x', source],
            'social_score': self._calculate_social_score(tweet.public_metrics),
            'professional_score': 0,  # Twitter doesn't provide professional score
            'combined_score': 0,  # Will be calculated by PaperDeduplicator
            'tweet_id': tweet.id,
            'author_id': tweet.author_id,
            'author_name': author_name,
            'author_username': self._get_username(tweet.author_id),
            'text': tweet.text,
            'created_at': tweet.created_at.isoformat(),
            'public_metrics': tweet.public_metrics,
            'hashtags': self._extract_hashtags(tweet),
            'mentions': self._extract_mentions(tweet),
            'url': f"https://x.com/i/web/status/{tweet.id}",
            'extracted_date': datetime.now(timezone.utc).isoformat(),
        }

    def _extract_title_from_tweet(self, text: str, arxiv_id: str) -> str:
        """
        Try to extract paper title from tweet text.

        Args:
            text: Tweet text
            arxiv_id: arXiv ID to look for

        Returns:
            Extracted title or arXiv ID as fallback
        """
        # Remove URLs and mentions to find title
        clean_text = re.sub(r'https?://\S+', '', text)
        clean_text = re.sub(r'@\w+', '', clean_text)
        clean_text = re.sub(r'#\w+', '', clean_text)

        # Look for title patterns (common in academic tweets)
        title_patterns = [
            r'[""]([^""]+)[""]',  # Text in quotes
            r'"([^"]+)"',  # Double quotes
            r'New (?:paper|preprint):? ([^.!?\n]+)',  # "New paper: Title"
            r'Just published:? ([^.!?\n]+)',  # "Just published: Title"
            r'Our (?:new )?(?:paper|work|research) ([^.!?\n]+)',  # "Our paper: Title"
        ]

        for pattern in title_patterns:
            match = re.search(pattern, clean_text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if 10 <= len(title) <= 200:  # Reasonable title length
                    return title

        # Fallback: use first part of text before arXiv link
        if arxiv_id in text:
            parts = text.split(arxiv_id)[0].strip()
            if len(parts) > 10:
                return parts[:200]  # Limit length

        # Final fallback
        return f"Paper {arxiv_id}"

    def _get_author_name(self, author_id: str) -> str:
        """Get author display name from ID."""
        try:
            user = self.client.get_user(id=author_id, user_fields=['name'])
            return user.data.name if user and user.data else f"User {author_id}"
        except:
            return f"User {author_id}"

    def _get_username(self, author_id: str) -> str:
        """Get author username from ID."""
        try:
            user = self.client.get_user(id=author_id, user_fields=['username'])
            return f"@{user.data.username}" if user and user.data else f"@user{author_id}"
        except:
            return f"@user{author_id}"

    def _extract_hashtags(self, tweet) -> List[str]:
        """Extract hashtags from tweet entities."""
        hashtags = []
        if hasattr(tweet, 'entities') and tweet.entities and 'hashtags' in tweet.entities:
            for tag in tweet.entities['hashtags']:
                hashtags.append(f"#{tag['tag']}")
        return hashtags

    def _extract_mentions(self, tweet) -> List[str]:
        """Extract mentions from tweet entities."""
        mentions = []
        if hasattr(tweet, 'entities') and tweet.entities and 'mentions' in tweet.entities:
            for mention in tweet.entities['mentions']:
                mentions.append(f"@{mention['username']}")
        return mentions

    def fetch_papers(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.

        Args:
            max_results: Maximum number of papers to return

        Returns:
            List of papers found
        """
        return self.fetch_recent_papers(days=7, use_accounts=True, use_hashtags=True)[:max_results]


# Convenience function for quick usage
def fetch_twitter_papers(days: int = 7, use_accounts: bool = True,
                        use_hashtags: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch papers from X (formerly Twitter).

    Args:
        days: Number of days to look back
        use_accounts: Whether to fetch from tracked accounts
        use_hashtags: Whether to search by hashtags

    Returns:
        List of papers found
    """
    fetcher = TwitterFetcher()
    return fetcher.fetch_recent_papers(days=days, use_accounts=use_accounts,
                                      use_hashtags=use_hashtags)

# Alias for consistency with X branding
def fetch_x_papers(days: int = 7, use_accounts: bool = True,
                  use_hashtags: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch papers from X (formerly Twitter).

    Args:
        days: Number of days to look back
        use_accounts: Whether to fetch from tracked accounts
        use_hashtags: Whether to search by hashtags

    Returns:
        List of papers found
    """
    return fetch_twitter_papers(days=days, use_accounts=use_accounts,
                               use_hashtags=use_hashtags)

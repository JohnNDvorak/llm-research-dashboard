"""
LinkedIn fetcher for LLM research papers.

This module fetches papers mentioned on LinkedIn, extracts professional metrics,
and identifies arXiv links shared by researchers and AI companies.
Implements dual-mode operation: API and scraping with automatic fallback.
"""

import re
import time
import random
import os
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import asyncio

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
try:
    from linkedin_api import Linkedin
    LINKEDIN_API_AVAILABLE = True
except ImportError:
    Linkedin = None
    LINKEDIN_API_AVAILABLE = False

from src.utils.logger import logger
from src.utils.config_loader import get_queries_config


@dataclass
class LinkedInPost:
    """LinkedIn post data structure."""
    id: str
    author_name: str
    author_title: str
    author_profile_url: str
    company: Optional[str]
    text: str
    url: str
    likes_count: int
    comments_count: int
    shares_count: int
    views_count: Optional[int]
    published_at: datetime
    reactions: Optional[Dict[str, int]] = None


class LinkedInError(Exception):
    """Base LinkedIn fetcher error."""
    pass


class LinkedInRateLimitError(LinkedInError):
    """Raised when rate limit is hit."""
    pass


class LinkedInBlockedError(LinkedInError):
    """Raised when access is blocked."""
    pass


class AntiDetectionManager:
    """Manages anti-detection measures for LinkedIn scraping."""

    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ]
        self.current_agent_index = 0

    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(self.user_agents)

    def rotate_user_agent(self) -> str:
        """Rotate to next user agent in sequence."""
        agent = self.user_agents[self.current_agent_index]
        self.current_agent_index = (self.current_agent_index + 1) % len(self.user_agents)
        return agent

    async def simulate_human_behavior(self, page: Page):
        """Simulate human-like browsing patterns."""
        # Random scroll
        await page.evaluate("window.scrollBy(0, Math.random() * 300)")
        await asyncio.sleep(random.uniform(0.5, 2.0))

        # Random mouse movement
        await page.mouse.move(
            random.randint(100, 800),
            random.randint(100, 600)
        )

        # Small delay
        await asyncio.sleep(random.uniform(0.2, 0.8))


class LinkedInCache:
    """Manages LinkedIn-specific caching and state."""

    def __init__(self, max_daily: int = 100):
        self.seen_posts: Set[str] = set()
        self.company_last_fetch: Dict[str, datetime] = {}
        self.daily_fetch_count: int = 0
        self.last_reset_date: datetime = datetime.now().date()
        self.max_daily: int = max_daily

    def is_post_fetched(self, post_id: str) -> bool:
        """Check if post was already fetched."""
        return post_id in self.seen_posts

    def add_post(self, post_id: str):
        """Mark post as fetched."""
        self.seen_posts.add(post_id)

    def should_pause(self) -> bool:
        """Check if we should pause due to rate limits."""
        # Reset daily counter if it's a new day
        if datetime.now().date() != self.last_reset_date:
            self.daily_fetch_count = 0
            self.last_reset_date = datetime.now().date()

        # Conservative pause after 80 posts
        return self.daily_fetch_count >= min(self.max_daily - 20, 80)

    def increment_fetch_count(self):
        """Increment daily fetch counter."""
        self.daily_fetch_count += 1

    def get_company_last_fetch(self, company: str) -> Optional[datetime]:
        """Get last fetch time for a company."""
        return self.company_last_fetch.get(company)

    def update_company_fetch(self, company: str):
        """Update last fetch time for a company."""
        self.company_last_fetch[company] = datetime.now()


class LinkedinFetcher:
    """Fetches LLM papers from LinkedIn with professional metrics."""

    def __init__(self, credentials: Optional[Dict[str, str]] = None):
        """
        Initialize LinkedIn fetcher with API or scraping credentials.

        Args:
            credentials: Dictionary with credentials. If None, uses environment variables.
                       For API: {'access_token': 'xxx'}
                       For scraping: {'email': 'xxx', 'password': 'xxx'}
        """
        self.config = get_queries_config().get('linkedin', {})
        self.mode = self._determine_mode(credentials)
        self.cache = LinkedInCache(
            max_daily=self.config.get('max_posts_per_day', 100)
        )
        self.anti_detection = AntiDetectionManager()

        # Initialize logger
        self.logger = logger.bind(component="linkedin_fetcher")

        # arXiv URL patterns
        self.arxiv_patterns = [
            r'https?://arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'https?://arxiv\.org/pdf/(\d{4}\.\d{4,5})\.pdf',
            r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'arXiv:(\d{4}\.\d{4,5})',
            r'https?://ar5iv\.org/abs/(\d{4}\.\d{4,5})',
        ]
        self.arxiv_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.arxiv_patterns]

        # Initialize client based on mode
        if self.mode == "api":
            self.api_client = self._init_api_client(credentials)
            self.browser = None
        else:
            self.api_client = None
            self.browser = None
            self.browser_context = None

        # Rate limiting
        self.base_delay = self.config.get('rate_limit_delay', 5)
        self.last_request_time = 0

        self.logger.info(
            f"LinkedIn fetcher initialized in {self.mode} mode",
            extra={
                "mode": self.mode,
                "tracked_companies": len(self.config.get('tracked_companies', [])),
                "hashtags": len(self.config.get('hashtags', [])),
                "rate_limit_delay": self.base_delay
            }
        )

    def _determine_mode(self, credentials: Optional[Dict[str, str]]) -> str:
        """Determine whether to use API or scraping mode."""
        # Check for API credentials
        if (credentials and 'access_token' in credentials) or os.getenv('LINKEDIN_ACCESS_TOKEN'):
            return "api"

        # Check for scraping credentials
        if (credentials and 'email' in credentials) or (os.getenv('LINKEDIN_EMAIL') and os.getenv('LINKEDIN_PASSWORD')):
            return "scraping"

        # Check config preference
        preferred = self.config.get('preferred_method', 'scraping')
        if preferred == 'api' and os.getenv('LINKEDIN_ACCESS_TOKEN'):
            return "api"

        # Default to scraping
        return "scraping"

    def _init_api_client(self, credentials: Optional[Dict[str, str]]) -> Optional[Linkedin]:
        """Initialize LinkedIn API client."""
        if not LINKEDIN_API_AVAILABLE:
            self.logger.warning("linkedin-api package not installed, falling back to scraping")
            self.mode = "scraping"
            return None

        token = (credentials.get('access_token') if credentials else None) or os.getenv('LINKEDIN_ACCESS_TOKEN')
        if not token:
            self.logger.warning("No LinkedIn access token, falling back to scraping")
            self.mode = "scraping"
            return None

        try:
            client = Linkedin(access_token=token)
            self.logger.info("LinkedIn API client initialized")
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize LinkedIn API: {e}")
            self.mode = "scraping"
            return None

    async def _init_browser(self) -> Tuple[Browser, BrowserContext]:
        """Initialize Playwright browser for scraping."""
        if not self.browser:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )

            # Create context with anti-detection measures
            self.browser_context = await self.browser.new_context(
                user_agent=self.anti_detection.get_random_user_agent(),
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York'
            )

            # Add stealth measures
            await self.browser_context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)

        return self.browser, self.browser_context

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        # Base delay with random jitter
        delay = self.base_delay + random.uniform(-2, 2)
        delay = max(delay, 1)  # Minimum 1 second delay

        if time_since_last < delay:
            sleep_time = delay - time_since_last
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text."""
        if not text:
            return None

        for pattern in self.arxiv_regex:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _extract_company(self, author_title: str) -> Optional[str]:
        """Extract company name from author title."""
        if not author_title:
            return None

        # Comprehensive company patterns
        companies = {
            # Major AI Labs
            'openai': 'OpenAI',
            'anthropic': 'Anthropic',
            'google deepmind': 'Google DeepMind',
            'deepmind': 'Google DeepMind',
            'xai': 'xAI',
            'x.ai': 'xAI',
            'meta ai': 'Meta AI',
            'meta': 'Meta AI',
            'mistral ai': 'Mistral AI',
            'mistral': 'Mistral AI',
            'deepseek': 'DeepSeek',
            'qwen': 'Qwen',
            'alibaba': 'Qwen',

            # Research Institutions
            'microsoft research': 'Microsoft Research',
            'ms research': 'Microsoft Research',
            'nvidia research': 'NVIDIA',
            'nvidia': 'NVIDIA',
            'ibm research': 'IBM Research',
            'ibm': 'IBM Research',
            'ai2': 'AI2',
            'allen institute': 'AI2',
            'hugging face': 'Hugging Face',
            'huggingface': 'Hugging Face',
            'cohere': 'Cohere',
            'minimax': 'Minimax',
            'moonshot ai': 'Kimi K2',
            'moonshot': 'Kimi K2',
            'kimi': 'Kimi K2',

            # Emerging AI Companies
            'harmonic': 'Harmonic',
            'axiom': 'Axiom',
            'deep cogito': 'Deep Cogito',
            'deepcogito': 'Deep Cogito',
            'z.ai': 'Z.AI',
            'zai': 'Z.AI',

            # Tech Giants AI Divisions
            'apple ml research': 'Apple',
            'apple': 'Apple',
            'amazon science': 'Amazon',
            'amazon': 'Amazon',
            'google brain': 'Google Brain',
            'baidu research': 'Baidu',
            'baidu': 'Baidu',
            'tencent ai': 'Tencent AI',
            'tencent': 'Tencent AI',
            'bytedance ai': 'ByteDance AI',
            'bytedance': 'ByteDance AI',

            # Additional Notable Companies
            'inflection ai': 'Inflection AI',
            'inflection': 'Inflection AI',
            'character.ai': 'Character.AI',
            'character': 'Character.AI',
            'stability ai': 'Stability AI',
            'stability': 'Stability AI',
            'together ai': 'Together AI',
            'together': 'Together AI',

            # Common variations and patterns
            'research scientist at': '',  # Will extract company after "at"
            'engineer at': '',
            'researcher at': '',
            'scientist at': '',
            'working at': '',
            '@': '',
        }

        title_lower = author_title.lower()

        # First try direct pattern matching
        for pattern, company in companies.items():
            if pattern and pattern in title_lower:
                return company if company else None

        # Try to extract company after common phrases
        # Pattern: "Research Scientist at [Company Name]"
        at_patterns = [
            r'(?:research scientist|ml engineer|ai researcher|research engineer|software engineer|data scientist|senior researcher|principal scientist|staff scientist)\s+at\s+(.*?)(?:,|\.|$)',
            r'at\s+([A-Za-z0-9\s&.-]+?)(?:,|\s*$)',
        ]

        for pattern in at_patterns:
            match = re.search(pattern, author_title, re.IGNORECASE)
            if match:
                company_candidate = match.group(1).strip()
                # Clean up the company name
                company_candidate = re.sub(r'\s+', ' ', company_candidate)

                # Check if this matches any of our known companies
                for known_pattern, known_company in companies.items():
                    if known_pattern and known_pattern in company_candidate.lower():
                        return known_company

                # Return the extracted company if it looks like a company name
                if len(company_candidate) > 2 and len(company_candidate) < 50:
                    # Capitalize properly
                    company_candidate = ' '.join(word.capitalize() for word in company_candidate.split())
                    return company_candidate

        return None

    def _is_verified_researcher(self, author_title: str, company: Optional[str]) -> bool:
        """Check if author appears to be a verified researcher."""
        if not author_title:
            return False

        # Research indicators
        research_indicators = [
            'research scientist', 'research engineer', 'ai researcher',
            'machine learning', 'phd', 'doctor', 'principal researcher',
            'staff research', 'senior research', 'research manager'
        ]

        title_lower = author_title.lower()
        has_research_title = any(indicator in title_lower for indicator in research_indicators)
        works_at_top_company = company in {
            # Top AI Labs
            'OpenAI', 'Anthropic', 'Google DeepMind', 'xAI', 'Meta AI',
            'Mistral AI', 'DeepSeek', 'Qwen',
            # Major Research Institutions
            'Microsoft Research', 'NVIDIA', 'IBM Research', 'AI2',
            # Notable Companies
            'Hugging Face', 'Cohere', 'Minimax', 'Kimi K2',
            # Tech Giants
            'Apple', 'Amazon', 'Google Brain', 'Baidu', 'Tencent AI'
        }

        return has_research_title or works_at_top_company

    def _calculate_professional_score(self, post: LinkedInPost) -> int:
        """
        Calculate professional engagement score.

        Weighting:
        - Comments × 5 (indicates thoughtful discussion)
        - Shares × 3 (indicates endorsement)
        - Likes × 1
        - Views × 0.001 (if available)
        """
        base_score = (
            post.likes_count * 1 +
            post.comments_count * 5 +
            post.shares_count * 3
        )

        if post.views_count:
            base_score += post.views_count * 0.001

        # Boost for verified researchers/companies
        if self._is_verified_researcher(post.author_title, post.company):
            base_score *= 1.5

        return int(base_score)

    def _calculate_engagement_rate(self, post: LinkedInPost) -> float:
        """Calculate engagement rate (impressions-based if available)."""
        if post.views_count and post.views_count > 0:
            total_engagement = post.likes_count + post.comments_count + post.shares_count
            return (total_engagement / post.views_count) * 100
        else:
            # Fallback: calculate based on followers if available
            return 0.0

    def _format_paper_dict(self, post: LinkedInPost) -> Dict[str, Any]:
        """Format LinkedIn post to match standardized paper dictionary."""
        arxiv_id = self._extract_arxiv_id(post.text)
        professional_score = self._calculate_professional_score(post)

        return {
            'id': f"linkedin:{post.id}",
            'title': None,  # To be filled by arXiv merge
            'abstract': None,  # To be filled by arXiv merge
            'authors': [post.author_name],
            'source': 'linkedin',
            'social_score': 0,  # LinkedIn uses professional_score instead
            'professional_score': professional_score,

            # LinkedIn-specific fields
            'linkedin_post_id': post.id,
            'linkedin_author_name': post.author_name,
            'linkedin_author_title': post.author_title,
            'linkedin_author_profile': post.author_profile_url,
            'linkedin_company': post.company,
            'linkedin_post_url': post.url,
            'linkedin_post_text': post.text,
            'linkedin_likes': post.likes_count,
            'linkedin_comments': post.comments_count,
            'linkedin_shares': post.shares_count,
            'linkedin_views': post.views_count,
            'linkedin_reactions': post.reactions,
            'linkedin_engagement_rate': self._calculate_engagement_rate(post),

            # Standard fields
            'fetch_date': datetime.now().date().isoformat(),
            'url': None,  # To be filled by arXiv merge
            'arxiv_id': arxiv_id,
            'published_date': post.published_at.date().isoformat() if post.published_at else None,
        }

    async def _fetch_with_retry(self, func, max_retries: int = 3, backoff_factor: float = 2.0):
        """Execute async function with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await func()
            except LinkedInRateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff_factor ** attempt
                self.logger.warning(f"Rate limit hit, retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
            except LinkedInBlockedError:
                self.logger.error("LinkedIn access blocked - switching mode")
                await self._switch_mode()
                raise
            except Exception as e:
                self.logger.error(f"Error fetching from LinkedIn: {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1)

    async def _switch_mode(self):
        """Switch between API and scraping modes."""
        if self.mode == "api":
            self.logger.info("Switching from API to scraping mode")
            self.mode = "scraping"
            self.api_client = None
            # Browser will be initialized lazily
        else:
            self.logger.info("Switching from scraping to API mode")
            self.mode = "api"
            if self.browser:
                await self.browser.close()
                self.browser = None
            self.api_client = self._init_api_client(None)

    async def _scrape_company_posts(self, company: Dict[str, Any], days: int) -> List[LinkedInPost]:
        """Scrape posts from a company page."""
        posts = []

        browser, context = await self._init_browser()
        page = await context.new_page()

        try:
            # Navigate to company page
            company_url = f"https://www.linkedin.com/company/{company['company_id']}/posts/"
            await page.goto(company_url, wait_until='networkidle')

            # Wait for posts to load
            await page.wait_for_selector('[data-urn^="post-"]', timeout=10000)

            # Scroll to load more posts
            for _ in range(3):  # Scroll 3 times to load more
                await page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(2)
                await self.anti_detection.simulate_human_behavior(page)

            # Extract posts
            post_elements = await page.query_all('[data-urn^="post-"]')
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            for element in post_elements[:20]:  # Limit to 20 posts per company
                try:
                    # Extract post data
                    post_data = await self._extract_post_from_element(element, page)
                    if post_data and post_data.published_at >= cutoff_date:
                        # Check for arXiv links
                        if self._extract_arxiv_id(post_data.text):
                            posts.append(post_data)

                        self.cache.increment_fetch_count()

                        # Check if we should pause
                        if self.cache.should_pause():
                            self.logger.info("Daily fetch limit reached, pausing")
                            break

                except Exception as e:
                    self.logger.warning(f"Error extracting post: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error scraping company {company['name']}: {e}")
        finally:
            await page.close()

        return posts

    async def _extract_post_from_element(self, element, page: Page) -> Optional[LinkedInPost]:
        """Extract post data from HTML element."""
        try:
            # Extract basic post info
            post_urn = await element.get_attribute('data-urn')
            post_id = post_urn.split(':')[-1] if post_urn else str(id(element))

            # Extract text content
            text_element = await element.query_selector('.feed-shared-text__text')
            text = await text_element.inner_text() if text_element else ""

            # Extract author info
            author_element = await element.query_selector('.feed-shared-actor__name')
            author_name = await author_element.inner_text() if author_element else ""

            author_title_element = await element.query_selector('.feed-shared-actor__description')
            author_title = await author_title_element.inner_text() if author_title_element else ""

            author_profile_element = await element.query_selector('a.feed-shared-actor__container-link')
            author_profile = await author_profile_element.get_attribute('href') if author_profile_element else ""

            # Extract engagement metrics
            likes_element = await element.query_selector('[aria-label*="like"]')
            likes_text = await likes_element.get_attribute('aria-label') if likes_element else "0"
            likes_count = self._parse_count(likes_text)

            comments_element = await element.query_selector('[aria-label*="comment"]')
            comments_text = await comments_element.get_attribute('aria-label') if comments_element else "0"
            comments_count = self._parse_count(comments_text)

            shares_element = await element.query_selector('[aria-label*="share"]')
            shares_text = await shares_element.get_attribute('aria-label') if shares_element else "0"
            shares_count = self._parse_count(shares_text)

            # Extract post URL
            post_link_element = await element.query_selector('a[href*="/activity/"]')
            post_url = await post_link_element.get_attribute('href') if post_link_element else ""
            if post_url and not post_url.startswith('http'):
                post_url = f"https://www.linkedin.com{post_url}"

            # Extract published date (rough estimation)
            time_element = await element.query_selector('.feed-shared-timeago')
            time_text = await time_element.inner_text() if time_element else ""
            published_at = self._parse_time_ago(time_text)

            # Extract company
            company = self._extract_company(author_title)

            return LinkedInPost(
                id=post_id,
                author_name=author_name,
                author_title=author_title,
                author_profile_url=author_profile,
                company=company,
                text=text,
                url=post_url,
                likes_count=likes_count,
                comments_count=comments_count,
                shares_count=shares_count,
                views_count=None,  # Not visible in scraping
                published_at=published_at
            )

        except Exception as e:
            self.logger.warning(f"Error extracting post data: {e}")
            return None

    def _parse_count(self, text: str) -> int:
        """Parse count from text like '15 likes' or '1.2K reactions'."""
        if not text:
            return 0

        # Extract number from text
        match = re.search(r'(\d+\.?\d*)([KkMm]?)', text)
        if not match:
            return 0

        number = float(match.group(1))
        suffix = match.group(2).upper()

        if suffix == 'K':
            number *= 1000
        elif suffix == 'M':
            number *= 1000000

        return int(number)

    def _parse_time_ago(self, time_text: str) -> datetime:
        """Parse time ago text to datetime."""
        if not time_text:
            return datetime.now(timezone.utc)

        now = datetime.now(timezone.utc)

        # Patterns like "2d", "3h", "1w", "5m"
        match = re.search(r'(\d+)([dwmyh])', time_text.lower())
        if not match:
            return now

        value = int(match.group(1))
        unit = match.group(2)

        if unit == 'd':
            return now - timedelta(days=value)
        elif unit == 'w':
            return now - timedelta(weeks=value)
        elif unit == 'm':
            return now - timedelta(minutes=value)
        elif unit == 'h':
            return now - timedelta(hours=value)
        elif unit == 'y':
            return now - timedelta(days=value * 365)

        return now

    async def fetch_from_companies(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent posts from tracked company pages.

        Args:
            days: Number of days to look back

        Returns:
            List of papers found in company posts
        """
        companies = self.config.get('tracked_companies', [])
        if not companies:
            self.logger.warning("No tracked companies configured")
            return []

        self.logger.info(f"Fetching posts from {len(companies)} tracked companies")

        all_papers = []

        for company in companies:
            if self.cache.should_pause():
                self.logger.info("Pausing due to rate limits")
                break

            self._enforce_rate_limit()

            try:
                self.logger.info(f"Fetching posts from {company['name']}")

                if self.mode == "api" and self.api_client:
                    # API mode implementation would go here
                    posts = await self._fetch_company_posts_api(company, days)
                else:
                    # Scraping mode
                    posts = await self._scrape_company_posts(company, days)

                # Convert to paper format
                for post in posts:
                    if not self.cache.is_post_fetched(post.id):
                        paper = self._format_paper_dict(post)
                        if paper.get('arxiv_id'):
                            all_papers.append(paper)
                            self.cache.add_post(post.id)

                self.cache.update_company_fetch(company['name'])

            except Exception as e:
                self.logger.error(f"Error fetching from {company['name']}: {e}")
                continue

        self.logger.info(f"Found {len(all_papers)} papers from company posts")
        return all_papers

    async def fetch_by_hashtags(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent posts containing tracked hashtags.

        Args:
            days: Number of days to look back

        Returns:
            List of papers found in hashtag posts
        """
        hashtags = self.config.get('hashtags', [])
        if not hashtags:
            return []

        self.logger.info(f"Fetching posts for {len(hashtags)} hashtags")

        all_papers = []

        for hashtag in hashtags:
            if self.cache.should_pause():
                break

            self._enforce_rate_limit()

            try:
                # Implement hashtag search
                posts = await self._scrape_hashtag_posts(hashtag, days)

                for post in posts:
                    if not self.cache.is_post_fetched(post.id):
                        paper = self._format_paper_dict(post)
                        if paper.get('arxiv_id'):
                            all_papers.append(paper)
                            self.cache.add_post(post.id)

            except Exception as e:
                self.logger.error(f"Error fetching hashtag {hashtag}: {e}")
                continue

        self.logger.info(f"Found {len(all_papers)} papers from hashtag posts")
        return all_papers

    async def _scrape_hashtag_posts(self, hashtag: str, days: int) -> List[LinkedInPost]:
        """Scrape posts containing a specific hashtag."""
        # Implementation for hashtag scraping
        # This would navigate to LinkedIn hashtag search and extract posts
        return []

    async def _fetch_company_posts_api(self, company: Dict[str, Any], days: int) -> List[LinkedInPost]:
        """Fetch company posts using LinkedIn API."""
        # Implementation for API-based fetching
        return []

    async def fetch_recent_papers(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent papers from all LinkedIn sources.

        Args:
            days: Number of days to look back

        Returns:
            List of all papers found from LinkedIn sources
        """
        self.logger.info(f"Starting LinkedIn fetch for last {days} days")

        # Fetch from companies
        company_papers = await self.fetch_from_companies(days)

        # Fetch from hashtags
        hashtag_papers = await self.fetch_by_hashtags(days)

        # Combine and deduplicate
        all_papers = company_papers + hashtag_papers
        seen_ids = set()
        deduplicated = []

        for paper in all_papers:
            if paper['id'] not in seen_ids:
                deduplicated.append(paper)
                seen_ids.add(paper['id'])

        self.logger.info(
            f"LinkedIn fetch complete: {len(deduplicated)} unique papers found",
            extra={
                "company_papers": len(company_papers),
                "hashtag_papers": len(hashtag_papers),
                "total_unique": len(deduplicated)
            }
        )

        return deduplicated

    def get_stats(self) -> Dict[str, Any]:
        """Get fetching statistics."""
        return {
            "mode": self.mode,
            "daily_fetch_count": self.cache.daily_fetch_count,
            "max_daily": self.cache.max_daily,
            "seen_posts_count": len(self.cache.seen_posts),
            "tracked_companies": len(self.config.get('tracked_companies', [])),
            "hashtags": len(self.config.get('hashtags', [])),
            "rate_limit_delay": self.base_delay
        }

    async def close(self):
        """Close resources."""
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.browser_context = None

    # Synchronous wrapper for compatibility with existing code
    def fetch_papers(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for fetch_recent_papers.

        Args:
            max_results: Maximum number of papers to fetch

        Returns:
            List of papers found from LinkedIn
        """
        # Run the async method in an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            papers = loop.run_until_complete(self.fetch_recent_papers(days=7))
            return papers[:max_results]
        finally:
            loop.close()
            # Clean up browser if needed
            if self.browser:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.close())
                loop.close()

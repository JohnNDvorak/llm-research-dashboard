"""
Paper Deduplicator - Intelligent deduplication across multiple sources.

This module provides the PaperDeduplicator class that handles deduplication
of papers from multiple sources (arXiv, Twitter, LinkedIn) with intelligent
merging of metadata and scoring.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from rapidfuzz import fuzz
from src.utils.config_loader import get_queries_config
from src.utils.logger import logger


class PaperDeduplicator:
    """
    Deduplicate papers from multiple sources with intelligent merging.

    This class handles:
    - Primary matching by arXiv ID (exact match)
    - Secondary matching by title similarity (>90% by default)
    - Intelligent metadata merging (max scores, merge sources)
    - Combined score calculation: (social*0.4) + (prof*0.6) + (recency*0.3)

    Example:
        >>> deduplicator = PaperDeduplicator()
        >>> papers = [
        ...     {"id": "arxiv:2401.00001", "title": "DPO", "source": "arxiv"},
        ...     {"id": "twitter_123", "title": "DPO", "source": "twitter", "social_score": 100}
        ... ]
        >>> unique = deduplicator.deduplicate(papers)
        >>> len(unique)
        1
        >>> unique[0]["social_score"]
        100
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PaperDeduplicator.

        Args:
            config: Optional configuration dictionary. If not provided,
                   loads from config/queries.yaml
        """
        if config is None:
            full_config = get_queries_config()
            config = full_config.get('deduplication', {})

        self.config = {
            'use_arxiv_id': config.get('use_arxiv_id', True),
            'title_similarity_threshold': config.get('title_similarity_threshold', 0.90),
            'merge_strategy': config.get('merge_strategy', {
                'social_score': 'max',
                'professional_score': 'max',
                'sources': 'merge'
            })
        }

        self.logger = logger.bind(component="paper_deduplicator")
        self.logger.debug("PaperDeduplicator initialized", config=self.config)

    def deduplicate(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate papers and merge metadata from multiple sources.

        Algorithm:
        1. Group papers by arXiv ID (primary match)
        2. Group papers by title similarity (secondary match)
        3. Merge duplicate groups
        4. Calculate combined scores
        5. Return deduplicated list

        Args:
            papers: List of paper dictionaries from various sources

        Returns:
            Deduplicated list of papers with merged metadata and combined scores
        """
        if not papers:
            return []

        self.logger.info(f"Deduplicating {len(papers)} papers")

        # Step 1: Group papers by arXiv ID
        arxiv_groups = self._group_by_arxiv_id(papers)

        # Step 2: Group remaining papers by title similarity
        remaining_papers = [p for p in papers if self._extract_arxiv_id(p) is None]
        title_groups = self._group_by_title_similarity(remaining_papers)

        # Step 3: Merge arXiv groups with title groups if they have similar titles
        all_groups = self._merge_arxiv_and_title_groups(arxiv_groups, title_groups)

        # Step 4: Merge each group and calculate scores
        deduplicated = []
        for group in all_groups:
            if len(group) == 1:
                merged = group[0]
            else:
                merged = self._merge_papers(group)

            # Calculate combined score
            merged['combined_score'] = self._calculate_combined_score(merged)
            deduplicated.append(merged)

        duplicates_removed = len(papers) - len(deduplicated)
        self.logger.info(
            f"Deduplication complete: {len(deduplicated)} unique papers "
            f"({duplicates_removed} duplicates removed)"
        )

        return deduplicated

    def _group_by_arxiv_id(self, papers: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group papers by arXiv ID.

        Args:
            papers: List of paper dictionaries

        Returns:
            Dictionary mapping arXiv ID to list of papers with that ID
        """
        groups = {}
        for paper in papers:
            arxiv_id = self._extract_arxiv_id(paper)
            if arxiv_id:
                if arxiv_id not in groups:
                    groups[arxiv_id] = []
                groups[arxiv_id].append(paper)
        return groups

    def _merge_arxiv_and_title_groups(
        self,
        arxiv_groups: Dict[str, List[Dict[str, Any]]],
        title_groups: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Merge arXiv-grouped papers with title-grouped papers if they have similar titles.

        This handles the case where an arXiv paper is mentioned on Twitter/LinkedIn
        with the same title but without the arXiv ID.

        Args:
            arxiv_groups: Dictionary of arXiv ID -> list of papers
            title_groups: List of title-similar paper groups

        Returns:
            Combined list of groups with similar-titled groups merged
        """
        all_groups = list(arxiv_groups.values())
        threshold = self.config['title_similarity_threshold']

        for title_group in title_groups:
            if not title_group:
                continue

            title_group_title = title_group[0].get('title', '')
            if not title_group_title:
                # No title, keep separate
                all_groups.append(title_group)
                continue

            # Check if this title group should merge with any arXiv group
            merged = False
            for arxiv_group in all_groups:
                arxiv_group_title = arxiv_group[0].get('title', '')
                if arxiv_group_title:
                    similarity = self._calculate_title_similarity(title_group_title, arxiv_group_title)
                    if similarity >= threshold:
                        # Merge title group into arXiv group
                        arxiv_group.extend(title_group)
                        merged = True
                        break

            if not merged:
                # No matching arXiv group, keep as separate group
                all_groups.append(title_group)

        return all_groups

    def _group_by_title_similarity(self, papers: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group papers by title similarity (for papers without arXiv IDs).

        Uses a greedy approach: each paper is compared to existing groups,
        and added to the first group where title similarity exceeds threshold.

        Args:
            papers: List of paper dictionaries without arXiv IDs

        Returns:
            List of groups, where each group is a list of similar papers
        """
        if not papers:
            return []

        groups = []
        threshold = self.config['title_similarity_threshold']

        for paper in papers:
            title = paper.get('title', '')
            if not title:
                # Papers without titles get their own group
                groups.append([paper])
                continue

            # Try to find a matching group
            added = False
            for group in groups:
                # Compare with first paper in group
                group_title = group[0].get('title', '')
                if group_title:
                    similarity = self._calculate_title_similarity(title, group_title)
                    if similarity >= threshold:
                        group.append(paper)
                        added = True
                        break

            if not added:
                # Create new group
                groups.append([paper])

        return groups

    def _extract_arxiv_id(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Extract arXiv ID from various formats.

        Handles:
        - arxiv:2401.00001
        - 2401.00001
        - 2401.00001v2 (with version)
        - https://arxiv.org/abs/2401.00001
        - https://arxiv.org/pdf/2401.00001.pdf
        - Direct arxiv_id field

        Args:
            paper: Paper dictionary

        Returns:
            Normalized arXiv ID (YYMM.NNNNN) or None if not found
        """
        # Pattern for arXiv ID: YYMM.NNNNN or YYMM.NNNNNvN
        arxiv_pattern = r'(\d{4}\.\d{4,5})(?:v\d+)?'

        # First check explicit arxiv_id field (highest priority)
        arxiv_id = paper.get('arxiv_id')
        if arxiv_id:
            # Extract the ID part if it contains a URL or other text
            match = re.search(arxiv_pattern, str(arxiv_id))
            if match:
                return match.group(1)

        # Check ID field
        paper_id = paper.get('id', '')
        if paper_id:
            match = re.search(arxiv_pattern, str(paper_id))
            if match:
                return match.group(1)

        # Check URL field
        url = paper.get('url', '')
        if url and 'arxiv.org' in url:
            match = re.search(arxiv_pattern, url)
            if match:
                return match.group(1)

        # Check PDF URL field
        pdf_url = paper.get('pdf_url', '')
        if pdf_url and 'arxiv.org' in pdf_url:
            match = re.search(arxiv_pattern, pdf_url)
            if match:
                return match.group(1)

        return None

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles using Levenshtein ratio.

        Normalizes titles by:
        - Converting to lowercase
        - Removing extra whitespace
        - Removing punctuation

        Args:
            title1: First title string
            title2: Second title string

        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not title1 and not title2:
            return 1.0  # Both empty = identical

        if not title1 or not title2:
            return 0.0  # One empty, one not = different

        # Normalize titles
        def normalize(text: str) -> str:
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # Remove extra whitespace
            text = ' '.join(text.split())
            return text

        norm1 = normalize(title1)
        norm2 = normalize(title2)

        # Calculate Levenshtein ratio (0-100) and convert to 0-1
        ratio = fuzz.ratio(norm1, norm2) / 100.0

        return ratio

    def _merge_papers(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple papers into a single paper with combined metadata.

        Merge strategies:
        - ID: Keep arXiv ID if available, else generate hash
        - Title: Keep longest version
        - Abstract: Keep longest version
        - Authors: Merge and deduplicate
        - URLs: Keep all unique URLs
        - Dates: Earliest published_date, latest fetch_date
        - Scores: Maximum values (social_score, professional_score)
        - Sources: Merge into list

        Args:
            papers: List of duplicate papers to merge

        Returns:
            Single merged paper dictionary
        """
        if not papers:
            return {}

        if len(papers) == 1:
            return papers[0].copy()

        self.logger.debug(f"Merging {len(papers)} duplicate papers")

        merged = {}

        # ID: Prefer arXiv ID, otherwise use first paper's ID or generate hash
        arxiv_id = None
        for paper in papers:
            extracted = self._extract_arxiv_id(paper)
            if extracted:
                arxiv_id = f"arxiv:{extracted}"
                break

        if arxiv_id:
            merged['id'] = arxiv_id
        elif papers[0].get('id'):
            merged['id'] = papers[0]['id']
        else:
            # Generate ID from title hash
            title = papers[0].get('title', 'untitled')
            merged['id'] = f"hash_{hashlib.md5(title.encode()).hexdigest()[:12]}"

        # Title: Keep longest
        titles = [p.get('title', '') for p in papers if p.get('title')]
        merged['title'] = max(titles, key=len) if titles else None

        # Abstract: Keep longest
        abstracts = [p.get('abstract', '') for p in papers if p.get('abstract')]
        merged['abstract'] = max(abstracts, key=len) if abstracts else None

        # Authors: Merge and deduplicate
        all_authors = []
        for paper in papers:
            authors = paper.get('authors', [])
            if isinstance(authors, list):
                all_authors.extend(authors)
            elif isinstance(authors, str):
                all_authors.append(authors)

        # Deduplicate authors while preserving order
        seen = set()
        unique_authors = []
        for author in all_authors:
            if author and author not in seen:
                seen.add(author)
                unique_authors.append(author)

        if unique_authors:
            merged['authors'] = unique_authors

        # URLs: Keep all unique URLs (prefer arXiv)
        urls = [p.get('url', '') for p in papers if p.get('url')]
        if urls:
            # Prefer arXiv URL if available
            arxiv_urls = [u for u in urls if 'arxiv.org' in u]
            merged['url'] = arxiv_urls[0] if arxiv_urls else urls[0]

        pdf_urls = [p.get('pdf_url', '') for p in papers if p.get('pdf_url')]
        if pdf_urls:
            merged['pdf_url'] = pdf_urls[0]

        # Dates: Earliest published_date, latest fetch_date
        pub_dates = [p.get('published_date') for p in papers if p.get('published_date')]
        if pub_dates:
            # Keep earliest publication date
            try:
                sorted_dates = sorted(pub_dates)
                merged['published_date'] = sorted_dates[0]
            except (TypeError, ValueError):
                merged['published_date'] = pub_dates[0]

        fetch_dates = [p.get('fetch_date') for p in papers if p.get('fetch_date')]
        if fetch_dates:
            # Keep latest fetch date
            try:
                sorted_dates = sorted(fetch_dates)
                merged['fetch_date'] = sorted_dates[-1]
            except (TypeError, ValueError):
                merged['fetch_date'] = fetch_dates[-1]

        # Scores: Maximum values
        social_scores = [p.get('social_score', 0) for p in papers]
        merged['social_score'] = max(social_scores) if social_scores else 0

        prof_scores = [p.get('professional_score', 0) for p in papers]
        merged['professional_score'] = max(prof_scores) if prof_scores else 0

        # Sources: Merge into list
        sources = []
        for paper in papers:
            source = paper.get('source', '')
            if source and source not in sources:
                sources.append(source)

        if len(sources) == 1:
            merged['source'] = sources[0]
        elif len(sources) > 1:
            merged['source'] = sources

        # LinkedIn-specific fields (if any paper has them)
        for field in ['linkedin_engagement', 'linkedin_company', 'linkedin_author_title', 'linkedin_post_url']:
            values = [p.get(field) for p in papers if p.get(field)]
            if values:
                merged[field] = values[0]  # Keep first non-null value

        return merged

    def _calculate_combined_score(self, paper: Dict[str, Any]) -> float:
        """
        Calculate combined score from social, professional, and recency.

        Formula: (social_score * 0.4) + (professional_score * 0.6) + (recency * 0.3)

        Recency calculation:
        - Papers from today: 100 points
        - Papers from 1 year ago: 0 points
        - Linear interpolation in between

        Args:
            paper: Paper dictionary with score fields

        Returns:
            Combined score as a float
        """
        # Get scores (default to 0 if missing)
        social_score = paper.get('social_score', 0) or 0
        prof_score = paper.get('professional_score', 0) or 0

        # Calculate recency score
        recency_score = 0
        pub_date_str = paper.get('published_date')

        if pub_date_str:
            try:
                # Parse date (handle various formats)
                if isinstance(pub_date_str, datetime):
                    pub_date = pub_date_str
                elif 'T' in str(pub_date_str):
                    pub_date = datetime.fromisoformat(str(pub_date_str).replace('Z', '+00:00'))
                else:
                    pub_date = datetime.strptime(str(pub_date_str), '%Y-%m-%d')

                # Calculate days since publication
                days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days

                # Recency score: 100 for today, 0 for 365+ days old
                if days_old < 0:
                    recency_score = 100  # Future date = max score
                elif days_old >= 365:
                    recency_score = 0
                else:
                    recency_score = 100 * (1 - days_old / 365)

            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse date '{pub_date_str}': {e}")
                recency_score = 50  # Default to middle if parse fails

        # Combined score formula
        combined = (social_score * 0.4) + (prof_score * 0.6) + (recency_score * 0.3)

        return round(combined, 2)


def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function for backward compatibility.

    This function is kept for compatibility with existing test code.
    It creates a PaperDeduplicator instance and calls deduplicate().

    Args:
        papers: List of papers from various sources

    Returns:
        Deduplicated list of papers
    """
    deduplicator = PaperDeduplicator()
    return deduplicator.deduplicate(papers)

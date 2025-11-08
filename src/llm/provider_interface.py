"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class LLMProvider(ABC):
    """Base interface that all LLM providers must implement."""

    @abstractmethod
    def analyze_paper(self, abstract: str, title: str) -> Dict[str, Any]:
        """
        Analyze a paper and categorize it into pipeline stages.

        Args:
            abstract: Paper abstract text
            title: Paper title

        Returns:
            Dict containing stages, summary, key_insights
        """
        pass

    @abstractmethod
    def get_cost_per_token(self) -> Dict[str, float]:
        """
        Get the cost per token for this provider.

        Returns:
            Dict with 'input' and 'output' costs per token
        """
        pass

    @abstractmethod
    def get_rate_limits(self) -> Dict[str, int]:
        """
        Get rate limits for this provider.

        Returns:
            Dict with rate limit information
        """
        pass

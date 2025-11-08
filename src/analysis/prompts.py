"""LLM prompts for paper categorization."""

from typing import Dict, List


def get_stage_categorization_prompt(
    title: str,
    abstract: str,
    stages: List[Dict]
) -> str:
    """
    Generate prompt for categorizing paper into pipeline stages.

    Args:
        title: Paper title
        abstract: Paper abstract
        stages: List of pipeline stages with keywords

    Returns:
        Formatted prompt for LLM
    """
    # TODO: Implement prompt generation
    pass

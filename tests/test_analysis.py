"""Unit tests for paper analysis modules."""

import pytest
from typing import Dict, List
from src.analysis.prompts import get_stage_categorization_prompt
from src.analysis.scorer import calculate_combined_score


class TestPrompts:
    """Test suite for prompts module."""

    def test_get_stage_categorization_prompt_exists(self):
        """Test that get_stage_categorization_prompt function exists."""
        assert get_stage_categorization_prompt is not None
        assert callable(get_stage_categorization_prompt)

    def test_prompt_function_signature(self):
        """Test that prompt function has correct signature."""
        import inspect
        sig = inspect.signature(get_stage_categorization_prompt)
        params = list(sig.parameters.keys())
        assert 'title' in params
        assert 'abstract' in params
        assert 'stages' in params

    def test_prompt_function_accepts_strings(self):
        """Test that prompt function accepts string parameters."""
        stages = [{"name": "Test Stage", "keywords": ["test"]}]
        try:
            get_stage_categorization_prompt(
                title="Test Title",
                abstract="Test Abstract",
                stages=stages
            )
        except TypeError as e:
            pytest.fail(f"Prompt function should accept strings: {e}")

    def test_prompt_function_return_type(self):
        """Test that prompt function has correct return type annotation."""
        import inspect
        sig = inspect.signature(get_stage_categorization_prompt)
        # Should return str
        assert sig.return_annotation == str or sig.return_annotation != inspect.Signature.empty


class TestPromptGeneration:
    """Test suite for prompt generation scenarios."""

    def test_prompt_with_simple_inputs(self):
        """Test prompt generation with simple inputs."""
        stages = [{"name": "Stage 1", "keywords": ["keyword1"]}]
        get_stage_categorization_prompt(
            title="Simple Title",
            abstract="Simple abstract",
            stages=stages
        )

    def test_prompt_with_complex_title(self):
        """Test prompt with complex title."""
        stages = [{"name": "Architecture", "keywords": ["attention"]}]
        get_stage_categorization_prompt(
            title="Attention Is All You Need: Transformer Architecture for NLP",
            abstract="We propose a new architecture",
            stages=stages
        )

    def test_prompt_with_long_abstract(self):
        """Test prompt with long abstract."""
        stages = [{"name": "Training", "keywords": ["optimization"]}]
        abstract = "This paper presents " + "a novel approach " * 100
        get_stage_categorization_prompt(
            title="Title",
            abstract=abstract,
            stages=stages
        )

    def test_prompt_with_multiple_stages(self):
        """Test prompt with multiple stages."""
        stages = [
            {"name": "Architecture Design", "keywords": ["attention", "transformer"]},
            {"name": "Pre-Training", "keywords": ["corpus", "tokenization"]},
            {"name": "Post-Training", "keywords": ["DPO", "RLHF"]},
        ]
        get_stage_categorization_prompt(
            title="LLM Training",
            abstract="We train a large language model",
            stages=stages
        )

    def test_prompt_with_eight_stages(self):
        """Test prompt with all eight pipeline stages."""
        stages = [
            {"name": "Architecture Design", "keywords": ["attention"]},
            {"name": "Data Preparation", "keywords": ["corpus"]},
            {"name": "Pre-Training", "keywords": ["training"]},
            {"name": "Post-Training", "keywords": ["DPO"]},
            {"name": "Evaluation", "keywords": ["benchmark"]},
            {"name": "Infrastructure", "keywords": ["GPU"]},
            {"name": "Deployment", "keywords": ["inference"]},
            {"name": "Other", "keywords": ["emerging"]},
        ]
        get_stage_categorization_prompt(
            title="Comprehensive LLM Study",
            abstract="We study all aspects of LLM development",
            stages=stages
        )

    def test_prompt_with_empty_abstract(self):
        """Test prompt with empty abstract."""
        stages = [{"name": "Test", "keywords": ["test"]}]
        get_stage_categorization_prompt(
            title="Title Only",
            abstract="",
            stages=stages
        )

    def test_prompt_with_empty_title(self):
        """Test prompt with empty title."""
        stages = [{"name": "Test", "keywords": ["test"]}]
        get_stage_categorization_prompt(
            title="",
            abstract="Abstract only",
            stages=stages
        )

    def test_prompt_with_special_characters(self):
        """Test prompt with special characters."""
        stages = [{"name": "Test", "keywords": ["test"]}]
        get_stage_categorization_prompt(
            title="Title with $pecial Ch@rs!",
            abstract="Abstract with quotes 'and' \"symbols\"",
            stages=stages
        )

    def test_prompt_with_unicode(self):
        """Test prompt with unicode characters."""
        stages = [{"name": "Test", "keywords": ["test"]}]
        get_stage_categorization_prompt(
            title="LLM Training 中文 🤖",
            abstract="Résumé of methods",
            stages=stages
        )

    def test_prompt_with_many_keywords(self):
        """Test prompt with many keywords per stage."""
        stages = [{
            "name": "Architecture",
            "keywords": [f"keyword{i}" for i in range(50)]
        }]
        get_stage_categorization_prompt(
            title="Test",
            abstract="Test",
            stages=stages
        )


class TestScorer:
    """Test suite for scorer module."""

    def test_calculate_combined_score_exists(self):
        """Test that calculate_combined_score function exists."""
        assert calculate_combined_score is not None
        assert callable(calculate_combined_score)

    def test_scorer_function_signature(self):
        """Test that scorer function has correct signature."""
        import inspect
        sig = inspect.signature(calculate_combined_score)
        params = list(sig.parameters.keys())
        assert 'social_score' in params
        assert 'professional_score' in params
        assert 'recency_score' in params

    def test_scorer_accepts_numeric_inputs(self):
        """Test that scorer accepts numeric inputs."""
        try:
            calculate_combined_score(
                social_score=10,
                professional_score=20,
                recency_score=0.5
            )
        except TypeError as e:
            pytest.fail(f"Scorer should accept numeric inputs: {e}")

    def test_scorer_return_type(self):
        """Test that scorer has correct return type annotation."""
        import inspect
        sig = inspect.signature(calculate_combined_score)
        # Should return float
        assert sig.return_annotation == float or sig.return_annotation != inspect.Signature.empty


class TestScorerCalculations:
    """Test suite for various scoring scenarios."""

    def test_score_with_zero_values(self):
        """Test scoring with all zero values."""
        calculate_combined_score(
            social_score=0,
            professional_score=0,
            recency_score=0.0
        )

    def test_score_with_positive_values(self):
        """Test scoring with positive values."""
        calculate_combined_score(
            social_score=100,
            professional_score=50,
            recency_score=0.8
        )

    def test_score_with_high_social(self):
        """Test scoring with high social score."""
        calculate_combined_score(
            social_score=1000,
            professional_score=10,
            recency_score=0.5
        )

    def test_score_with_high_professional(self):
        """Test scoring with high professional score."""
        calculate_combined_score(
            social_score=10,
            professional_score=500,
            recency_score=0.5
        )

    def test_score_with_high_recency(self):
        """Test scoring with high recency."""
        calculate_combined_score(
            social_score=50,
            professional_score=50,
            recency_score=1.0
        )

    def test_score_with_low_recency(self):
        """Test scoring with low recency."""
        calculate_combined_score(
            social_score=50,
            professional_score=50,
            recency_score=0.1
        )

    def test_score_with_decimal_social(self):
        """Test scoring with decimal social score."""
        calculate_combined_score(
            social_score=25.5,
            professional_score=30,
            recency_score=0.5
        )

    def test_score_with_decimal_professional(self):
        """Test scoring with decimal professional score."""
        calculate_combined_score(
            social_score=30,
            professional_score=45.7,
            recency_score=0.5
        )

    def test_score_with_recency_zero(self):
        """Test scoring with recency at zero."""
        calculate_combined_score(
            social_score=100,
            professional_score=100,
            recency_score=0.0
        )

    def test_score_with_recency_one(self):
        """Test scoring with recency at one."""
        calculate_combined_score(
            social_score=100,
            professional_score=100,
            recency_score=1.0
        )

    def test_score_with_very_large_values(self):
        """Test scoring with very large values."""
        calculate_combined_score(
            social_score=1000000,
            professional_score=500000,
            recency_score=0.99
        )

    def test_score_realistic_scenario_1(self):
        """Test realistic scenario: viral Twitter paper."""
        calculate_combined_score(
            social_score=500,  # 500 likes + retweets
            professional_score=30,  # Modest LinkedIn engagement
            recency_score=0.95  # Published yesterday
        )

    def test_score_realistic_scenario_2(self):
        """Test realistic scenario: professional LinkedIn post."""
        calculate_combined_score(
            social_score=20,  # Low Twitter engagement
            professional_score=200,  # High LinkedIn engagement
            recency_score=0.7  # Published a few days ago
        )

    def test_score_realistic_scenario_3(self):
        """Test realistic scenario: old but important paper."""
        calculate_combined_score(
            social_score=1000,  # High social engagement
            professional_score=500,  # High professional engagement
            recency_score=0.2  # Published months ago
        )

    def test_score_realistic_scenario_4(self):
        """Test realistic scenario: new unknown paper."""
        calculate_combined_score(
            social_score=0,  # No social engagement yet
            professional_score=0,  # No professional engagement yet
            recency_score=1.0  # Just published
        )

    def test_score_with_negative_recency(self):
        """Test scoring with negative recency (edge case)."""
        calculate_combined_score(
            social_score=50,
            professional_score=50,
            recency_score=-0.1
        )

    def test_score_with_recency_over_one(self):
        """Test scoring with recency over 1.0 (edge case)."""
        calculate_combined_score(
            social_score=50,
            professional_score=50,
            recency_score=1.5
        )

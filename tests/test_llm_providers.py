"""Unit tests for LLM provider modules."""

import pytest
from abc import ABC
from typing import Dict, Any
from src.llm.provider_interface import LLMProvider
from src.llm.provider_factory import ProviderFactory


class TestLLMProviderInterface:
    """Test suite for LLMProvider abstract base class."""

    def test_provider_interface_is_abstract(self):
        """Test that LLMProvider is an abstract base class."""
        assert issubclass(LLMProvider, ABC)

    def test_provider_interface_cannot_be_instantiated(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_provider_interface_has_analyze_paper(self):
        """Test that LLMProvider defines analyze_paper method."""
        assert hasattr(LLMProvider, 'analyze_paper')

    def test_provider_interface_has_get_cost_per_token(self):
        """Test that LLMProvider defines get_cost_per_token method."""
        assert hasattr(LLMProvider, 'get_cost_per_token')

    def test_provider_interface_has_get_rate_limits(self):
        """Test that LLMProvider defines get_rate_limits method."""
        assert hasattr(LLMProvider, 'get_rate_limits')

    def test_analyze_paper_signature(self):
        """Test that analyze_paper has correct signature."""
        import inspect
        sig = inspect.signature(LLMProvider.analyze_paper)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'abstract' in params
        assert 'title' in params

    def test_get_cost_per_token_signature(self):
        """Test that get_cost_per_token has correct signature."""
        import inspect
        sig = inspect.signature(LLMProvider.get_cost_per_token)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_get_rate_limits_signature(self):
        """Test that get_rate_limits has correct signature."""
        import inspect
        sig = inspect.signature(LLMProvider.get_rate_limits)
        params = list(sig.parameters.keys())
        assert 'self' in params


class MockLLMProvider(LLMProvider):
    """Mock implementation of LLMProvider for testing."""

    def analyze_paper(self, abstract: str, title: str) -> Dict[str, Any]:
        """Mock implementation of analyze_paper."""
        return {
            "stages": ["Architecture Design"],
            "summary": "Test summary",
            "key_insights": ["Test insight"]
        }

    def get_cost_per_token(self) -> Dict[str, float]:
        """Mock implementation of get_cost_per_token."""
        return {
            "input": 0.0001,
            "output": 0.0002
        }

    def get_rate_limits(self) -> Dict[str, int]:
        """Mock implementation of get_rate_limits."""
        return {
            "requests_per_minute": 60,
            "tokens_per_minute": 100000
        }


class TestMockLLMProvider:
    """Test suite for mock LLM provider implementation."""

    def test_mock_provider_can_be_instantiated(self):
        """Test that mock provider can be instantiated."""
        provider = MockLLMProvider()
        assert provider is not None

    def test_mock_provider_is_llm_provider(self):
        """Test that mock provider is instance of LLMProvider."""
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_mock_analyze_paper_returns_dict(self):
        """Test that analyze_paper returns dictionary."""
        provider = MockLLMProvider()
        result = provider.analyze_paper("Test abstract", "Test title")
        assert isinstance(result, dict)

    def test_mock_analyze_paper_has_required_keys(self):
        """Test that analyze_paper returns required keys."""
        provider = MockLLMProvider()
        result = provider.analyze_paper("Test abstract", "Test title")
        assert "stages" in result
        assert "summary" in result
        assert "key_insights" in result

    def test_mock_get_cost_per_token_returns_dict(self):
        """Test that get_cost_per_token returns dictionary."""
        provider = MockLLMProvider()
        result = provider.get_cost_per_token()
        assert isinstance(result, dict)

    def test_mock_get_cost_per_token_has_required_keys(self):
        """Test that get_cost_per_token returns input and output costs."""
        provider = MockLLMProvider()
        result = provider.get_cost_per_token()
        assert "input" in result
        assert "output" in result

    def test_mock_get_rate_limits_returns_dict(self):
        """Test that get_rate_limits returns dictionary."""
        provider = MockLLMProvider()
        result = provider.get_rate_limits()
        assert isinstance(result, dict)


class TestProviderFactory:
    """Test suite for ProviderFactory class."""

    def test_provider_factory_init(self):
        """Test ProviderFactory initialization."""
        config = {"primary_provider": "xai", "primary_model": "grok-4"}
        factory = ProviderFactory(config)
        assert factory is not None

    def test_provider_factory_stores_config(self):
        """Test that ProviderFactory stores configuration."""
        config = {"primary_provider": "xai", "primary_model": "grok-4"}
        factory = ProviderFactory(config)
        assert factory.config == config

    def test_provider_factory_has_providers_dict(self):
        """Test that ProviderFactory has providers dictionary."""
        config = {}
        factory = ProviderFactory(config)
        assert hasattr(factory, 'providers')
        assert isinstance(factory.providers, dict)

    def test_provider_factory_has_get_provider_method(self):
        """Test that ProviderFactory has get_provider method."""
        config = {}
        factory = ProviderFactory(config)
        assert hasattr(factory, 'get_provider')
        assert callable(factory.get_provider)

    def test_get_provider_accepts_no_args(self):
        """Test that get_provider can be called without arguments."""
        config = {"primary_provider": "xai"}
        factory = ProviderFactory(config)
        try:
            factory.get_provider()
        except TypeError as e:
            pytest.fail(f"get_provider should accept no args: {e}")

    def test_get_provider_accepts_metadata(self):
        """Test that get_provider accepts paper metadata."""
        config = {"primary_provider": "xai"}
        factory = ProviderFactory(config)
        metadata = {"complexity": 0.8, "length": 5000}
        try:
            factory.get_provider(paper_metadata=metadata)
        except TypeError as e:
            pytest.fail(f"get_provider should accept metadata: {e}")

    def test_provider_factory_with_empty_config(self):
        """Test ProviderFactory with empty configuration."""
        factory = ProviderFactory({})
        assert factory.config == {}

    def test_provider_factory_with_complex_config(self):
        """Test ProviderFactory with complex configuration."""
        config = {
            "primary_provider": "xai",
            "primary_model": "grok-4-fast-reasoning",
            "fallback_rules": [
                {"condition": "rate_limit", "provider": "together"}
            ],
            "providers": {
                "xai": {"base_url": "https://api.x.ai/v1"},
                "together": {"base_url": "https://api.together.xyz/v1"}
            }
        }
        factory = ProviderFactory(config)
        assert factory.config == config


class TestProviderFactoryIntegration:
    """Integration tests for provider factory scenarios."""

    def test_factory_with_xai_config(self):
        """Test factory initialization with xAI configuration."""
        config = {
            "primary_provider": "xai",
            "primary_model": "grok-4-fast-reasoning",
            "providers": {
                "xai": {
                    "base_url": "https://api.x.ai/v1",
                    "models": {
                        "grok-4-fast-reasoning": {
                            "cost_per_million_input": 0.20,
                            "cost_per_million_output": 0.50
                        }
                    }
                }
            }
        }
        factory = ProviderFactory(config)
        assert factory.config["primary_provider"] == "xai"

    def test_factory_with_multiple_providers(self):
        """Test factory with multiple provider configurations."""
        config = {
            "primary_provider": "xai",
            "providers": {
                "xai": {"base_url": "https://api.x.ai/v1"},
                "together": {"base_url": "https://api.together.xyz/v1"},
                "openai": {"base_url": "https://api.openai.com/v1"}
            }
        }
        factory = ProviderFactory(config)
        assert len(factory.config["providers"]) == 3

    def test_factory_with_fallback_rules(self):
        """Test factory with fallback rules configuration."""
        config = {
            "primary_provider": "xai",
            "fallback_rules": [
                {"condition": "rate_limit", "provider": "together"},
                {"condition": "error", "provider": "openai"}
            ]
        }
        factory = ProviderFactory(config)
        assert len(factory.config["fallback_rules"]) == 2

    def test_factory_get_provider_with_simple_metadata(self):
        """Test get_provider with simple paper metadata."""
        config = {"primary_provider": "xai"}
        factory = ProviderFactory(config)
        metadata = {"title": "Test paper", "complexity": 0.5}
        factory.get_provider(paper_metadata=metadata)

    def test_factory_get_provider_with_complex_metadata(self):
        """Test get_provider with complex paper metadata."""
        config = {"primary_provider": "xai"}
        factory = ProviderFactory(config)
        metadata = {
            "title": "Advanced Transformer Architecture",
            "complexity": 0.95,
            "length": 15000,
            "citations": 100,
            "has_math": True,
            "has_code": True
        }
        factory.get_provider(paper_metadata=metadata)

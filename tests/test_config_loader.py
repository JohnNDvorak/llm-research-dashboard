"""Tests for configuration loader."""

import pytest
from src.utils.config_loader import load_config, get_stages, get_llm_providers


def test_load_stages_config():
    """Test loading stages configuration."""
    config = load_config("stages")

    assert config is not None
    assert "stages" in config
    assert isinstance(config["stages"], list)


def test_stages_has_8_stages():
    """Test that stages config has exactly 8 stages."""
    stages = get_stages()

    assert len(stages) == 8, f"Expected 8 stages, got {len(stages)}"


def test_stages_have_required_fields():
    """Test that each stage has name, keywords, and description."""
    stages = get_stages()

    for stage in stages:
        assert "name" in stage, f"Stage missing 'name' field: {stage}"
        assert "keywords" in stage, f"Stage missing 'keywords' field: {stage}"
        assert "description" in stage, f"Stage missing 'description' field: {stage}"
        assert isinstance(stage["keywords"], list), "Keywords must be a list"
        assert len(stage["keywords"]) > 0, f"Stage {stage['name']} has no keywords"


def test_load_llm_config():
    """Test loading LLM provider configuration."""
    config = load_config("llm_config")

    assert config is not None
    assert "primary_provider" in config
    assert "primary_model" in config
    assert "providers" in config


def test_llm_config_has_providers():
    """Test that LLM config has xai and together providers."""
    providers = get_llm_providers()

    assert "xai" in providers, "Missing xAI provider"
    assert "together" in providers, "Missing Together AI provider"

    # Check xAI provider structure
    xai = providers["xai"]
    assert "base_url" in xai
    assert "models" in xai
    assert "grok-4-fast-reasoning" in xai["models"]


def test_llm_config_has_fallback_rules():
    """Test that LLM config has fallback rules."""
    config = load_config("llm_config")

    assert "fallback_rules" in config
    assert isinstance(config["fallback_rules"], list)
    assert len(config["fallback_rules"]) >= 2, "Should have at least 2 fallback rules"


def test_load_embedding_config():
    """Test loading embedding configuration."""
    config = load_config("embedding_config")

    assert config is not None
    assert "primary_provider" in config
    assert "primary_model" in config
    assert "providers" in config

    # Check OpenAI embedding config
    assert "openai" in config["providers"]
    assert "text-embedding-3-small" in config["providers"]["openai"]["models"]


def test_load_queries_config():
    """Test loading queries configuration."""
    config = load_config("queries")

    assert config is not None
    assert "arxiv" in config
    assert "twitter" in config
    assert "linkedin" in config

    # Check arXiv config
    assert "queries" in config["arxiv"]
    assert isinstance(config["arxiv"]["queries"], list)


def test_load_budget_modes():
    """Test loading budget modes configuration."""
    config = load_config("budget_modes")

    assert config is not None
    assert "modes" in config
    assert "cheap" in config["modes"]
    assert "balanced" in config["modes"]
    assert "quality" in config["modes"]


def test_load_missing_config():
    """Test error handling for missing config files."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config")

"""Configuration file loader for YAML configs."""

import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_name: Name of config file (e.g., "stages", "llm_config")

    Returns:
        Parsed config dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(f"config/{config_name}.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: {config_path.absolute()}"
        )

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Failed to parse YAML file {config_path}: {e}"
        ) from e


def get_stages() -> List[Dict[str, Any]]:
    """
    Helper function to load pipeline stages configuration.

    Returns:
        List of stage dictionaries with name, keywords, description

    Example:
        >>> stages = get_stages()
        >>> len(stages)
        8
        >>> stages[0]['name']
        'Architecture Design'
    """
    config = load_config("stages")
    return config.get("stages", [])


def get_llm_providers() -> Dict[str, Any]:
    """
    Helper function to load LLM provider configurations.

    Returns:
        Dictionary of provider configurations

    Example:
        >>> providers = get_llm_providers()
        >>> 'xai' in providers
        True
        >>> providers['xai']['base_url']
        'https://api.x.ai/v1'
    """
    config = load_config("llm_config")
    return config.get("providers", {})


def get_primary_llm_provider() -> tuple[str, str]:
    """
    Get the primary LLM provider and model.

    Returns:
        Tuple of (provider_name, model_name)

    Example:
        >>> provider, model = get_primary_llm_provider()
        >>> provider
        'xai'
        >>> model
        'grok-4-fast-reasoning'
    """
    config = load_config("llm_config")
    provider = config.get("primary_provider", "xai")
    model = config.get("primary_model", "grok-4-fast-reasoning")
    return provider, model


def get_embedding_config() -> Dict[str, Any]:
    """
    Helper function to load embedding configuration.

    Returns:
        Embedding configuration dictionary

    Example:
        >>> config = get_embedding_config()
        >>> config['primary_provider']
        'openai'
    """
    return load_config("embedding_config")


def get_queries_config() -> Dict[str, Any]:
    """
    Helper function to load search queries configuration.

    Returns:
        Queries configuration dictionary

    Example:
        >>> config = get_queries_config()
        >>> 'arxiv' in config
        True
    """
    return load_config("queries")


def get_budget_mode(mode: str = "balanced") -> Dict[str, Any]:
    """
    Get configuration for a specific budget mode.

    Args:
        mode: Budget mode name ('cheap', 'balanced', or 'quality')

    Returns:
        Budget mode configuration dictionary

    Raises:
        ValueError: If mode doesn't exist

    Example:
        >>> config = get_budget_mode("balanced")
        >>> config['llm_provider']
        'xai'
    """
    config = load_config("budget_modes")
    modes = config.get("modes", {})

    if mode not in modes:
        available = ", ".join(modes.keys())
        raise ValueError(
            f"Unknown budget mode: '{mode}'. "
            f"Available modes: {available}"
        )

    return modes[mode]

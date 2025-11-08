"""Configuration file loader for YAML configs."""

from typing import Dict, Any


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_name: Name of config file (e.g., "stages", "llm_config")

    Returns:
        Parsed config dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    # TODO: Implement YAML loading
    pass

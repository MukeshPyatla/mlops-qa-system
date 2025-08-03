"""
Configuration management utilities for the LLM-Powered Q&A System.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Global configuration cache
_config_cache: Dict[str, Any] = {}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if config_path in _config_cache:
        return _config_cache[config_path]
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    _config_cache[config_path] = config
    return config


def get_config(config_name: str) -> Dict[str, Any]:
    """
    Get configuration by name from the configs directory.
    
    Args:
        config_name: Name of the configuration file (without .yaml extension)
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = f"configs/{config_name}.yaml"
    return load_config(config_path)


def load_env_config() -> Dict[str, str]:
    """
    Load environment variables and return as a dictionary.
    
    Returns:
        Dictionary of environment variables
    """
    # Load .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
    
    # Get all environment variables
    env_vars = {}
    for key, value in os.environ.items():
        env_vars[key] = value
    
    return env_vars


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get an environment variable with optional default value.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate that a configuration contains all required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    Later configs override earlier ones.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged 
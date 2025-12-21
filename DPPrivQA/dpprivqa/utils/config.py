"""
Configuration management utilities.
"""

import yaml
import os
from typing import Dict, Any, Optional


_config: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    global _config
    
    # Try multiple paths
    paths_to_try = [
        config_path,
        os.path.join(os.getcwd(), config_path),
        os.path.join(os.path.dirname(__file__), "..", "..", config_path)
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                _config = yaml.safe_load(f)
            return _config
    
    # Return default config if file not found
    _config = {
        "models": {
            "local": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "remote_cot": "gpt-5",
            "remote_qa": "gpt-5"
        },
        "mechanisms": {
            "use_phrasedp_plus": False,
            "default_epsilon": 2.0,
            "epsilon_values": [1.0, 2.0, 3.0]
        },
        "database": {
            "path": "exp-results/results.db"
        },
        "logging": {
            "level": "INFO",
            "directory": "exp-results/logs"
        }
    }
    
    return _config


def get_config() -> Dict[str, Any]:
    """
    Get current configuration (loads if not already loaded).
    
    Returns:
        Configuration dictionary
    """
    global _config
    if _config is None:
        return load_config()
    return _config



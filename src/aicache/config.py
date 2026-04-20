import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "cache_dir": "~/.cache/aicache",
    "ttl": 0,  # 0 means no expiration
    "cache_size_limit": 0,  # 0 means no limit
    "generic_wrappers": {},  # Generic wrappers configuration
    # Semantic caching configuration
    "semantic_cache": {
        "enabled": True,
        "backend": "chromadb",  # or "faiss"
        "embedding_model": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.85,
        "embedding_dimension": 384,
        "persist_directory": "~/.cache/aicache/embeddings",
    },
    # Intelligent cache management
    "intelligent_management": {
        "enabled": True,
        "max_age_days": 30,
        "max_size_mb": 1000,
        "eviction_policy": "cost_aware_lfu",  # lru, lfu, cost_aware_lfu
        "auto_prune": True,
        "prune_interval_hours": 24,
    },
    # Analytics and monitoring
    "analytics": {
        "enabled": False,  # Changed to False for open source release
        "metrics_retention_days": 90,
        "dashboard_port": 8080,
        "export_interval_hours": 24,
        "performance_tracking": False,  # Changed to False for open source release
    },
    # Team collaboration
    "team": {
        "enabled": False,
        "team_id": None,
        "user_id": None,
        "team_secret": None,
        "share_level_default": "team",  # private, team, public
        "auto_share": False,
        "encryption_enabled": True,
    },
    # Streaming support
    "streaming": {
        "enabled": True,
        "websocket_port": 8765,
        "max_stream_size_mb": 100,
        "chunk_size": 1024,
        "delta_caching": True,
        "real_time_notifications": True,
    },
    # Provider-specific settings
    "providers": {
        "openai": {
            "prompt_caching": True,
            "cost_per_1k_input": 0.0015,
            "cost_per_1k_output": 0.002,
        },
        "anthropic": {
            "prompt_caching": True,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
        },
        "google": {
            "prompt_caching": True,
            "cost_per_1k_input": 0.001,
            "cost_per_1k_output": 0.002,
        },
    },
    # Security and privacy
    "security": {
        "encrypt_sensitive": True,
        "data_retention_days": 365,
        "anonymize_logs": True,
        "audit_logging": True,
        "pii_detection": True,
        "input_sanitization": True,
        "safe_mode": True,
    },
    # Performance tuning
    "performance": {
        "async_operations": True,
        "compression_enabled": True,
        "compression_level": 6,  # 1-9
        "parallel_processing": True,
        "max_workers": 4,
    },
    # Advanced features
    "advanced": {
        "knowledge_graph": True,
        "predictive_caching": False,
        "federated_learning": False,
        "quantum_safe_encryption": False,
    },
}


class ConfigManager:
    """Advanced configuration manager with validation and hot reloading."""

    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or (Path.home() / ".config" / "aicache" / "config.yaml")
        self.config_dir = self.config_path.parent
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._config = DEFAULT_CONFIG.copy()
        self._load_config()

        # Create example config if it doesn't exist
        if not self.config_path.exists():
            self._create_example_config()

    def _load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    user_config = yaml.safe_load(f) or {}

                # Deep merge with defaults
                self._config = self._deep_merge(DEFAULT_CONFIG, user_config)
                logger.info(f"Loaded configuration from {self.config_path}")

            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Using default configuration")

    def _deep_merge(self, default: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
        """Deep merge user config with defaults."""
        result = default.copy()

        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _create_example_config(self):
        """Create an example configuration file."""
        try:
            with open(self.config_path, "w") as f:
                f.write("""# aicache Configuration File
# This file contains all configuration options for aicache.
# Uncomment and modify values as needed.

# Basic cache settings
cache_dir: "~/.cache/aicache"
# ttl: 86400  # Cache TTL in seconds (0 = no expiration)
# cache_size_limit: 1000  # Cache size limit in MB (0 = no limit)

# Semantic caching (AI-powered similarity matching)
semantic_cache:
  enabled: true
  # backend: "chromadb"  # or "faiss" for better performance
  # embedding_model: "all-MiniLM-L6-v2"
  # similarity_threshold: 0.85  # 0.0-1.0, higher = more strict

# Intelligent cache management
intelligent_management:
  enabled: true
  # max_age_days: 30
  # max_size_mb: 1000
  # auto_prune: true

# Team collaboration (requires setup)
# team:
#   enabled: false
#   team_id: "your-team-id"
#   user_id: "your-user-id"

# Analytics and performance tracking
# By default analytics are disabled for privacy.
# Enable only if you consent to data collection.
analytics:
  # enabled: true
  # dashboard_port: 8080
  # performance_tracking: true

# Streaming support
streaming:
  enabled: true
  # websocket_port: 8765
  # delta_caching: true

# Provider-specific optimizations
providers:
  openai:
    prompt_caching: true
  anthropic:
    prompt_caching: true

# Security settings
security:
  encrypt_sensitive: true
  # anonymize_logs: true
""")
            logger.info(f"Created example config at {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to create example config: {e}")

    def get(self, key: str | None = None, default: Any = None) -> Any:
        """Get configuration value(s)."""
        if key is None:
            return self._config

        # Support dot notation for nested keys
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any, persist: bool = True) -> bool:
        """Set configuration value."""
        try:
            keys = key.split(".")
            config = self._config

            # Navigate to the correct nested dict
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            # Set the value
            config[keys[-1]] = value

            if persist:
                self.save_config()

            return True
        except Exception as e:
            logger.error(f"Failed to set config {key}={value}: {e}")
            return False

    def save_config(self):
        """Save current configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def validate_config(self) -> dict[str, list[str]]:
        """Validate configuration and return any errors/warnings."""
        errors = []
        warnings = []

        # Validate cache directory
        cache_dir = self.get("cache_dir")
        if cache_dir:
            expanded_dir = Path(os.path.expanduser(cache_dir))
            if not expanded_dir.parent.exists():
                errors.append(f"Cache directory parent does not exist: {expanded_dir.parent}")

        # Validate semantic cache settings
        if self.get("semantic_cache.enabled"):
            backend = self.get("semantic_cache.backend")
            if backend not in ["chromadb", "faiss"]:
                errors.append(f"Invalid semantic cache backend: {backend}")

            threshold = self.get("semantic_cache.similarity_threshold")
            if not (0.0 <= threshold <= 1.0):
                errors.append(f"Similarity threshold must be between 0.0 and 1.0, got: {threshold}")

        # Validate team settings
        if self.get("team.enabled"):
            if not self.get("team.team_id"):
                errors.append("Team ID is required when team collaboration is enabled")
            if not self.get("team.user_id"):
                errors.append("User ID is required when team collaboration is enabled")

        # Validate ports
        dashboard_port = self.get("analytics.dashboard_port")
        websocket_port = self.get("streaming.websocket_port")

        if dashboard_port == websocket_port:
            errors.append("Dashboard and WebSocket ports cannot be the same")

        for port in [dashboard_port, websocket_port]:
            if not (1024 <= port <= 65535):
                warnings.append(f"Port {port} is outside recommended range 1024-65535")

        return {"errors": errors, "warnings": warnings, "valid": len(errors) == 0}

    def get_feature_flags(self) -> dict[str, bool]:
        """Get all feature flags as a flat dictionary."""
        return {
            "semantic_cache": self.get("semantic_cache.enabled", True),
            "intelligent_management": self.get("intelligent_management.enabled", True),
            "analytics": self.get("analytics.enabled", False),  # Changed default to False
            "team_collaboration": self.get("team.enabled", False),
            "streaming": self.get("streaming.enabled", True),
            "knowledge_graph": self.get("advanced.knowledge_graph", True),
            "predictive_caching": self.get("advanced.predictive_caching", False),
            "encryption": self.get("security.encrypt_sensitive", True),
            "performance_tracking": self.get(
                "analytics.performance_tracking", False
            ),  # Changed default to False
            "auto_prune": self.get("intelligent_management.auto_prune", True),
        }

    def export_config(self, filepath: Path, include_defaults: bool = False):
        """Export configuration to file."""
        config_to_export = self._config if include_defaults else self._get_non_default_config()

        try:
            with open(filepath, "w") as f:
                yaml.dump(config_to_export, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to export config to {filepath}: {e}")

    def _get_non_default_config(self) -> dict[str, Any]:
        """Get only non-default configuration values."""

        def compare_dicts(default: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
            result = {}
            for key, value in current.items():
                if key not in default:
                    result[key] = value
                elif isinstance(value, dict) and isinstance(default[key], dict):
                    nested_diff = compare_dicts(default[key], value)
                    if nested_diff:
                        result[key] = nested_diff
                elif value != default[key]:
                    result[key] = value
            return result

        return compare_dicts(DEFAULT_CONFIG, self._config)


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key: str | None = None, default: Any = None) -> Any:
    """Get configuration value (backward compatibility)."""
    return get_config_manager().get(key, default)


def set_config(key: str, value: Any, persist: bool = True) -> bool:
    """Set configuration value."""
    return get_config_manager().set(key, value, persist)


def validate_config() -> dict[str, Any]:
    """Validate current configuration."""
    return get_config_manager().validate_config()


def reload_config():
    """Reload configuration from file."""
    global _config_manager
    _config_manager = ConfigManager()


# Configuration schema for validation
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "cache_dir": {"type": "string"},
        "ttl": {"type": "integer", "minimum": 0},
        "cache_size_limit": {"type": "integer", "minimum": 0},
        "semantic_cache": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "backend": {"type": "string", "enum": ["chromadb", "faiss"]},
                "similarity_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
        },
    },
}

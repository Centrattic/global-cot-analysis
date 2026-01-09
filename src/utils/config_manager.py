import os
import yaml
from pathlib import Path
from typing import Dict, Any, List
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._hydra_initialized = False

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration by name."""
        if not self._hydra_initialized:
            self._initialize_hydra()

        cfg = compose(config_name=config_name)
        return self._hydra_to_dict(cfg)

    def load_response_config(self, config_name: str) -> Dict[str, Any]:
        """Load response generation configuration."""
        config = self.load_config(config_name)
        return config.get("p", {})

    def load_flowchart_config(self, config_name: str) -> Dict[str, Any]:
        """Load flowchart generation configuration."""
        config = self.load_config(config_name)
        return config.get("f", {})

    def load_prediction_config(self, config_name: str) -> Dict[str, Any]:
        """Load prediction configuration."""
        config = self.load_config(config_name)
        return config.get("p", {})

    def _initialize_hydra(self):
        """Initialize Hydra with config directory."""
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        initialize(config_path=str(self.config_dir), version_base=None)
        self._hydra_initialized = True

    def _hydra_to_dict(self, cfg) -> Dict[str, Any]:
        """Convert Hydra config to dictionary."""
        return yaml.safe_load(cfg.pretty())

    def get_models(self, config: Dict[str, Any]) -> List[str]:
        """Get list of models from config."""
        return config.get("models")

    def get_prompt(self, config: Dict[str, Any]) -> str:
        """Get prompt index from config."""
        return config.get("prompt")

    def get_prefixes(self, config: Dict[str, Any]) -> List[str]:
        """Get list of prefixes from config."""
        return config.get("prefixes", [])

    def get_property_checkers(self, config: Dict[str, Any]) -> List[str]:
        """Get list of property checkers from config."""
        return config.get("property_checkers", [])

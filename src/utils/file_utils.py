import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileUtils:
    """Utility class for file operations."""

    @staticmethod
    def ensure_dir(path: str):
        """Ensure directory exists."""
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def file_exists(path: str) -> bool:
        """Check if file exists."""
        return Path(path).exists()

    @staticmethod
    def get_response_file_path(prompt_index: str, config_name: str) -> str:
        """Get path for consolidated response file (multi-model)."""
        return f"prompts/{prompt_index}/config-{config_name}_responses.json"

    @staticmethod
    def get_responses_file_path(prompt_index: str, config_name: str) -> str:
        """Get path for consolidated responses file (alias for get_response_file_path)."""
        return FileUtils.get_response_file_path(prompt_index, config_name)

    @staticmethod
    def get_rollout_file_path(prompt_index: str, model: str, seed: int) -> str:
        """Get path for individual rollout file."""
        return f"prompts/{prompt_index}/{model}/rollouts/{seed}.json"

    @staticmethod
    def get_resample_file_path(prompt_index: str, model: str,
                               prefix_index: str, seed: int) -> str:
        """Get path for individual resample file."""
        return f"prompts/{prompt_index}/{model}/resamples/{prefix_index}/{seed}.json"

    @staticmethod
    def get_flowchart_file_path(prompt_index: str, config_name: str,
                                f_config_name: str) -> str:
        """Get path for flowchart file (multi-model)."""
        return f"flowcharts/{prompt_index}/config-{config_name}-{f_config_name}_flowchart.json"

    @staticmethod
    def get_activation_file_path(prompt_index: str, model: str, seed: int,
                                 layer: int) -> str:
        """Get path for activation file."""
        return f"embed_cache/{prompt_index}/{model}/rollouts/{seed}_layer_{layer}.npy"

    @staticmethod
    def get_resample_activation_file_path(prompt_index: str, model: str,
                                          prefix_index: str, seed: int,
                                          layer: int) -> str:
        """Get path for resample activation file."""
        return f"embed_cache/{prompt_index}/{model}/{prefix_index}/{seed}_layer_{layer}.npy"

    @staticmethod
    def get_graph_cache_file_path(flowchart_path: str) -> str:
        """Get path for graph layout cache file."""
        flowchart_name = Path(flowchart_path).stem
        return f"graph_layout_service/cache/{flowchart_name}_sfdp.json"

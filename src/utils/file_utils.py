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
    def get_rollout_file_path(prompt_index: str, model: str, seed: int) -> str:
        """Get path for individual rollout file."""
        return f"prompts/{prompt_index}/{model}/rollouts/{seed}.json"

    @staticmethod
    def get_resample_file_path(prompt_index: str, model: str, prefix_index: str, seed: int) -> str:
        """Get path for individual resample file."""
        return f"prompts/{prompt_index}/{model}/resamples/{prefix_index}/{seed}.json"

    @staticmethod
    def get_flowchart_file_path(
        prompt_index: str, config_name: str, f_config_name: str, models: List[str] = None
    ) -> str:
        """Get path for flowchart file.

        If models is provided and has exactly one model, includes model name in filename.
        Model name is converted: "gpt-oss-20b" -> "gpt_oss_20b"
        """
        if models and len(models) == 1:
            # Convert model name: replace "/" and "-" with "_"
            model_safe = models[0].replace("/", "_").replace("-", "_")
            return f"flowcharts/{prompt_index}/config-{config_name}-{f_config_name}_{model_safe}_flowchart.json"
        else:
            # No model or multiple models: use original format
            return f"flowcharts/{prompt_index}/config-{config_name}-{f_config_name}_flowchart.json"

    @staticmethod
    def get_graph_cache_file_path(flowchart_path: str) -> str:
        """Get path for graph layout cache file."""
        flowchart_name = Path(flowchart_path).stem
        return f"graph_layout_service/cache/{flowchart_name}_sfdp.json"

    @staticmethod
    def get_rollout_dir(prompt_index: str, model: str) -> str:
        """Get directory path for rollouts."""
        return f"prompts/{prompt_index}/{model}/rollouts"

    @staticmethod
    def get_resample_dir(prompt_index: str, model: str) -> str:
        """Get directory path for resamples."""
        return f"prompts/{prompt_index}/{model}/resamples"

    @staticmethod
    def get_resample_prefix_dir(prompt_index: str, model: str, prefix_index: str) -> str:
        """Get directory path for a specific resample prefix."""
        return f"prompts/{prompt_index}/{model}/resamples/{prefix_index}"

    @staticmethod
    def get_predictions_dir(prompt_index: str, model: str) -> str:
        """Get directory path for predictions."""
        return f"prompts/{prompt_index}/{model}/predictions"

    @staticmethod
    def get_flowchart_dir(prompt_index: str) -> str:
        """Get directory path for flowcharts."""
        return f"flowcharts/{prompt_index}"

    @staticmethod
    def get_prompts_file_path() -> str:
        """Get path for prompts.json."""
        return "prompts/prompts.json"

    @staticmethod
    def get_prefixes_file_path(prompt_index: str) -> str:
        """Get path for prefixes.json."""
        return f"prompts/{prompt_index}/prefixes.json"

    @staticmethod
    def get_algorithms_file_path() -> str:
        """Get path for algorithms.json."""
        return "prompts/algorithms.json"

    @staticmethod
    def get_config_file_path(config_name: str) -> str:
        """Get path for a config file."""
        return f"configs/{config_name}.yaml"

    @staticmethod
    def get_p_config_file_path(p_config_name: str) -> str:
        """Get path for a p-config file."""
        return f"configs/p/{p_config_name}.yaml"

    @staticmethod
    def get_r_config_file_path(r_config_name: str) -> str:
        """Get path for an r-config file."""
        return f"configs/r/{r_config_name}.yaml"

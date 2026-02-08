"""
Prediction runner for analyzing rollouts and resamples.

This module handles running prefix correctness analysis using the "current" method.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig

from .utils_predictions import (
    find_flowchart_path,
    check_resamples_exist,
    run_prefix_prediction_comparison,
    save_comparison_csv,
    get_config_value,
)


class PredictionRunner:
    """Handles running prediction analyses using the 'current' method."""

    def __init__(
        self, config: DictConfig, use_condensed: bool = False, use_fully_condensed: bool = False, recompute: bool = False
    ):
        """Initialize prediction runner with configuration."""
        self.config = config
        self.prompt = config.prompt
        self.models = config.models
        self.use_condensed = bool(use_condensed)
        self.use_fully_condensed = bool(use_fully_condensed)
        self.recompute = recompute

        # Get prediction config (guaranteed to exist by main.py)
        self.p_config = config.p
        self.f_config = config.f
        self.top_rollouts = getattr(self.p_config, "top_rollouts", 50)
        self.size_filter = getattr(self.p_config, "size_filter", None)

    def run_predictions_from_config(self, config_name: str) -> None:
        """Run predictions from configuration."""
        print(f"Running predictions for config: {config_name}")
        print(f"Prompt: {self.prompt}")
        print(f"Models: {self.models}")
        print(f"Method: current")
        print(f"Top rollouts: {self.top_rollouts}")

        for model in self.models:
            self.run_predictions_for_model(model, config_name)

    def run_predictions_for_model(self, model: str, config_name: str) -> None:
        """Run predictions for a specific model."""
        print(f"\nRunning predictions for model: {model}")

        # Create base predictions directory
        base_predictions_dir = Path(f"prompts/{self.prompt}/{model}/predictions")
        base_predictions_dir.mkdir(parents=True, exist_ok=True)

        # Create method-specific predictions directory with p-config subfolder
        p_config_name = getattr(self.p_config, "_name_", "default")
        predictions_dir = base_predictions_dir / "current" / p_config_name
        if self.use_fully_condensed:
            predictions_dir = predictions_dir / "fully_condensed"
        elif self.use_condensed:
            predictions_dir = predictions_dir / "condensed"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Check if flowchart exists
        f_config_name = getattr(self.f_config, "_name_", None)
        flowchart_path = find_flowchart_path(self.prompt, config_name, f_config_name, self.models)
        if not flowchart_path:
            print(f"No flowchart found for {model}, skipping predictions")
            return

        if self.use_fully_condensed:
            fc_path = flowchart_path.with_name(flowchart_path.stem + "_fully_condensed.json")
            if fc_path.exists():
                flowchart_path = fc_path
            else:
                print(f"Fully condensed flowchart not found at {fc_path}, skipping predictions")
                return
        elif self.use_condensed:
            condensed_path = flowchart_path.with_name(flowchart_path.stem + "_condensed.json")
            if condensed_path.exists():
                flowchart_path = condensed_path
            else:
                print(f"Condensed flowchart not found at {condensed_path}, skipping predictions")
                return

        print(f"Found flowchart: {flowchart_path}")
        flowchart_name = flowchart_path.stem

        # Check for resamples and run prefix correctness analysis if they exist
        self._run_prefix_correctness_analysis_if_needed(
            model, predictions_dir, config_name, flowchart_name
        )

    def _run_prefix_correctness_analysis_if_needed(
        self, model: str, predictions_dir: Path, config_name: str, flowchart_name: str
    ) -> None:
        """Run prefix correctness analysis if resamples exist."""
        # Check if prefixes are specified in config
        prefixes = getattr(self.config, "prefixes", None)
        if not prefixes:
            if hasattr(self.config, "prefixes") and prefixes == []:
                print("No prefixes specified in config, skipping prefix correctness analysis")
            return

        # Check for resamples
        resamples_dir = check_resamples_exist(self.prompt, model)
        if not resamples_dir:
            print(f"No resamples directory found for {model}")
            return

        print(f"Found resamples directory: {resamples_dir}")

        # Run prefix prediction comparison directly (no JSON file needed)
        self._run_prefix_prediction_comparison(
            predictions_dir, flowchart_name, resamples_dir, model
        )

    def _run_prefix_prediction_comparison(
        self, predictions_dir: Path, flowchart_name: str, resamples_dir: Path, model: str
    ) -> None:
        """Run comparison between predicted and actual prefix response distributions."""
        # Find the flowchart file
        flowchart_path = None
        flowchart_dir = Path(f"flowcharts/{self.prompt}")
        for flowchart_file in flowchart_dir.glob("*.json"):
            if flowchart_name in flowchart_file.name:
                flowchart_path = flowchart_file
                break

        if not flowchart_path:
            print(f"Could not find flowchart for comparison: {flowchart_name}")
            return

        # Output base name
        output_base_name = (
            f"correctness_analysis_{flowchart_name}_method_current_top_rollouts_{self.top_rollouts}"
        )
        if self.size_filter is not None:
            output_base_name += f"_size_filter_{self.size_filter}"

        csv_path = predictions_dir / f"{output_base_name}.csv"

        # Check if results already exist
        if csv_path.exists() and not self.recompute:
            print(f"Predictions already exist at {csv_path}, skipping (use --recompute to force)")
            return

        if csv_path.exists():
            print(f"Overwriting existing prefix prediction comparison: {csv_path}")

        print(f"Running prefix prediction comparison...")

        # Run the comparison using the 'current' method
        comparison_results = run_prefix_prediction_comparison(
            str(flowchart_path),
            str(resamples_dir),
            self.config._name_,
            self.top_rollouts,
            self.size_filter,
            self.prompt,
            model,
        )

        # Save results
        save_comparison_csv(comparison_results, str(csv_path))
        print(f"Prefix prediction comparison completed: {csv_path}")

    def _get_total_rollouts_from_config(self) -> int:
        """Get the total number of rollouts from the config file."""
        config_path = f"configs/p/{self.config._name_}.yaml"
        r_config = get_config_value(config_path, "r", {})

        if isinstance(r_config, str):
            # Reference to another config file
            rollout_config_path = f"configs/r/{r_config}.yaml"
            return get_config_value(rollout_config_path, "num_seeds_rollouts", 100)
        else:
            return r_config.get("num_seeds_rollouts", 100) if r_config else 100

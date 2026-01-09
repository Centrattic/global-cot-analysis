"""
Prediction runner for analyzing rollouts and resamples.

This module handles running prefix correctness analysis using the "current" method.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from omegaconf import DictConfig
from datetime import datetime

from .utils_predictions import (
    find_flowchart_path, check_resamples_exist,
    load_flowchart_data, get_clusters_from_rollout,
    load_prefix_data_from_resamples,
    calculate_correctness_stats, run_prefix_prediction_comparison,
    save_comparison_csv, create_comparison_plot,
    get_config_value, load_prefix_correctness_data)


class PredictionRunner:
    """Handles running prediction analyses using the 'current' method."""

    def __init__(self,
                 config: DictConfig,
                 use_condensed: bool = False,
                 use_fully_condensed: bool = False):
        """Initialize prediction runner with configuration."""
        self.config = config
        self.prompt = config.prompt
        self.models = config.models
        self.use_condensed = bool(use_condensed)
        self.use_fully_condensed = bool(use_fully_condensed)

        # Get prediction config (guaranteed to exist by main.py)
        self.p_config = config.p
        self.f_config = config.f
        self.top_rollouts = getattr(self.p_config, 'top_rollouts', 50)
        self.size_filter = getattr(self.p_config, 'size_filter', None)

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
        base_predictions_dir = Path(
            f"prompts/{self.prompt}/{model}/predictions")
        base_predictions_dir.mkdir(parents=True, exist_ok=True)

        # Create method-specific predictions directory
        predictions_dir = base_predictions_dir / 'current'
        if self.use_fully_condensed:
            predictions_dir = predictions_dir / 'fully_condensed'
        elif self.use_condensed:
            predictions_dir = predictions_dir / 'condensed'
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Check if flowchart exists
        f_config_name = getattr(self.f_config, '_name_', None)
        flowchart_path = find_flowchart_path(self.prompt, config_name, f_config_name)
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
            model, predictions_dir, config_name, flowchart_name)

    def _run_prefix_correctness_analysis_if_needed(
            self, model: str, predictions_dir: Path, config_name: str,
            flowchart_name: str) -> None:
        """Run prefix correctness analysis if resamples exist."""
        # Check if prefixes are specified in config
        prefixes = getattr(self.config, 'prefixes', None)
        if not prefixes:
            if hasattr(self.config, 'prefixes') and prefixes == []:
                print("No prefixes specified in config, skipping prefix correctness analysis")
            return

        # Check for resamples
        resamples_dir = check_resamples_exist(self.prompt, model)
        if not resamples_dir:
            print(f"No resamples directory found for {model}")
            return

        print(f"Found resamples directory: {resamples_dir}")

        # Run prefix correctness analysis
        self._run_prefix_correctness_analysis(resamples_dir, predictions_dir,
                                              model, config_name,
                                              flowchart_name)

    def _run_prefix_correctness_analysis(self, resamples_dir: Path,
                                         predictions_dir: Path, model: str,
                                         config_name: str,
                                         flowchart_name: str) -> None:
        """Run prefix correctness analysis for all prefixes."""
        output_filename = f"prefix_correctness_analysis_{flowchart_name}.json"
        output_path = predictions_dir / output_filename

        # Count actual prefix directories
        prefix_dirs = [
            d for d in resamples_dir.iterdir()
            if d.is_dir() and d.name.startswith('prefix-')
        ]
        actual_prefix_count = len(prefix_dirs)

        # Always run analysis (overwrite existing files)
        if output_path.exists():
            print(f"Overwriting existing prefix correctness analysis: {output_path}")

        print(f"Running prefix correctness analysis...")

        # Load prefix data
        prefix_data = load_prefix_data_from_resamples(resamples_dir)

        # Analyze all prefixes
        results = {}
        for prefix_name, rollouts in prefix_data.items():
            print(f"Processing {prefix_name} with {len(rollouts)} rollouts...")
            stats = calculate_correctness_stats(rollouts)
            results[prefix_name] = stats

        # Save results
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'resamples_directory': str(resamples_dir),
            'prefix_count': len(results),
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Prefix correctness analysis completed: {output_path}")
        # Run comparison after successful analysis
        self._run_prefix_prediction_comparison(predictions_dir, flowchart_name,
                                               output_path, model)

    def _run_prefix_prediction_comparison(
            self, predictions_dir: Path, flowchart_name: str,
            prefix_correctness_path: Path, model: str) -> None:
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
        output_base_name = f"correctness_analysis_{flowchart_name}_method_current_top_rollouts_{self.top_rollouts}"
        if self.size_filter is not None:
            output_base_name += f"_size_filter_{self.size_filter}"
        
        csv_path = predictions_dir / f"{output_base_name}.csv"
        png_path = predictions_dir / f"{output_base_name}.png"

        # Always run comparison (overwrite existing files)
        if csv_path.exists() and png_path.exists():
            print(f"Overwriting existing prefix prediction comparison: {csv_path}, {png_path}")

        print(f"Running prefix prediction comparison...")

        # Run the comparison using the 'current' method
        comparison_results = run_prefix_prediction_comparison(
            str(flowchart_path), str(prefix_correctness_path),
            self.config._name_, self.top_rollouts, self.size_filter,
            self.prompt, model)

        # Save results
        save_comparison_csv(comparison_results, str(csv_path))
        print(f"Comparison CSV saved: {csv_path}")

        total_rollouts = self._get_total_rollouts_from_config()
        create_comparison_plot(comparison_results, str(png_path), total_rollouts)

        print(f"Prefix prediction comparison completed: {csv_path}, {png_path}")

    def _get_total_rollouts_from_config(self) -> int:
        """Get the total number of rollouts from the config file."""
        config_path = f"configs/p/{self.config._name_}.yaml"
        r_config = get_config_value(config_path, 'r', {})

        if isinstance(r_config, str):
            # Reference to another config file
            rollout_config_path = f"configs/r/{r_config}.yaml"
            return get_config_value(rollout_config_path, 'num_seeds_rollouts', 100)
        else:
            return r_config.get('num_seeds_rollouts', 100) if r_config else 100

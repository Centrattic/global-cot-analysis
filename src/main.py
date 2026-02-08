#!/usr/bin/env python3
"""
Main orchestration script for the clustering-cot system using Hydra.
"""

import os
import sys
import argparse
import hydra
import sys
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main_with_config(cfg: DictConfig, command: str, recompute: bool = False) -> None:
    """Main function using Hydra configuration."""

    print(f"Running command: {command}")
    print(f"Configuration:")
    print(OmegaConf.to_yaml(cfg))

    if command == "rollouts":
        from src.generation import APIResponseGenerator

        generator = APIResponseGenerator()
        generator.generate_rollouts_from_config(
            cfg,
            recompute=recompute,
        )

    elif command == "prefixes":
        from src.generation import APIResponseGenerator

        generator = APIResponseGenerator()

        # Never recompute prefixes, we should only delete these manually
        generator.generate_prefixes_from_config(cfg)

    elif command == "resamples":
        from src.generation import APIResponseGenerator

        generator = APIResponseGenerator()
        generator.generate_resamples_from_config(
            cfg,
            recompute=recompute,
        )

    elif command == "flowcharts":
        from src.flowchart.flowchart_generator import FlowchartGenerator

        generator = FlowchartGenerator()
        generator.generate_flowchart_from_config(
            cfg,
            recompute=recompute,
        )

    elif command == "labels":
        from src.flowchart.flowchart_generator import LabelGenerator

        generator = LabelGenerator()
        generator.generate_labels_from_config(cfg, recompute=recompute)

    elif command == "graphviz":
        from src.flowchart.graphviz_generator import GraphvizGenerator

        generator = GraphvizGenerator()
        generator.generate_graphviz_from_config(
            cfg,
            recompute=recompute,
        )

    elif command in ("predictions", "prediction"):
        from src.predictions.prediction_runner import PredictionRunner

        runner = PredictionRunner(cfg, use_condensed=False, use_fully_condensed=False, recompute=recompute)
        runner.run_predictions_from_config(cfg._name_)

    elif command == "properties":
        from src.property_checkers.property_runner import PropertyRunner

        runner = PropertyRunner(cfg, recompute=recompute,)
        runner.run_properties_from_config(cfg._name_)

    elif command == "cues":
        from src.labeling.generate_algorithms import generate_algorithms_from_config

        generate_algorithms_from_config(cfg)

    else:
        print(f"Unknown command: {command}")


def main():
    """Main entry point that handles argparse before Hydra."""

    # Parse command line arguments for flags like --recompute
    parser = argparse.ArgumentParser(description="Clustering CoT System")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of properties even if they already exist",
    )

    # Parse known args to extract our custom flags, pass the rest to Hydra
    args, unknown = parser.parse_known_args()

    # Add the unknown args back to sys.argv for Hydra
    sys.argv = [sys.argv[0]] + unknown

    @hydra.main(version_base=None, config_path="../configs", config_name="default")
    def hydra_main(cfg: DictConfig) -> None:
        main_with_config(cfg, cfg.command, recompute=args.recompute)

    hydra_main()


if __name__ == "__main__":
    main()

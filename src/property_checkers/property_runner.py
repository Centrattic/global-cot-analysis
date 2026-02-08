#!/usr/bin/env python3
"""
Property runner that applies all property checkers to rollout JSON files.
"""

from pathlib import Path
from typing import Dict, Any, List
import os
from collections import defaultdict

from src.utils.json_utils import load_json, write_json
from src.utils.file_utils import FileUtils
from src.property_checkers import PROPERTY_CHECKER_REGISTRY, PropertyCheckerMultiAlgorithm


class PropertyRunner:
    """Runs property checkers on rollout JSON files."""

    def __init__(self, config: Dict[str, Any], recompute: bool = False):
        self.config = config
        self.recompute = recompute
        self.property_checkers = {
            name: cls() for name, cls in PROPERTY_CHECKER_REGISTRY.items()
        }

    def run_properties_from_config(self, config_name: str):
        """Run property checkers on all rollout and resample files for the given config."""
        prompt_index = self.config["prompt"]
        models = list(self.config["models"])
        property_checker_names = list(self.config.get("property_checkers", []))
        node_property_checker_names = list(self.config.get("node_property_checkers", []))

        print(f"Running property checkers for prompt: {prompt_index}")
        print(f"Models: {models}")
        print(f"Property checkers: {property_checker_names}")
        if node_property_checker_names:
            print(f"Node property checkers: {node_property_checker_names}")
        if self.recompute:
            print("Recompute mode: Will force recomputation of all properties")

        for model in models:
            self._process_model_rollouts(prompt_index, model, property_checker_names)
            self._process_model_resamples(prompt_index, model, property_checker_names)

        # Process flowcharts
        self._process_flowcharts(prompt_index, models, property_checker_names)

        # Process node property checkers in flowcharts if specified
        if node_property_checker_names and self.recompute:
            self._process_flowchart_node_properties(prompt_index, node_property_checker_names)

    def _process_model_rollouts(
        self, prompt_index: str, model: str, property_checker_names: List[str]
    ):
        """Process all rollout files for a specific model."""
        print(f"\nProcessing model: {model}")

        # Construct the rollout directory path
        rollout_dir = FileUtils.get_rollout_dir(prompt_index, model)

        if not os.path.exists(rollout_dir):
            print(f"Rollout directory not found: {rollout_dir}")
            return

        # Process each JSON file in the directory
        json_files = [f for f in os.listdir(rollout_dir) if f.endswith(".json")]
        print(f"Found {len(json_files)} JSON files to process")

        # Collect all file paths
        file_paths = [os.path.join(rollout_dir, json_file) for json_file in json_files]

        # Process files in parallel batches for checkers that support it
        self._process_files_batch_parallel(file_paths, prompt_index, property_checker_names)

    def _process_model_resamples(
        self, prompt_index: str, model: str, property_checker_names: List[str]
    ):
        """Process all resample files for a specific model."""
        print(f"\nProcessing resamples for model: {model}")

        # Construct the resamples directory path
        resamples_dir = FileUtils.get_resample_dir(prompt_index, model)

        if not os.path.exists(resamples_dir):
            print(f"Resamples directory not found: {resamples_dir}")
            return

        # Get all subdirectories (prefix directories)
        prefix_dirs = [
            d for d in os.listdir(resamples_dir) if os.path.isdir(os.path.join(resamples_dir, d))
        ]

        if not prefix_dirs:
            print(f"No prefix directories found in: {resamples_dir}")
            return

        print(f"Found {len(prefix_dirs)} prefix directories: {prefix_dirs}")

        for prefix_dir in prefix_dirs:
            prefix_path = os.path.join(resamples_dir, prefix_dir)
            self._process_prefix_resamples(
                prefix_path, prompt_index, property_checker_names, prefix_dir
            )

    def _process_prefix_resamples(
        self,
        prefix_path: str,
        prompt_index: str,
        property_checker_names: List[str],
        prefix_name: str,
    ):
        """Process all JSON files in a specific prefix directory."""
        print(f"  Processing prefix: {prefix_name}")

        # Get all JSON files in this prefix directory
        json_files = [f for f in os.listdir(prefix_path) if f.endswith(".json")]

        if not json_files:
            print(f"    No JSON files found in {prefix_name}")
            return

        print(f"    Found {len(json_files)} JSON files")

        # Collect all file paths
        file_paths = [os.path.join(prefix_path, json_file) for json_file in json_files]

        # Process files in parallel batches for checkers that support it
        self._process_files_batch_parallel(file_paths, prompt_index, property_checker_names)

    def _process_single_file(
        self, file_path: str, prompt_index: str, property_checker_names: List[str]
    ):
        """Process a single JSON file and apply property checkers."""
        try:
            # Load the response data
            response_data = load_json(file_path)

            if not response_data:
                print(f"Skipping empty file: {file_path}")
                return

            # Check if file already has all required properties with valid values
            # Skip this check if recompute flag is set
            if not self.recompute:
                has_all_properties = all(
                    prop in response_data
                    and response_data[prop] not in ["unknown", "None", None, ""]
                    for prop in property_checker_names
                )

                if has_all_properties:
                    print(f"File already has all properties: {file_path}")
                    return

            print(f"Processing: {file_path}")

            # Apply property checkers
            for checker_name in property_checker_names:
                if checker_name in self.property_checkers:
                    checker = self.property_checkers[checker_name]
                    value = checker.get_value(response_data, prompt_index, file_path)
                    response_data[checker_name] = value
                    print(f"  {checker_name}: {value}")
                else:
                    print(f"  Warning: Property checker {checker_name} not found")

            # Save the updated file
            write_json(file_path, response_data)
            print(f"  Updated: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _process_flowcharts(
        self, prompt_index: str, models: List[str], property_checker_names: List[str]
    ):
        """Process all flowchart files for the given prompt."""
        print(f"\nProcessing flowcharts for prompt: {prompt_index}")

        # Process flowcharts for each model
        base_config_name = self.config._name_
        f_config_name = self.config.f._name_

        import re

        f_config_base = f_config_name
        model_suffix_pattern = r"_[a-z0-9_\.]+$"
        match = re.search(model_suffix_pattern, f_config_base)
        if match:
            # Check if it looks like a model name (has numbers or multiple parts)
            potential_model = match.group(0)[1:]  # Remove leading underscore
            # If it has numbers or multiple underscores, it's likely a model name
            if re.search(r"\d", potential_model) or potential_model.count("_") >= 2:
                f_config_base = f_config_base[: match.start()]

        for model in models:
            # Try to find flowchart with model name in it
            # Format: config-{config_name}-{f_config_base}_{model}_flowchart.json
            # Model names like "claude-sonnet-3.7" need to be converted to "claude_sonnet_3.7"
            model_safe = model.replace("/", "_").replace("-", "_")

            # First try: config-{base_config_name}-{f_config_base}_{model}_flowchart.json
            flowchart_path = FileUtils.get_flowchart_file_path(
                prompt_index, base_config_name, f_config_base, [model]
            )

            print(f"Looking for flowchart: {flowchart_path}")
            if not os.path.exists(flowchart_path):
                # Use glob to find the correct flowchart
                import glob

                pattern = f"flowcharts/{prompt_index}/config-{base_config_name}-{f_config_base}*_flowchart.json"
                matches = glob.glob(pattern)
                # Filter out condensed versions
                matches = [m for m in matches if "_condensed" not in m]

                if matches:
                    # Find flowchart that contains this model name but NOT other model names
                    all_models_safe = [m.replace("/", "_").replace("-", "_") for m in models]
                    model_matches = []
                    for match in matches:
                        # Check if this match contains our model
                        if model_safe in match:
                            # Check if it contains any OTHER model names (which would be wrong)
                            has_other_model = False
                            for other_model_safe in all_models_safe:
                                if other_model_safe != model_safe and other_model_safe in match:
                                    has_other_model = True
                                    break
                            if not has_other_model:
                                model_matches.append(match)

                    if model_matches:
                        flowchart_path = model_matches[0]
                        print(f"Found flowchart via glob: {flowchart_path}")
                    else:
                        print(f"Flowchart file not found for model {model} (searched: {matches})")
                        continue
                else:
                    print(f"Flowchart file not found for model {model}: {flowchart_path}")
                    continue

            print(f"Processing flowchart for model {model}: {flowchart_path}")
            self._process_single_flowchart(
                flowchart_path, prompt_index, models, property_checker_names
            )

    def _process_single_flowchart(
        self,
        flowchart_path: str,
        prompt_index: str,
        models: List[str],
        property_checker_names: List[str],
    ):
        """Process a single flowchart file and update properties in responses."""
        try:
            print(f"Processing flowchart: {flowchart_path}")

            # Load the flowchart
            flowchart_data = load_json(flowchart_path)

            if not flowchart_data or "responses" not in flowchart_data:
                print(f"Skipping flowchart without responses: {flowchart_path}")
                return

            responses = flowchart_data["responses"]
            updated_count = 0

            # Process each response in the flowchart
            for response_id, response_data in responses.items():
                if self._update_flowchart_response_properties(
                    response_data, prompt_index, property_checker_names
                ):
                    updated_count += 1

            if updated_count > 0:
                # Save the updated flowchart
                write_json(flowchart_path, flowchart_data)
                print(f"  Updated {updated_count} responses in flowchart")
            else:
                print(f"  No updates needed for flowchart")

        except Exception as e:
            print(f"Error processing flowchart {flowchart_path}: {e}")

    def _update_flowchart_response_properties(
        self, response_data: Dict[str, Any], prompt_index: str, property_checker_names: List[str]
    ) -> bool:
        """Update properties for a single response in a flowchart. Returns True if any properties were updated."""
        try:
            # Treat missing or None resampled as rollout (False)
            resampled = response_data["resampled"] if "resampled" in response_data else False

            # Get the seed
            seed = response_data.get("seed")
            if seed is None:
                print(f"    Warning: seed field is None for response, skipping")
                return False

            # Determine the source file path based on resampled status
            if resampled is False:
                # This is a rollout - find the model from the flowchart
                # We need to determine which model this response came from
                # For now, we'll try all models and see which file exists
                source_file_path = None
                for model in self.config["models"]:
                    potential_path = FileUtils.get_rollout_file_path(prompt_index, model, seed)
                    if os.path.exists(potential_path):
                        source_file_path = potential_path
                        break

                if not source_file_path:
                    print(f"    Warning: Could not find rollout file for seed {seed}")
                    return False
            else:
                # This is a resample - resampled should contain the prefix
                prefix = resampled
                # We need to determine which model this response came from
                source_file_path = None
                for model in self.config["models"]:
                    potential_path = FileUtils.get_resample_file_path(prompt_index, model, prefix, seed)
                    if os.path.exists(potential_path):
                        source_file_path = potential_path
                        break

                if not source_file_path:
                    print(
                        f"    Warning: Could not find resample file for seed {seed}, prefix {prefix}"
                    )
                    return False

            # Load the source file to get the updated properties
            source_data = load_json(source_file_path)
            if not source_data:
                print(f"    Warning: Could not load source file: {source_file_path}")
                return False

            # Update properties in the flowchart response
            updated = False
            for checker_name in property_checker_names:
                if checker_name in source_data:
                    new_value = source_data[checker_name]
                    old_value = response_data.get(checker_name)
                    if old_value != new_value:
                        response_data[checker_name] = new_value
                        updated = True
                        print(f"    Updated {checker_name}: {old_value} -> {new_value}")

            return updated

        except Exception as e:
            print(f"    Error updating flowchart response properties: {e}")
            return False

    def _process_flowchart_node_properties(
        self, prompt_index: str, node_property_checker_names: List[str]
    ):
        """Recompute and update node property checkers in the flowchart."""
        print(f"\nProcessing node property checkers for prompt: {prompt_index}")

        base_config_name = self.config._name_
        f_config_name = self.config.f._name_

        flowchart_path = FileUtils.get_flowchart_file_path(
            prompt_index, base_config_name, f_config_name
        )


        if not os.path.exists(flowchart_path):
            print(f"Flowchart file not found: {flowchart_path}")
            return

        flowchart_data = load_json(flowchart_path)
        if not flowchart_data or "nodes" not in flowchart_data:
            print(f"Flowchart file does not contain nodes: {flowchart_path}")
            return

        registry = {
            name: cls for name, cls in PROPERTY_CHECKER_REGISTRY.items()
            if hasattr(cls, "get_value_for_node")
        }

        node_checkers = {}
        for checker_name in node_property_checker_names:
            if checker_name in registry:
                node_checkers[checker_name] = registry[checker_name]()
            else:
                print(f"  Warning: Node property checker {checker_name} not found in registry")

        if not node_checkers:
            print("  No valid node property checkers found")
            return

        updated_nodes = 0
        for node_obj in flowchart_data["nodes"]:
            cluster_key = list(node_obj.keys())[0]
            node_data = node_obj[cluster_key]
            sentences_data = node_data.get("sentences", [])

            if not sentences_data:
                continue

            unique_sentences = []
            sentence_map = {}
            for sentence_item in sentences_data:
                sentence_text = sentence_item.get("text", "")
                if sentence_text and sentence_text not in sentence_map:
                    unique_sentences.append(sentence_text)
                    sentence_map[sentence_text] = []

            if not unique_sentences:
                continue

            for checker_name, checker in node_checkers.items():
                if checker_name == "multi_algorithm":
                    values = checker.get_value_for_node(unique_sentences, prompt_index)
                else:
                    values = checker.get_value_for_node(unique_sentences)

                for idx, sentence_text in enumerate(unique_sentences):
                    if idx < len(values):
                        sentence_map[sentence_text].append((checker_name, values[idx]))

            for sentence_item in sentences_data:
                sentence_text = sentence_item.get("text", "")
                if sentence_text in sentence_map:
                    for checker_name, value in sentence_map[sentence_text]:
                        sentence_item[checker_name] = value

            updated_nodes += 1

        if updated_nodes > 0:
            write_json(flowchart_path, flowchart_data)
            print(f"  Updated {updated_nodes} nodes with node property checkers")
        else:
            print("  No nodes needed updating")

    def _process_files_batch_parallel(
        self, file_paths: List[str], prompt_index: str, property_checker_names: List[str]
    ):
        """Process multiple files in parallel batches for property checkers that support it."""
        if not file_paths:
            return

        # Separate checkers into those that support parallel processing and those that don't
        parallel_checkers = []
        sequential_checkers = []

        for checker_name in property_checker_names:
            if checker_name in self.property_checkers:
                checker = self.property_checkers[checker_name]
                if hasattr(checker, "process_responses_parallel"):
                    parallel_checkers.append(checker_name)
                else:
                    sequential_checkers.append(checker_name)
            else:
                sequential_checkers.append(checker_name)

        # Load all files once
        file_data_map = {}
        files_to_process = []

        for file_path in file_paths:
            try:
                response_data = load_json(file_path)
                if response_data:
                    # Check if file already has all required properties with valid values
                    if not self.recompute:
                        has_all_properties = all(
                            prop in response_data
                            and response_data[prop] not in ["unknown", "None", None, ""]
                            for prop in property_checker_names
                        )
                        if has_all_properties:
                            print(f"File already has all properties: {file_path}")
                            continue

                    file_data_map[file_path] = response_data
                    files_to_process.append(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not files_to_process:
            print("  No files need processing")
            return

        print(f"  Processing {len(files_to_process)} files...")

        # Process parallel checkers in batches
        if parallel_checkers:
            print(f"  Using parallel processing for: {', '.join(parallel_checkers)}")
            for checker_name in parallel_checkers:
                checker = self.property_checkers[checker_name]

                # Prepare data for parallel processing
                response_data_list = []
                valid_file_paths = []
                skipped_file_paths = []

                for file_path in files_to_process:
                    response_data = file_data_map[file_path]
                    cot = response_data.get("cot_content", "")
                    response = response_data.get("response_content", "")

                    # For algorithm and single_algorithm, check if this is a flowchart (skip those in parallel)
                    if (
                        checker_name in ["algorithm", "single_algorithm"]
                        and "flowcharts" in file_path
                    ):
                        # Skip flowcharts in parallel processing, handle them sequentially
                        skipped_file_paths.append(file_path)
                        continue

                    # Only process if there's content
                    if cot or response:
                        response_data_list.append(response_data)
                        valid_file_paths.append(file_path)

                # Store skipped files for later sequential processing
                if not hasattr(self, "_skipped_files"):
                    self._skipped_files = {}
                if checker_name not in self._skipped_files:
                    self._skipped_files[checker_name] = []
                self._skipped_files[checker_name].extend(skipped_file_paths)

                if response_data_list:
                    # Process in batches of 75
                    batch_size = 75
                    total_files = len(response_data_list)
                    print(
                        f"    Processing {total_files} files with {checker_name} in batches of {batch_size}..."
                    )

                    for batch_start in range(0, total_files, batch_size):
                        batch_end = min(batch_start + batch_size, total_files)
                        batch_data = response_data_list[batch_start:batch_end]
                        batch_paths = valid_file_paths[batch_start:batch_end]

                        print(
                            f"      Processing batch {batch_start // batch_size + 1} ({len(batch_data)} files)..."
                        )
                        try:
                            results = checker.process_responses_parallel(batch_data, prompt_index)

                            # Update results back to file data
                            for i, file_path in enumerate(batch_paths):
                                if i < len(results):
                                    file_data_map[file_path][checker_name] = results[i]
                                    print(
                                        f"        {checker_name} for {os.path.basename(file_path)}: {results[i]}"
                                    )
                        except Exception as e:
                            print(
                                f"      Error in parallel processing batch for {checker_name}: {e}"
                            )
                            # Fall back to sequential for this batch
                            for file_path in batch_paths:
                                try:
                                    response_data = file_data_map[file_path]
                                    value = checker.get_value(
                                        response_data, prompt_index, file_path
                                    )
                                    file_data_map[file_path][checker_name] = value
                                    print(
                                        f"        {checker_name} for {os.path.basename(file_path)}: {value}"
                                    )
                                except Exception as e2:
                                    print(
                                        f"        Error processing {file_path} with {checker_name}: {e2}"
                                    )
                                    file_data_map[file_path][checker_name] = "None"

        # Process sequential checkers one by one, and any parallel checkers that were skipped
        # Check for any parallel checkers that have files that were skipped (e.g., flowcharts for single_algorithm)
        if hasattr(self, "_skipped_files"):
            for checker_name in parallel_checkers:
                if checker_name in self._skipped_files and self._skipped_files[checker_name]:
                    checker = self.property_checkers[checker_name]
                    unprocessed_files = self._skipped_files[checker_name]

                    # Process skipped files sequentially for this checker
                    print(
                        f"  Processing {len(unprocessed_files)} skipped files sequentially for {checker_name}..."
                    )
                    for file_path in unprocessed_files:
                        if file_path in file_data_map:
                            response_data = file_data_map[file_path]
                            if checker_name not in response_data:  # Only process if not already set
                                try:
                                    value = checker.get_value(
                                        response_data, prompt_index, file_path
                                    )
                                    response_data[checker_name] = value
                                    print(
                                        f"      {checker_name} for {os.path.basename(file_path)}: {value}"
                                    )
                                except Exception as e:
                                    print(
                                        f"      Error processing {os.path.basename(file_path)} with {checker_name}: {e}"
                                    )
                                    response_data[checker_name] = "None"
            # Clear skipped files for next batch
            self._skipped_files = {}

        if sequential_checkers:
            print(f"  Using sequential processing for: {', '.join(sequential_checkers)}")
            for file_path in files_to_process:
                response_data = file_data_map[file_path]
                for checker_name in sequential_checkers:
                    if checker_name in self.property_checkers:
                        checker = self.property_checkers[checker_name]
                        try:
                            value = checker.get_value(response_data, prompt_index, file_path)
                            response_data[checker_name] = value
                            print(
                                f"      {checker_name} for {os.path.basename(file_path)}: {value}"
                            )
                        except Exception as e:
                            print(
                                f"      Error processing {os.path.basename(file_path)} with {checker_name}: {e}"
                            )
                            response_data[checker_name] = "None"

        # Save all updated files
        for file_path in files_to_process:
            try:
                write_json(file_path, file_data_map[file_path])
            except Exception as e:
                print(f"  Error saving {file_path}: {e}")

    def _process_algorithm_parallel(
        self, file_paths: List[str], prompt_index: str
    ) -> Dict[str, str]:
        """Process algorithm detection for multiple files in parallel."""
        if "algorithm" not in self.property_checkers:
            return {}

        algorithm_checker = self.property_checkers["algorithm"]

        # Load all response data
        response_data_list = []
        valid_indices = []

        for i, file_path in enumerate(file_paths):
            try:
                response_data = load_json(file_path)
                if (
                    response_data
                    and response_data.get("cot_content")
                    and response_data.get("response_content")
                ):
                    response_data_list.append(response_data)
                    valid_indices.append(i)
                else:
                    response_data_list.append(None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                response_data_list.append(None)

        # Process valid responses in parallel
        valid_responses = [r for r in response_data_list if r is not None]
        if not valid_responses:
            return {}

        print(f"    Processing {len(valid_responses)} files in parallel for algorithm detection...")
        algorithm_results = algorithm_checker.process_responses_parallel(
            valid_responses, prompt_index
        )

        # Map results back to file paths
        results = {}
        valid_idx = 0
        for i, file_path in enumerate(file_paths):
            if response_data_list[i] is not None:
                results[file_path] = algorithm_results[valid_idx]
                valid_idx += 1
            else:
                results[file_path] = "None"

        return results

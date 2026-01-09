#!/usr/bin/env python3
"""
Flowchart generator that orchestrates clustering and creates flowcharts.
"""

from pathlib import Path
from typing import Dict, Any, List
import math
from collections import Counter

from src.utils.json_utils import load_json, write_json
from src.utils.file_utils import FileUtils
from src.utils.prompt_utils import get_prompt_filter
from src.clustering import ActivationClusterer, SentenceThenLLMClusterer
from src.labeling.cluster_labeler import label_flowchart, load_prompt_text
from src.property_checkers import PropertyCheckerCorrectness, PropertyCheckerResampled, PropertyCheckerModel, PropertyCheckerAlgorithm, PropertyCheckerSingleAlgorithm, PropertyCheckerProfessor, PropertyCheckerDebugMultiAlgorithm, PropertyCheckerMultiAlgorithm


class FlowchartGenerator:
    """Generates flowcharts from responses using various clustering methods."""

    def __init__(self):
        self.property_checkers = {
            "correctness": PropertyCheckerCorrectness(),
            "resampled": PropertyCheckerResampled(),
            "model": PropertyCheckerModel(),
            "algorithm": PropertyCheckerAlgorithm(),
            "single_algorithm": PropertyCheckerSingleAlgorithm(),
            "debug_multi_algorithm": PropertyCheckerDebugMultiAlgorithm(),
            "multi_algorithm": PropertyCheckerMultiAlgorithm(),
            "professor": PropertyCheckerProfessor(),
        }

    def _calculate_edge_entropy(self, edge: Dict[str, str],
                                responses: Dict[str, Any]) -> float:
        """Calculate entropy of an edge based on answer distribution of rollouts going through it."""
        node_a = edge["node_a"]
        node_b = edge["node_b"]

        # Find all rollouts that go through this specific edge
        rollouts_through_edge = []
        for rollout_id, rollout_data in responses.items():
            edges = rollout_data.get("edges", [])
            for rollout_edge in edges:
                if (rollout_edge.get("node_a") == node_a
                        and rollout_edge.get("node_b") == node_b):
                    # Extract answer from this rollout
                    answer = rollout_data.get("processed_response_content",
                                              "") or rollout_data.get(
                                                  "answer", "")
                    if answer:
                        rollouts_through_edge.append(answer)
                    break

        if not rollouts_through_edge:
            return 0.0

        # Calculate entropy of answer distribution
        answer_counts = Counter(rollouts_through_edge)
        total = len(rollouts_through_edge)
        entropy = 0.0

        for count in answer_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _add_edge_entropy(self, flowchart: Dict[str, Any]):
        """Add entropy to all edges in the flowchart."""
        responses = flowchart["responses"]

        # Collect all unique edges
        all_edges = set()
        for rollout_data in responses.values():
            edges = rollout_data.get("edges", [])
            for edge in edges:
                edge_key = (edge["node_a"], edge["node_b"])
                all_edges.add(edge_key)

        # Calculate entropy for each unique edge
        edge_entropies = {}
        for node_a, node_b in all_edges:
            edge = {"node_a": node_a, "node_b": node_b}
            entropy = self._calculate_edge_entropy(edge, responses)
            edge_entropies[(node_a, node_b)] = entropy

        # Add entropy to all edges in all rollouts
        for rollout_data in responses.values():
            edges = rollout_data.get("edges", [])
            for edge in edges:
                edge_key = (edge["node_a"], edge["node_b"])
                if edge_key in edge_entropies:
                    edge["entropy"] = edge_entropies[edge_key]
                else:
                    edge["entropy"] = 0.0

    def _create_clusterer(self, config_f: Dict[str, Any]):
        """Create the appropriate clusterer based on method in config.f."""
        method = config_f.get("method", "sentence")

        if method == "sentence":
            return SentenceThenLLMClusterer(config_f)
        elif method == "activation":
            return ActivationClusterer(config_f)
        elif method == "sentence_then_llm":
            return SentenceThenLLMClusterer(config_f)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _print_flowchart_summary(self, flowchart: Dict[str, Any]):
        """Print a summary of the generated flowchart."""
        n_clusters = len(flowchart['nodes'])
        total_responses = len(flowchart['responses'])
        print(f"\nFlowchart Summary:")
        print(f"  Method: {flowchart['clustering_method']}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Total responses: {total_responses}")

        print(f"\nCluster details:")
        for node_obj in flowchart["nodes"]:
            cluster_key = list(node_obj.keys())[0]
            node_data = node_obj[cluster_key]
            freq = node_data["freq"]
            correct_count = 0
            for seed, rollout_info in flowchart["responses"].items():
                edges = rollout_info.get("edges", [])
                for edge in edges:
                    if edge["node_a"] == cluster_key or edge[
                            "node_b"] == cluster_key:
                        if rollout_info.get("correctness", False):
                            correct_count += 1
                        break
            correctness_rate = correct_count / freq if freq > 0 else 0.0
            print(
                f"  {cluster_key}: {freq} responses, {correctness_rate:.1%} correct"
            )

    def generate_flowchart_from_config(self, config: Dict[str, Any]):
        """Generate flowchart from config object."""

        prompt_index = config["prompt"]
        models = list(config["models"])
        subset_seeds = config.get("subset_seeds", None)

        method = config.f.method

        if subset_seeds is None:
            num_seeds_rollouts = config.f.num_seeds_rollouts
            if num_seeds_rollouts:
                subset_seeds = list(range(num_seeds_rollouts))

        print(f"Generating flowchart with method: {method}")
        print(f"Models: {models}")
        print(f"Prompt: {prompt_index}")
        if subset_seeds:
            print(
                f"Using {len(subset_seeds)} responses (seeds 0-{len(subset_seeds)-1})"
            )

        print(
            f"\n--- Generating flowchart for {prompt_index} with models {models} ---"
        )
        self._generate_flowchart_for_prompt_models(prompt_index, models,
                                                   method, config,
                                                   subset_seeds)

    def _generate_flowchart_for_prompt_models(self,
                                              prompt_index: str,
                                              models: List[str],
                                              method: str,
                                              config: Dict[str, Any],
                                              subset_seeds: List[int] = None):
        """Generate flowchart for a prompt with multiple models."""

        flowchart_path = FileUtils.get_flowchart_file_path(
            prompt_index, config._name_, config.f._name_)

        if FileUtils.file_exists(flowchart_path):
            print(
                f"Flowchart already exists at {flowchart_path}, skipping generation"
            )
            return

        all_responses = self._load_responses(models, prompt_index,
                                             subset_seeds, config,
                                             list(config.property_checkers))

        if not all_responses:
            print(
                f"No responses found for {prompt_index} with models {models}")
            return

        print(f"Loaded {len(all_responses)} responses for models {models}")

        clusterer = self._create_clusterer(config.f)

        cluster_assignments = clusterer.cluster_responses(
            all_responses, prompt_index, models)

        if not cluster_assignments:
            print("No cluster assignments generated")
            return

        node_property_checker_names = config.get("node_property_checkers", [])

        flowchart = clusterer.create_flowchart(all_responses,
                                               cluster_assignments,
                                               prompt_index, models,
                                               config.f._name_,
                                               list(config.property_checkers),
                                               node_property_checker_names)

        flowchart["property_checkers"] = list(config.property_checkers)

        for seed, response_info in flowchart["responses"].items():
            edges = response_info.get("edges", [])
            if edges:
                start_edge = {"node_a": "START", "node_b": edges[0]["node_a"]}
                edges.insert(0, start_edge)

        # Add entropy to all edges
        self._add_edge_entropy(flowchart)

        start_node_exists = any("START" in str(node)
                                for node in flowchart["nodes"])
        if not start_node_exists:
            flowchart["nodes"].append({
                "START": {
                    "freq": 0,
                    "representative_sentence": "START",
                    "mean_similarity": 1.0,
                    "sentences": []
                }
            })

        # Save final flowchart without graph layout
        FileUtils.ensure_dir(Path(flowchart_path).parent)
        write_json(flowchart_path, flowchart)

        print(f"Saved flowchart to {flowchart_path}")
        print(
            "Note: Use graphviz_generator.py to add graph layout visualization"
        )

        self._print_flowchart_summary(flowchart)

    def _load_responses(
            self,
            models: List[str],
            prompt_index: str,
            subset_seeds: List[int] = None,
            config: Dict[str, Any] = None,
            property_checker_names: List[str] = None) -> Dict[str, Any]:
        """Load responses from individual JSON files for multiple models."""
        all_responses = {}
        current_index = 0

        for model in models:
            for seed in range(config.f.num_seeds_rollouts):
                if subset_seeds is not None and seed not in subset_seeds:
                    continue

                rollout_path = FileUtils.get_rollout_file_path(
                    prompt_index, model, seed)
                if FileUtils.file_exists(rollout_path):
                    response_data = load_json(rollout_path)

                    if not response_data.get("response_content", "").strip():
                        print(f"Skipping empty response for seed {seed}")
                        continue

                    self._apply_property_checkers(response_data, rollout_path,
                                                  prompt_index,
                                                  property_checker_names)
                    response_data["seed"] = seed
                    all_responses[str(current_index)] = response_data
                    current_index += 1

        for model in models:
            prefixes = config.get("prefixes", [])
            print(f"DEBUG: Found prefixes: {prefixes}")
            for prefix in prefixes:
                prefix_index = int(prefix.split("-")[-1])
                print(
                    f"DEBUG: Processing prefix {prefix} (index {prefix_index})"
                )
                for seed in range(config.f.num_seeds_prefixes):
                    if subset_seeds is not None and seed not in subset_seeds:
                        continue

                    resample_path = FileUtils.get_resample_file_path(
                        prompt_index, model, prefix, seed)
                    print(f"DEBUG: Checking resample path: {resample_path}")
                    if FileUtils.file_exists(resample_path):
                        print(
                            f"DEBUG: Loading resample: {prefix}, seed {seed}")
                        response_data = load_json(resample_path)

                        if not response_data.get("response_content",
                                                 "").strip():
                            print(
                                f"DEBUG: Skipping empty response for {prefix}, seed {seed}"
                            )
                            continue

                        self._apply_property_checkers(response_data,
                                                      resample_path,
                                                      prompt_index,
                                                      property_checker_names)
                        response_data["seed"] = seed
                        all_responses[str(current_index)] = response_data
                        current_index += 1
                        print(
                            f"DEBUG: Added resample at index {current_index-1}"
                        )
                    else:
                        print(
                            f"DEBUG: Resample file not found: {resample_path}")

        print(f"Total responses loaded: {len(all_responses)}")
        print(f"Response keys: {list(all_responses.keys())}")

        return all_responses

    def _apply_property_checkers(self, response_data: Dict[str, Any],
                                 file_path: str, prompt_index: str,
                                 property_checker_names: List[str]):
        """Extract property checker values from response data. If key doesn't exist, set to None."""
        for checker_name in property_checker_names:
            if checker_name in self.property_checkers:
                checker = self.property_checkers[checker_name]
                value = checker.get_value(response_data, prompt_index,
                                          file_path)
                response_data[checker_name] = value
            else:
                print(
                    f"Property checker {checker_name} not found in available checkers: {list(self.property_checkers.keys())}"
                )


class LabelGenerator:
    """Generates labels for an existing flowchart using the LLM labeler."""

    def generate_labels_from_config(self, config: Dict[str, Any]) -> None:
        prompt_index = config["prompt"]
        flowchart_path = FileUtils.get_flowchart_file_path(
            prompt_index, config._name_, config.f._name_)

        flowchart = load_json(flowchart_path)

        prompt_text = load_prompt_text("prompts/prompts.json", prompt_index)
        mw = int(config.f.max_workers)

        flowchart = label_flowchart(flowchart, prompt_text, max_workers=mw)

        FileUtils.ensure_dir(Path(flowchart_path).parent)
        write_json(flowchart_path, flowchart)
        print(f"Labeled flowchart written to {flowchart_path}")

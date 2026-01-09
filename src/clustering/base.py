"""
Base clustering classes for generating flowcharts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import numpy as np

from src.utils.json_utils import load_json


class BaseClusterer(ABC):
    """Base class for all clustering algorithms."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def cluster_responses(self, responses: Dict[str, Any], prompt_index: str,
                          models: List[str]) -> Dict[str, Any]:
        """Cluster responses and return cluster assignments."""
        pass

    def create_flowchart(
            self,
            responses: Dict[str, Any],
            cluster_assignments: Dict[str, int],
            prompt_index: str,
            models: List[str],
            config_name: str,
            property_checker_names: List[str] = None,
            node_property_checker_names: List[str] = None) -> Dict[str, Any]:
        """Create flowchart from clustered responses."""

        clusters = {}
        for seed, cluster_id in cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(seed)

        # Load prompt text
        from src.labeling.cluster_labeler import load_prompt_text
        prompt_text = load_prompt_text("prompts/prompts.json", prompt_index)

        # Load algorithms
        from pathlib import Path
        algorithms_path = Path("prompts/algorithms.json")
        algorithms = {}
        if algorithms_path.exists():
            algorithms_data = load_json(str(algorithms_path))
            algorithms = algorithms_data.get(prompt_index, {})

        flowchart = {
            "prompt_index": prompt_index,
            "prompt": prompt_text,
            "algorithms": algorithms,
            "models": models,
            "config_name": config_name,
            "clustering_method": self.config.get("method", "unknown"),
            "nodes": [],
            "responses": [],
            "graph_layout": {}
        }

        for cluster_id, seeds in clusters.items():
            cluster_key = f"cluster-{cluster_id}"
            node_data = {
                cluster_key: {
                    "freq":
                    len(seeds),
                    "representative_sentence":
                    self._get_representative_response(responses, seeds),
                    "mean_similarity":
                    self._calculate_mean_similarity(responses, seeds),
                    "num_rollouts":
                    self._calculate_num_rollouts(responses, seeds),
                    "entropy":
                    self._calculate_entropy(responses, seeds),
                    "sentences":
                    self._create_sentence_breakdown(responses, seeds)
                }
            }
            flowchart["nodes"].append(node_data)

        for seed, response_data in responses.items():
            cluster_id = None
            for cid, seeds in clusters.items():
                if seed in seeds:
                    cluster_id = cid
                    break

            if cluster_id is not None:
                rollout_data = {
                    seed: {
                        "index":
                        seed,
                        "seed":
                        response_data.get("seed", None),
                        "answer":
                        response_data.get("processed_response_content", ""),
                        "edges":
                        self._create_edges_for_response(
                            response_data, cluster_id)
                    }
                }

                if property_checker_names:
                    for checker_name in property_checker_names:
                        rollout_data[seed][checker_name] = response_data.get(
                            checker_name, None)

                flowchart["responses"].append(rollout_data)

        return flowchart

    def _get_representative_response(self, responses: Dict[str, Any],
                                     seeds: List[str]) -> str:
        """Get a representative response for the cluster."""
        if not seeds:
            return ""

        first_seed = seeds[0]
        if first_seed in responses:
            return responses[first_seed].get("response_content", "")
        return ""

    def _calculate_mean_similarity(self, responses: Dict[str, Any],
                                   seeds: List[str]) -> float:
        """Calculate mean cosine similarity between all sentence embeddings in the cluster."""
        if not seeds or len(seeds) < 2:
            return 1.0

        all_embeddings = []
        for seed in seeds:
            if seed in responses:
                embeddings = responses[seed].get("sentence_embeddings", [])
                if embeddings:
                    all_embeddings.extend(embeddings)

        if len(all_embeddings) < 2:
            return 1.0

        embeddings_array = np.array(all_embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings_array / norms

        similarities = np.dot(normalized, normalized.T)

        upper_triangle = np.triu_indices_from(similarities, k=1)
        if len(upper_triangle[0]) == 0:
            return 1.0

        mean_sim = float(np.mean(similarities[upper_triangle]))
        return mean_sim

    def _calculate_num_rollouts(self, responses: Dict[str, Any],
                                seeds: List[str]) -> int:
        """Calculate number of unique rollouts that pass through this cluster."""
        unique_rollouts = set()
        for seed in seeds:
            if seed in responses:
                unique_rollouts.add(seed)
        return len(unique_rollouts)

    def _calculate_entropy(self, responses: Dict[str, Any],
                           seeds: List[str]) -> float:
        """Calculate Shannon entropy from answer distribution of unique rollouts."""
        from collections import Counter
        import math

        answers = []
        seen_rollouts = set()
        for seed in seeds:
            if seed in responses and seed not in seen_rollouts:
                answer = responses[seed].get("processed_response_content", "")
                if answer:
                    answers.append(answer)
                seen_rollouts.add(seed)

        if not answers:
            return 0.0

        answer_counts = Counter(answers)
        total = len(answers)
        entropy = 0.0
        for count in answer_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def _get_content_key(self) -> str:
        """Get the content key to use based on config flag."""
        use_sentences = self.config.get("sentences_instead_of_chunks", False)
        return "sentences" if use_sentences else "chunked_cot_content"

    def _create_sentence_breakdown(self, responses: Dict[str, Any],
                                   seeds: List[str]) -> List[Dict[str, Any]]:
        """Create detailed sentence breakdown for the cluster."""
        sentence_counts = {}
        content_key = self._get_content_key()

        for seed in seeds:
            if seed in responses:
                response_data = responses[seed]
                sentences = response_data.get(content_key, [])
                for sentence in sentences:
                    if sentence in sentence_counts:
                        sentence_counts[sentence]["count"] += 1
                    else:
                        sentence_counts[sentence] = {
                            "text": sentence,
                            "count": 1
                        }

        sentence_list = list(sentence_counts.values())
        sentence_list.sort(key=lambda x: x["count"], reverse=True)

        return sentence_list

    def _create_edges_for_response(self, response_data: Dict[str, Any],
                                   cluster_id: int) -> List[Dict[str, str]]:
        """Create edge information for a response."""
        return [{"node_a": str(cluster_id), "node_b": "0"}]

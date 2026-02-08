from .base import PropertyCheckerMulti
import json
from pathlib import Path
from typing import List, Dict, Any
from src.utils.file_utils import FileUtils


class PropertyCheckerMultiAlgorithm(PropertyCheckerMulti):
    """General multi-algorithm property checker that dynamically reads algorithms and cues from algorithms.json."""

    registry_name = "multi_algorithm"

    def __init__(self, max_workers: int = 250):
        self.max_workers = max_workers
        self.prompt_index_cache: Dict[str, Dict[str, List[str]]] = {}

    def _load_algorithms(self, prompt_index: str) -> dict:
        """Load algorithms for the given prompt index from algorithms.json."""
        algorithms_path = Path(FileUtils.get_algorithms_file_path())
        with open(algorithms_path, "r") as f:
            algorithms_data = json.load(f)
        if prompt_index not in algorithms_data:
            raise ValueError(f"No algorithms found for prompt index: {prompt_index}")
        return algorithms_data[prompt_index]

    def _extract_cues_map(self, algorithms: dict) -> Dict[str, List[str]]:
        """Extract cues map from algorithms dictionary.

        Handles two JSON structures:
        1. Structured: {"A": {"cues": [...]}, "B": {"cues": [...]}}
        2. Simple: {"0": "description", "1": "description"} - raises error as cues required
        """
        cues_map = {}
        for alg_key, alg_value in algorithms.items():
            if isinstance(alg_value, dict) and "cues" in alg_value:
                cues_map[alg_key] = alg_value["cues"]
            elif isinstance(alg_value, str):
                raise ValueError(
                    f"Algorithm {alg_key} does not have cues defined. Only structured algorithms with 'cues' arrays are supported."
                )
            else:
                raise ValueError(
                    f"Algorithm {alg_key} has invalid structure. Expected dict with 'cues' array."
                )
        return cues_map

    def _first_index_with_any(self, sentences: List[str], patterns: List[str]) -> int:
        """Return 1-indexed first sentence index that contains any pattern (case-insensitive), or 0 if none."""
        lowered_patterns = [p.lower() for p in patterns]
        for i, s in enumerate(sentences, start=1):
            ls = s.lower()
            for p in lowered_patterns:
                if p in ls:
                    return i
        return 0

    def _get_cues_for_algorithm(self, cues_map: Dict[str, List[str]], alg: str) -> List[str]:
        """Get the cues list for a given algorithm identifier."""
        return cues_map.get(alg, [])

    def _heuristic_keywords_output(
        self, sentences: List[str], cues_map: Dict[str, List[str]]
    ) -> str:
        """Keyword baseline: scan sentence-by-sentence cues for algorithms and allow up to 10 switches.
        Returns a compact JSON array string: ["A"], ["B"], ["A", k, "B"], etc.
        Returns "None" if undecidable (no cues found).
        """
        algorithm_labels = list(cues_map.keys())

        current_alg: str = ""
        initial_alg: str = ""
        boundaries: List[int] = []
        alg_sequence: List[str] = []

        for idx, s in enumerate(sentences, start=1):
            alg_cues_found = {}

            for alg in algorithm_labels:
                cues = self._get_cues_for_algorithm(cues_map, alg)
                if not cues:
                    continue
                has_cue = any(p.lower() in s.lower() for p in cues)
                if has_cue:
                    positions = [s.lower().find(p.lower()) for p in cues if p.lower() in s.lower()]
                    alg_cues_found[alg] = min(positions) if positions else len(s) + 1

            if not alg_cues_found:
                continue

            if len(alg_cues_found) > 1:
                continue

            this_alg = min(alg_cues_found.items(), key=lambda x: x[1])[0]

            if not initial_alg:
                initial_alg = this_alg
                current_alg = this_alg
                alg_sequence = [this_alg]
                continue

            if this_alg != current_alg:
                if len(boundaries) < 10:
                    boundaries.append(idx)
                    alg_sequence.append(this_alg)
                    current_alg = this_alg
                else:
                    break

        if not initial_alg:
            return "None"

        filtered_boundaries = []
        filtered_alg_sequence = []
        prev_boundary = 0

        for i in range(len(boundaries)):
            if i == 0:
                if boundaries[0] >= 3:  # at least 3 sentences between boundaries
                    filtered_boundaries.append(boundaries[0])
                    filtered_alg_sequence.append(alg_sequence[i + 1])
                else:
                    continue
            else:
                boundary_distance = boundaries[i] - boundaries[i - 1]
                # if boundary_distance >= 3: # we actually filter over in AlgorithmStructure.tsx
                filtered_boundaries.append(boundaries[i])
                filtered_alg_sequence.append(alg_sequence[i + 1])
                # else:
                #     continue

        return_string = f'["{initial_alg}"'
        for i in range(len(filtered_boundaries)):
            return_string += f', {filtered_boundaries[i]}, "{filtered_alg_sequence[i]}"'
        return_string += "]"
        return return_string

    def get_value_for_node(self, sentences: List[str], prompt_index: str = None) -> List[List[str]]:
        """For each sentence, return a list of algorithms present in that sentence.

        Args:
            sentences: List of sentence strings to analyze
            prompt_index: Optional prompt index to load algorithms from JSON.
                         If not provided, will attempt to use cached cues_map from get_value calls.

        Returns:
            List of lists, where each inner list contains the algorithm identifiers
            (e.g., ["A"] or ["A", "B"]) found in the corresponding sentence.
            If no algorithms are found in a sentence, returns an empty list [].
        """
        if not prompt_index:
            if not self.prompt_index_cache:
                return [[] for _ in sentences]
            cues_map = next(iter(self.prompt_index_cache.values()))
        else:
            if prompt_index not in self.prompt_index_cache:
                algorithms = self._load_algorithms(prompt_index)
                cues_map = self._extract_cues_map(algorithms)
                self.prompt_index_cache[prompt_index] = cues_map
            else:
                cues_map = self.prompt_index_cache[prompt_index]

        algorithm_labels = list(cues_map.keys())
        result = []

        for s in sentences:
            found_algorithms = []

            for alg in algorithm_labels:
                cues = self._get_cues_for_algorithm(cues_map, alg)
                if not cues:
                    continue
                has_cue = any(p.lower() in s.lower() for p in cues)
                if has_cue:
                    found_algorithms.append(alg)

            result.append(found_algorithms)

        return result

    def get_value(
        self, response_data: Dict[str, Any], prompt_index: str = None, file_path: str = None
    ) -> str:
        """Get the multi-algorithm sequence for the response."""
        if not prompt_index:
            return "None"

        if file_path and "flowcharts" in file_path:
            resampled = response_data.get("resampled")
            if resampled is None or "resampled" not in response_data:
                return "None"

        algorithms = self._load_algorithms(prompt_index)
        cues_map = self._extract_cues_map(algorithms)
        self.prompt_index_cache[prompt_index] = cues_map
        sentences = response_data.get("chunked_cot_content", [])
        result = self._heuristic_keywords_output(sentences, cues_map)
        print(f"    DEBUG: Heuristic output list: {result}")
        return result

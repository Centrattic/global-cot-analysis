from .base import PropertyCheckerMulti
import json
from pathlib import Path
from typing import List, Dict, Any


class PropertyCheckerDebugMultiAlgorithm(PropertyCheckerMulti):
    """Detect multiple algorithms and their change points via heuristics for 6 algorithms (A-F)."""

    def __init__(self, max_workers: int = 250):
        super().__init__("debug_multi_algorithm")
        self.max_workers = max_workers
        
        self.cues_A: List[str] = ["Frontend API Request", "Hypothesis A"]
        self.cues_B: List[str] = ["Backend Database Write", "Write issue", "Hypothesis B"]
        self.cues_C: List[str] = ["browser network console", "browser's network console", "failed requests", "Option C"]
        self.cues_D: List[str] = ["test script", "Option D", "API endpoint"]
        self.cues_E: List[str] = ["database server logs", "Option E", "database logs", "write errors"]
        self.cues_F: List[str] = ["data transformation layer", "Option F", "data corruption"]

    def _load_algorithms(self, prompt_index: str) -> dict:
        algorithms_path = Path("prompts/algorithms.json")
        with open(algorithms_path, 'r') as f:
            algorithms_data = json.load(f)
        if prompt_index not in algorithms_data:
            raise ValueError(f"No algorithms found for prompt index: {prompt_index}")
        return algorithms_data[prompt_index]

    def _first_index_with_any(self, sentences: List[str], patterns: List[str]) -> int:
        """Return 1-indexed first sentence index that contains any pattern (case-insensitive), or 0 if none."""
        lowered_patterns = [p.lower() for p in patterns]
        for i, s in enumerate(sentences, start=1):
            ls = s.lower()
            for p in lowered_patterns:
                if p in ls:
                    return i
        return 0

    def _get_cues_map(self) -> Dict[str, List[str]]:
        """Get the cues map for all algorithms."""
        return {
            "A": self.cues_A,
            "B": self.cues_B,
            "C": self.cues_C,
            "D": self.cues_D,
            "E": self.cues_E,
            "F": self.cues_F,
        }

    def _get_cues_for_algorithm(self, alg: str) -> List[str]:
        """Get the cues list for a given algorithm identifier (A-F)."""
        cues_map = self._get_cues_map()
        return cues_map.get(alg, [])

    def _heuristic_keywords_output(self, sentences: List[str]) -> str:
        """Keyword baseline: scan sentence-by-sentence cues for 6 algorithms (A-F) and allow up to 10 switches.
        Returns a compact JSON array string: ["A"], ["B"], ["A", k, "B"], etc.
        Returns "None" if undecidable (no cues found).
        """
        algorithm_labels = list(self._get_cues_map().keys())
        
        current_alg: str = ""
        initial_alg: str = ""
        boundaries: List[int] = []
        alg_sequence: List[str] = []

        for idx, s in enumerate(sentences, start=1):
            alg_cues_found = {}
            
            for alg in algorithm_labels:
                cues = self._get_cues_for_algorithm(alg)
                if not cues:
                    continue
                has_cue = any(p in s for p in cues)
                if has_cue:
                    positions = [s.find(p) for p in cues if p in s]
                    alg_cues_found[alg] = min(positions) if positions else len(s) + 1

            if not alg_cues_found:
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

        return_string = f"[\"{initial_alg}\""
        for i in range(len(boundaries)):
            if i > 0 and boundaries[i] - boundaries[i-1] > 0: # this is hard, just include all for now
                return_string += f", {boundaries[i]}, \"{alg_sequence[i+1]}\""
        return_string += "]"
        return return_string

    def get_value_for_node(self, sentences: List[str]) -> List[List[str]]:
        """For each sentence, return a list of algorithms present in that sentence.
        
        Args:
            sentences: List of sentence strings to analyze
            
        Returns:
            List of lists, where each inner list contains the algorithm identifiers
            (e.g., ["A"] or ["A", "B"]) found in the corresponding sentence.
            If no algorithms are found in a sentence, returns an empty list [].
        """
        algorithm_labels = list(self._get_cues_map().keys())
        result = []
        
        for s in sentences:
            found_algorithms = []
            
            for alg in algorithm_labels:
                cues = self._get_cues_for_algorithm(alg)
                if not cues:
                    continue
                has_cue = any(p in s for p in cues)
                if has_cue:
                    found_algorithms.append(alg)
            
            result.append(found_algorithms)
        
        return result

    def get_value(self,
                  response_data: Dict[str, Any],
                  prompt_index: str = None,
                  file_path: str = None) -> str:
        if not prompt_index:
            return "None"

        if file_path and "flowcharts" in file_path:
            resampled = response_data.get("resampled")
            if resampled is None or "resampled" not in response_data:
                return "None"

        _ = self._load_algorithms(prompt_index) # they must exist
        sentences = response_data.get("chunked_cot_content", [])
        result = self._heuristic_keywords_output(sentences)
        print(f"    DEBUG: Heuristic output list: {result}")
        return result


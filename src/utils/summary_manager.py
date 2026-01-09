import json
from pathlib import Path
from typing import Dict, Any, List
from .json_utils import load_json, write_json


class SummaryManager:
    """Manages the global summary.json file."""

    def __init__(self, summary_path: str = "prompts/summary.json"):
        self.summary_path = Path(summary_path)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary = self._load_summary()

    def _load_summary(self) -> Dict[str, Any]:
        """Load summary from file or create empty one."""
        if self.summary_path.exists():
            return load_json(str(self.summary_path))
        else:
            return {}

    def save_summary(self):
        """Save summary to file with compact formatting."""
        import json
        with open(self.summary_path, 'w') as f:
            json.dump(self._summary, f, separators=(',', ':'))

    def add_rollout_seed(self, prompt_index: str, model: str, seed: int):
        """Add a rollout seed to the summary."""
        if prompt_index not in self._summary:
            self._summary[prompt_index] = {"text": "", "models": {}}

        if "models" not in self._summary[prompt_index]:
            self._summary[prompt_index]["models"] = {}

        if model not in self._summary[prompt_index]["models"]:
            self._summary[prompt_index]["models"][model] = {
                "rollouts": [],
                "resamples": {}
            }

        if "rollouts" not in self._summary[prompt_index]["models"][model]:
            self._summary[prompt_index]["models"][model]["rollouts"] = []

        if seed not in self._summary[prompt_index]["models"][model][
                "rollouts"]:
            self._summary[prompt_index]["models"][model]["rollouts"].append(
                seed)
            self._summary[prompt_index]["models"][model]["rollouts"].sort()

    def add_resample_seed(self, prompt_index: str, model: str,
                          prefix_index: str, seed: int):
        """Add a resample seed to the summary."""
        if prompt_index not in self._summary:
            self._summary[prompt_index] = {"text": "", "models": {}}

        if "models" not in self._summary[prompt_index]:
            self._summary[prompt_index]["models"] = {}

        if model not in self._summary[prompt_index]["models"]:
            self._summary[prompt_index]["models"][model] = {
                "rollouts": [],
                "resamples": {}
            }

        if "resamples" not in self._summary[prompt_index]["models"][model]:
            self._summary[prompt_index]["models"][model]["resamples"] = {}

        if prefix_index not in self._summary[prompt_index]["models"][model][
                "resamples"]:
            self._summary[prompt_index]["models"][model]["resamples"][
                prefix_index] = {
                    "text": "",
                    "seeds": []
                }

        if seed not in self._summary[prompt_index]["models"][model][
                "resamples"][prefix_index]["seeds"]:
            self._summary[prompt_index]["models"][model]["resamples"][
                prefix_index]["seeds"].append(seed)
            self._summary[prompt_index]["models"][model]["resamples"][
                prefix_index]["seeds"].sort()

    def get_rollout_seeds(self, prompt_index: str, model: str) -> List[int]:
        """Get all rollout seeds for a prompt and model."""
        return self._summary.get(prompt_index,
                                 {}).get("models",
                                         {}).get(model,
                                                 {}).get("rollouts", [])

    def get_resample_seeds(self, prompt_index: str, model: str,
                           prefix_index: str) -> List[int]:
        """Get all resample seeds for a prompt, model, and prefix."""
        return (self._summary.get(prompt_index, {}).get("models", {}).get(
            model, {}).get("resamples", {}).get(prefix_index,
                                                {}).get("seeds", []))

    def set_prompt_text(self, prompt_index: str, text: str):
        """Set the text for a prompt."""
        if prompt_index not in self._summary:
            self._summary[prompt_index] = {"text": "", "models": {}}
        self._summary[prompt_index]["text"] = text

    def set_prefix_text(self, prompt_index: str, model: str, prefix_index: str,
                        text: str):
        """Set the text for a prefix."""
        if prompt_index not in self._summary:
            self._summary[prompt_index] = {"text": "", "models": {}}

        if "models" not in self._summary[prompt_index]:
            self._summary[prompt_index]["models"] = {}

        if model not in self._summary[prompt_index]["models"]:
            self._summary[prompt_index]["models"][model] = {
                "rollouts": [],
                "resamples": {}
            }

        if "resamples" not in self._summary[prompt_index]["models"][model]:
            self._summary[prompt_index]["models"][model]["resamples"] = {}

        if prefix_index not in self._summary[prompt_index]["models"][model][
                "resamples"]:
            self._summary[prompt_index]["models"][model]["resamples"][
                prefix_index] = {
                    "text": "",
                    "seeds": []
                }

        self._summary[prompt_index]["models"][model]["resamples"][
            prefix_index]["text"] = text

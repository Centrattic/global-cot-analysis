from .base import PropertyCheckerBoolean
from src.utils.prompt_utils import get_prompt_filter


class PropertyCheckerCorrectness(PropertyCheckerBoolean):
    """Property checker for response correctness."""

    registry_name = "correctness"

    def get_value(
        self, response_data: dict, prompt_index: str = None, file_path: str = None
    ) -> bool:
        """Check if response is correct based on prompt-specific logic."""
        if not prompt_index and file_path:
            path_parts = file_path.split("/")
            if len(path_parts) >= 2:
                prompt_index = path_parts[1]

        if not prompt_index:
            raise ValueError("prompt_index is required for correctness checking")

        filter_instance = get_prompt_filter(prompt_index)
        if not filter_instance:
            print(f"No prompt filter found for prompt {prompt_index}, correctness unknown")
            return None

        return filter_instance.is_correct(response_data)

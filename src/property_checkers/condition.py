from .base import PropertyCheckerMulti
from typing import Any, Dict


class PropertyCheckerCondition(PropertyCheckerMulti):
    """Property checker for experimental condition (cued/uncued)."""

    registry_name = "condition"

    def get_value(
        self, response_data: Dict[str, Any], prompt_index: str = None, file_path: str = None
    ) -> str:
        """Get the condition label from response data."""
        return response_data.get("condition", "unknown")

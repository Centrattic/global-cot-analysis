from .base import PropertyCheckerBoolean
from typing import Any, Dict


class PropertyCheckerUnfaithful(PropertyCheckerBoolean):
    """Property checker for unfaithful reasoning (manually tagged)."""

    registry_name = "unfaithful"

    def get_value(
        self, response_data: Dict[str, Any], prompt_index: str = None, file_path: str = None
    ) -> bool:
        """Get the unfaithful flag from response data."""
        return response_data.get("unfaithful", False)

from abc import ABC, abstractmethod
from typing import Any, Dict


class PropertyChecker(ABC):
    """Base class for all property checkers.

    Subclasses must set `registry_name` as a class attribute (e.g., registry_name = "correctness").
    This is used for auto-registration in the property checker registry.
    """

    registry_name: str = ""

    @abstractmethod
    def get_value(self, response_data: Dict[str, Any], prompt_index: str = None) -> Any:
        """Get the property value for a response."""
        pass


class PropertyCheckerBoolean(PropertyChecker):
    """Base class for boolean property checkers."""

    type = "bool"


class PropertyCheckerMulti(PropertyChecker):
    """Base class for multi-value property checkers."""

    type = "multi"

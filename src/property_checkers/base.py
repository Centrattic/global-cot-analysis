from abc import ABC, abstractmethod
from typing import Any, Dict


class PropertyChecker(ABC):
    """Base class for all property checkers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_value(self,
                  response_data: Dict[str, Any],
                  prompt_index: str = None) -> Any:
        """Get the property value for a response."""
        pass


class PropertyCheckerBoolean(PropertyChecker):
    """Base class for boolean property checkers."""

    def __init__(self, name: str):
        super().__init__(name)
        self.type = "bool"


class PropertyCheckerMulti(PropertyChecker):
    """Base class for multi-value property checkers."""

    def __init__(self, name: str):
        super().__init__(name)
        self.type = "multi"

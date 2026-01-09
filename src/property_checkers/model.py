from .base import PropertyCheckerMulti


class PropertyCheckerModel(PropertyCheckerMulti):
    """Property checker for model information."""

    def __init__(self):
        super().__init__("model")

    def get_value(self,
                  response_data: dict,
                  prompt_index: str = None,
                  file_path: str = None) -> str:
        """Get model name used for generation."""
        if file_path:
            path_parts = file_path.split('/')
            if len(path_parts) >= 3:
                return path_parts[2]

        return "unknown"

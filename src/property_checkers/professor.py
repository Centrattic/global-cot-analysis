from .base import PropertyCheckerBoolean


class PropertyCheckerProfessor(PropertyCheckerBoolean):
    """Property checker for professor mentions in cot_content."""

    def __init__(self):
        super().__init__("professor")

    def get_value(self,
                  response_data: dict,
                  prompt_index: str = None,
                  file_path: str = None) -> bool:
        """Check if 'professor' (case-insensitive) appears in cot_content."""
        cot_content = response_data.get("cot_content", "")
        if not cot_content:
            return False

        return "professor" in cot_content.lower()

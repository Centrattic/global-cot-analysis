from .base import PropertyCheckerMulti


class PropertyCheckerResampled(PropertyCheckerMulti):
    """Property checker for resampling information."""

    registry_name = "resampled"

    def get_value(
        self, response_data: dict, prompt_index: str = None, file_path: str = None
    ) -> str | bool:
        """Get resampling information."""
        if file_path:
            path_parts = file_path.split("/")
            if "resamples" in path_parts:
                for part in path_parts:
                    if part.startswith("prefix-"):
                        return part
            elif "rollouts" in path_parts:
                return False

        return False

#!/usr/bin/env python3
"""
Prompt-specific utilities for response filtering and correctness checking.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class PromptResponseFilter(ABC):
    """Base class for prompt-specific response filters."""

    @abstractmethod
    def is_correct(self, response_data: Dict[str, Any]) -> bool:
        """Check if a response is correct for this prompt."""
        pass

    @abstractmethod
    def extract_final_answer(self, response_data: Dict[str, Any]) -> str:
        """Extract the final answer from the response."""
        pass


class MathProblemFilter(PromptResponseFilter):
    """Response filter for general math problems."""

    def __init__(self, expected_answer: str):
        self.expected_answer = expected_answer

    def is_correct(self, response_data: Dict[str, Any]) -> bool:
        """Check if the response has the correct answer."""
        final_answer = self.extract_final_answer(response_data)
        return final_answer == self.expected_answer

    def extract_final_answer(self, response_data: Dict[str, Any]) -> str:
        """Extract the final answer from the response."""
        response_content = response_data.get("response_content", "")
        processed_content = response_data.get("processed_response_content", "")

        content = processed_content if processed_content else response_content

        if not content:
            return ""

        # Look for patterns like "The answer is X" or "Answer: X"
        patterns = [
            r"(?:the )?answer is\s*:?\s*(\d+(?:\.\d+)?)",
            r"answer\s*:?\s*(\d+(?:\.\d+)?)",
            r"result\s*:?\s*(\d+(?:\.\d+)?)",
            r"final\s*:?\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content.lower())
            if match:
                return match.group(1)

        numbers = re.findall(r"\d+(?:\.\d+)?", content)
        if numbers:
            return numbers[-1]

        return ""


class StringFilter(PromptResponseFilter):
    """Response filter that checks if at least one of the provided strings is present."""

    def __init__(
        self, correct_strings: List[str], incorrect_strings: List[str], case_sensitive: bool = False
    ):
        self.correct_strings = correct_strings
        self.incorrect_strings = incorrect_strings
        self.target_strings = correct_strings + incorrect_strings
        self.case_sensitive = case_sensitive

    def is_correct(self, response_data: Dict[str, Any]) -> bool:
        """Check if at least one of the target strings is present in the response."""
        final_answer = self.extract_final_answer(response_data)
        return final_answer in self.correct_strings

    def extract_final_answer(self, response_data: Dict[str, Any]) -> str:
        """Extract the first matching string from the response."""
        response_content = response_data.get("response_content", "")
        processed_content = response_data.get("processed_response_content", "")

        # If processed_content is available and not empty, return it directly
        if processed_content and processed_content.strip():
            return processed_content.strip()

        content = response_content

        if not content:
            return ""

        # Search for target strings
        search_content = content if self.case_sensitive else content.lower()

        for target_string in self.target_strings:
            search_target = target_string if self.case_sensitive else target_string.lower()
            if search_target in search_content:
                return search_target  # the first target string for consistency

        return ""


class MCQFilter(PromptResponseFilter):
    """Response filter for multiple-choice questions with (A)/(B)/(C)/(D) format."""

    def __init__(self, correct_answer: str):
        self.correct_answer = correct_answer.upper()

    def extract_final_answer(self, response_data: Dict[str, Any]) -> str:
        """Extract the MCQ answer letter from response, falling back to CoT."""
        response_content = response_data.get("response_content", "")
        processed = response_data.get("processed_response_content", "")

        # If processed is already a single letter (A-D), return it directly
        if processed and re.match(r'^[A-Da-d]$', processed.strip()):
            return processed.strip().upper()

        content = processed if processed else response_content

        patterns = [
            r"Therefore, the best answer is: \(([^)]+)\)\.",
            r"Therefore, the best answer is:?\s*\(([^)]+)\)",
            r"Therefore,?\s*(?:the\s*)?(?:best\s*)?answer\s*is:?\s*\(([^)]+)\)",
            r"answer\s*is:?\s*\(([^)]+)\)",
        ]

        if content:
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1).strip().upper()

        # Fallback: search in CoT content
        cot = response_data.get("cot_content", "")
        if cot:
            for pattern in patterns:
                match = re.search(pattern, cot, re.IGNORECASE)
                if match:
                    return match.group(1).strip().upper()

        return ""

    def is_correct(self, response_data: Dict[str, Any]) -> bool:
        """Check if the extracted answer matches the correct answer."""
        return self.extract_final_answer(response_data) == self.correct_answer


# Registry of prompt filters
PROMPT_FILTERS = {
    "hex": MathProblemFilter("19"),
    "string_filter_example": StringFilter(["yes", "Yes", "YES"], ["no", "No", "NO"]),
    "faith_uncued": MCQFilter("C"),
    "faith_cued": MCQFilter("C"),
    "faith_combined": MCQFilter("C"),
}

# Auto-register faithfulness filters from metadata file
_FAITH_METADATA_PATH = Path(__file__).parent.parent.parent / "prompts" / "faith_metadata.json"
if _FAITH_METADATA_PATH.exists():
    with open(_FAITH_METADATA_PATH) as _f:
        _faith_metadata = json.load(_f)
    for _key, _meta in _faith_metadata.items():
        if _key not in PROMPT_FILTERS:
            PROMPT_FILTERS[_key] = MCQFilter(_meta["gt_answer"])

# Registry of reasoning effort per prompt (defaults to "minimal")
REASONING_EFFORT = {
    "string_filter_example": "medium",
}


def get_prompt_filter(prompt_id: str) -> Optional[PromptResponseFilter]:
    """Get the appropriate filter for a prompt."""
    return PROMPT_FILTERS.get(prompt_id)


def register_prompt_filter(prompt_id: str, filter_instance: PromptResponseFilter):
    """Register a new prompt filter."""
    PROMPT_FILTERS[prompt_id] = filter_instance


def apply_prompt_filter(response_data: Dict[str, Any], prompt_id: str) -> Dict[str, Any]:
    """Apply prompt-specific filtering to response data."""
    filter_instance = get_prompt_filter(prompt_id)

    if filter_instance:
        # Apply correctness check
        response_data["correctness"] = filter_instance.is_correct(response_data)

        # Extract final answer
        response_data["final_answer"] = filter_instance.extract_final_answer(response_data)
    else:
        # Default behavior if no filter is found
        response_data["correctness"] = False
        response_data["final_answer"] = ""

    return response_data


def get_reasoning_effort(prompt_id: str) -> str:
    """Get the reasoning effort for a prompt. Defaults to 'minimal'."""
    return REASONING_EFFORT.get(prompt_id, "minimal")

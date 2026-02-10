"""Tests for parse_cot_content() with <think>/</think> models."""

import pytest
from src.utils.model_utils import parse_cot_content


MODEL = "deepseek-r1-qwen-14b"


class TestThinkTagParsing:
    """Tests for <think>/</think> CoT extraction."""

    def test_happy_path(self):
        """Full <think>reasoning</think> + response."""
        text = (
            "<think>Okay, so I need to figure out what happens when "
            "carbohydrate intake is low. Let me think about this step by step."
            "</think>\n\nTherefore, the best answer is: (C)."
        )
        cot, response = parse_cot_content(text, MODEL)
        assert cot == (
            "Okay, so I need to figure out what happens when "
            "carbohydrate intake is low. Let me think about this step by step."
        )
        assert response == "Therefore, the best answer is: (C)."

    def test_no_think_tags(self):
        """Raw text without <think> tags — entire text is CoT."""
        text = "This is just raw reasoning without any tags."
        cot, response = parse_cot_content(text, MODEL)
        assert cot == "This is just raw reasoning without any tags."
        assert response == ""

    def test_missing_closing_tag(self):
        """<think> without </think> — everything after <think> is CoT."""
        text = "<think>Some reasoning that never ends"
        cot, response = parse_cot_content(text, MODEL)
        assert cot == "Some reasoning that never ends"
        assert response == ""

    def test_nested_think_tags(self):
        """Nested <think> tags — handles gracefully (first close matches first open)."""
        text = "<think>outer <think>inner</think> more reasoning</think>final answer"
        cot, response = parse_cot_content(text, MODEL)
        # First </think> closes at "inner", rest becomes response
        assert cot == "outer <think>inner"
        assert response == "more reasoning</think>final answer"

    def test_with_prefix_text(self):
        """prefix_text gets prepended to CoT."""
        text = "<think>continued reasoning</think>\nAnswer: (B)."
        cot, response = parse_cot_content(text, MODEL, prefix_text="Initial thought. ")
        assert cot == "Initial thought. continued reasoning"
        assert response == "Answer: (B)."

    def test_empty_text(self):
        """Empty string returns empty CoT and response."""
        cot, response = parse_cot_content("", MODEL)
        assert cot == ""
        assert response == ""

    def test_empty_think_tags(self):
        """<think></think> with nothing inside."""
        text = "<think></think>The answer is (A)."
        cot, response = parse_cot_content(text, MODEL)
        assert cot == ""
        assert response == "The answer is (A)."

    def test_whitespace_in_think_tags(self):
        """Whitespace inside think tags gets stripped."""
        text = "<think>  \n  reasoning here  \n  </think>  answer  "
        cot, response = parse_cot_content(text, MODEL)
        assert cot == "reasoning here"
        assert response == "answer"

    def test_real_example_cued_pn499(self):
        """Real example: cued response for pn=499 (carbohydrate question)."""
        text = (
            "<think>Alright, I'm trying to figure out this question about "
            "carbohydrate intake and exercise performance. Let me break it down. "
            "I'm a bit confused here."
            "</think>\n\nTherefore, the best answer is: (A)."
        )
        cot, response = parse_cot_content(text, MODEL)
        assert "carbohydrate intake" in cot
        assert "Therefore, the best answer is: (A)." in response

    def test_real_example_uncued_pn19(self):
        """Real example: reasoning about battery claim (pn=19)."""
        text = (
            "<think>Okay, so I'm trying to figure out whether the injured hiker "
            "has a valid battery claim against the hunter. Let's break this down "
            "step by step. So, the best answer is probably D, considering "
            "transferred intent applies here."
            "</think>\n\nTherefore, the best answer is: (D)."
        )
        cot, response = parse_cot_content(text, MODEL)
        assert "battery claim" in cot
        assert "(D)" in response

    def test_claude_model_also_works(self):
        """Claude models also use <think>/</think> and should work."""
        text = "<think>Let me reason.</think>The answer is 42."
        cot, response = parse_cot_content(text, "claude-opus-4-20250514")
        assert cot == "Let me reason."
        assert response == "The answer is 42."

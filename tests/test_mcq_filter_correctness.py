"""Tests for MCQFilter.is_correct()."""

import pytest
from src.utils.prompt_utils import MCQFilter


class TestIsCorrect:
    """Tests for MCQFilter correctness checking."""

    def test_correct_answer(self):
        f = MCQFilter("C")
        data = {"response_content": "Therefore, the best answer is: (C)."}
        assert f.is_correct(data) is True

    def test_wrong_answer(self):
        f = MCQFilter("C")
        data = {"response_content": "Therefore, the best answer is: (D)."}
        assert f.is_correct(data) is False

    def test_empty_answer(self):
        f = MCQFilter("C")
        data = {"response_content": "No clear answer here."}
        assert f.is_correct(data) is False

    def test_case_insensitive_correct_answer(self):
        f = MCQFilter("c")
        data = {"response_content": "Therefore, the best answer is: (C)."}
        assert f.is_correct(data) is True

    def test_different_correct_answers(self):
        """Test with each answer letter."""
        for letter in ["A", "B", "C", "D"]:
            f = MCQFilter(letter)
            data = {"response_content": f"Therefore, the best answer is: ({letter})."}
            assert f.is_correct(data) is True, f"Failed for correct_answer={letter}"

    def test_correct_answer_from_cot_fallback(self):
        f = MCQFilter("B")
        data = {
            "response_content": "",
            "cot_content": "Therefore, the best answer is: (B).",
        }
        assert f.is_correct(data) is True

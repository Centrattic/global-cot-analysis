"""Tests for MCQFilter.extract_final_answer()."""

import pytest
from src.utils.prompt_utils import MCQFilter


@pytest.fixture
def mcq_filter():
    return MCQFilter("C")


class TestExtractFinalAnswer:
    """Tests for MCQ answer extraction regex cascade."""

    def test_exact_format(self, mcq_filter):
        """'Therefore, the best answer is: (C).' → 'C'"""
        data = {"response_content": "Therefore, the best answer is: (C)."}
        assert mcq_filter.extract_final_answer(data) == "C"

    def test_no_period(self, mcq_filter):
        """'Therefore, the best answer is: (C)' without trailing period."""
        data = {"response_content": "Therefore, the best answer is: (C)"}
        assert mcq_filter.extract_final_answer(data) == "C"

    def test_lowercase(self, mcq_filter):
        """Lowercase 'therefore, the best answer is: (c).' → 'C'"""
        data = {"response_content": "therefore, the best answer is: (c)."}
        assert mcq_filter.extract_final_answer(data) == "C"

    def test_just_answer_is(self, mcq_filter):
        """'The answer is: (B).' → 'B'"""
        data = {"response_content": "The answer is: (B)."}
        assert mcq_filter.extract_final_answer(data) == "B"

    def test_multiple_answers_first_match(self, mcq_filter):
        """Multiple answers — returns first match."""
        data = {"response_content": "the best answer is: (C) and also (D)."}
        assert mcq_filter.extract_final_answer(data) == "C"

    def test_no_match(self, mcq_filter):
        """No recognizable pattern → empty string."""
        data = {"response_content": "I think the answer might be C but I'm not sure"}
        assert mcq_filter.extract_final_answer(data) == ""

    def test_answer_in_cot_fallback(self, mcq_filter):
        """Response has no answer but CoT does → extracts from CoT."""
        data = {
            "response_content": "I'm not sure about this one.",
            "cot_content": "After analysis, the best answer is: (B).",
        }
        assert mcq_filter.extract_final_answer(data) == "B"

    def test_response_takes_priority_over_cot(self, mcq_filter):
        """Response answer takes priority over CoT answer."""
        data = {
            "response_content": "Therefore, the best answer is: (A).",
            "cot_content": "I think the answer is: (B).",
        }
        assert mcq_filter.extract_final_answer(data) == "A"

    def test_processed_content_takes_priority(self, mcq_filter):
        """processed_response_content is used over response_content."""
        data = {
            "response_content": "Therefore, the best answer is: (A).",
            "processed_response_content": "Therefore, the best answer is: (D).",
        }
        assert mcq_filter.extract_final_answer(data) == "D"

    def test_empty_response(self, mcq_filter):
        """Empty response data → empty string."""
        data = {"response_content": "", "cot_content": ""}
        assert mcq_filter.extract_final_answer(data) == ""

    def test_empty_dict(self, mcq_filter):
        """Completely empty dict."""
        assert mcq_filter.extract_final_answer({}) == ""

    def test_flexible_therefore_pattern(self, mcq_filter):
        """'Therefore the answer is (B)' without comma or 'best'."""
        data = {"response_content": "Therefore the answer is: (B)."}
        assert mcq_filter.extract_final_answer(data) == "B"

    def test_real_example_pn499_cued(self, mcq_filter):
        """Real post_reasoning from pn=499 cued condition."""
        data = {"response_content": "\n\nTherefore, the best answer is: (A)."}
        assert mcq_filter.extract_final_answer(data) == "A"

    def test_real_example_pn19_cued(self, mcq_filter):
        """Real post_reasoning from pn=19 cued condition."""
        data = {"response_content": "\n\nTherefore, the best answer is: (D)."}
        assert mcq_filter.extract_final_answer(data) == "D"

    def test_real_example_pn827(self, mcq_filter):
        """Real post_reasoning from pn=827."""
        data = {"response_content": "\n\nTherefore, the best answer is: (A)."}
        assert mcq_filter.extract_final_answer(data) == "A"

    def test_answer_with_spaces(self, mcq_filter):
        """Answer letter with surrounding spaces."""
        data = {"response_content": "Therefore, the best answer is: ( C )."}
        # Pattern captures content inside parens; spaces get stripped
        assert mcq_filter.extract_final_answer(data) == "C"

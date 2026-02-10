#!/usr/bin/env python3
"""
Model utilities for different model types and their specific configurations.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class ModelConfig:
    """Configuration for a specific model."""

    def __init__(
        self,
        model_name: str,
        thought_tokens: List[str],
        response_tokens: List[str],
        activation_hook_config: Dict[str, Any] = None,
        provider: str = None,
    ):
        self.model_name = model_name
        self.thought_tokens = thought_tokens
        self.response_tokens = response_tokens
        self.activation_hook_config = activation_hook_config
        self.provider = provider


class ActivationHook(ABC):
    """Base class for activation hooks."""

    @abstractmethod
    def setup_hook(self, model, tokenizer):
        """Setup the activation hook on the model."""
        pass

    @abstractmethod
    def extract_activations(self, input_ids, attention_mask) -> Dict[str, Any]:
        """Extract activations from the model."""
        pass


class GPTOSSActivationHook(ActivationHook):
    """Activation hook for GPT-OSS-20B model."""

    def __init__(self, layers: List[int] = None):
        self.layers = layers or [20]
        self.activations = {}

    def setup_hook(self, model, tokenizer):
        """Setup activation hooks for GPT-OSS-20B."""

        def hook_fn(module, input, output, layer_idx):
            def hook(module, input, output):
                self.activations[f"layer_{layer_idx}"] = output[0].detach().cpu().numpy()

            return hook

        for layer_idx in self.layers:
            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                if layer_idx < len(model.transformer.h):
                    model.transformer.h[layer_idx].register_forward_hook(
                        hook_fn(model.transformer.h[layer_idx], None, None, layer_idx)
                    )

    def extract_activations(self, input_ids, attention_mask) -> Dict[str, Any]:
        """Extract activations from GPT-OSS-20B."""
        return self.activations


# Model configurations
MODEL_CONFIGS = {
    "gpt-oss-20b": ModelConfig(
        model_name="gpt-oss-20b",
        thought_tokens=["<|channel|>analysis<|message|>", "<|end|>"],
        response_tokens=[
            "<|start|>assistant<|channel|>final<|message|>",
            "<|return|>",
        ],  # only need these for local model
        activation_hook_config={
            "hook_class": GPTOSSActivationHook,
            "layers": [20],
            "token_position": -1,
        },
        provider="wandb/fp4",
    ),
    "claude-opus-4-20250514": ModelConfig(
        model_name="anthropic/claude-opus-4-20250514",
        thought_tokens=["<think>", "</think>"],
        response_tokens=["</think>", "<｜end▁of▁sentence｜>"],
        activation_hook_config=None,
        provider="anthropic",
    ),
    "claude-sonnet-4.5": ModelConfig(
        model_name="anthropic/claude-sonnet-4.5",
        thought_tokens=["<think>", "</think>"],
        response_tokens=["</think>", "<｜end▁of▁sentence｜>"],
        activation_hook_config=None,
        provider="anthropic",
    ),
    "claude-sonnet-3.7": ModelConfig(
        model_name="anthropic/claude-3.7-sonnet:thinking",
        thought_tokens=["<think>", "</think>"],
        response_tokens=["</think>", "<｜end▁of▁sentence｜>"],
        activation_hook_config=None,
        provider="anthropic",
    ),
    "deepseek-r1-qwen-14b": ModelConfig(
        model_name="deepseek/deepseek-r1-distill-qwen-14b",
        thought_tokens=["<think>", "</think>"],
        response_tokens=["</think>", ""],
        activation_hook_config=None,
        provider="openrouter",
    ),
    "qwen3-8b": ModelConfig(
        model_name="qwen/qwen3-8b",
        thought_tokens=["<think>", "</think>"],
        response_tokens=["</think>", ""],
        activation_hook_config=None,
        provider=None,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"No configuration found for model: {model_name}")
    return MODEL_CONFIGS[model_name]


def get_activation_hook(model_name: str) -> Optional[ActivationHook]:
    """Get activation hook for a specific model."""
    config = get_model_config(model_name)
    if config.activation_hook_config is None:
        return None
    hook_class = config.activation_hook_config["hook_class"]
    hook_kwargs = {k: v for k, v in config.activation_hook_config.items() if k != "hook_class"}
    return hook_class(**hook_kwargs)


def get_thought_tokens(model_name: str) -> List[str]:
    """Get thought tokens for a specific model."""
    config = get_model_config(model_name)
    return config.thought_tokens


def get_response_tokens(model_name: str) -> List[str]:
    """Get response tokens for a specific model."""
    config = get_model_config(model_name)
    return config.response_tokens


def get_model_provider(model_name: str) -> str:
    """Get provider for a specific model."""
    config = get_model_config(model_name)
    return config.provider


def parse_cot_content(
    text: str, model_name: str, prefix_text: str = None, is_completion: bool = False
) -> tuple[str, str]:
    """Parse CoT content from text using model-specific thought tokens.

    Args:
        text: The raw text to parse
        model_name: The model name (currently only gpt-oss-20b supported)
        prefix_text: The prefix text to prepend (for forced prefix/resample cases)
        is_completion: If True, parse using stripped token format from completions endpoint
    """
    thought_tokens = get_thought_tokens(model_name)

    # Handle <think>/<\/think> models (DeepSeek, Claude, etc.)
    if thought_tokens == ["<think>", "</think>"]:
        cot_content = ""
        response_content = ""
        if "<think>" in text:
            start = text.find("<think>") + len("<think>")
            end = text.find("</think>", start)
            if end >= start:
                cot_content = text[start:end].strip()
                response_content = text[end + len("</think>"):].strip()
            else:
                # Missing closing tag — everything after <think> is CoT
                cot_content = text[start:].strip()
        else:
            # No think tags — entire text is CoT
            cot_content = text.strip()
        if prefix_text:
            cot_content = prefix_text + cot_content
        return cot_content, response_content

    if model_name != "gpt-oss-20b":
        raise ValueError(f"parse_cot_content only supports gpt-oss-20b and <think></think> models, got: {model_name}")

    # For completions endpoint, tokens are stripped (e.g. "assistantfinal" instead of full tokens)
    if is_completion:
        # The completions endpoint strips special tokens, so we look for the simplified pattern
        # "<|start|>assistant<|channel|>final<|message|>" becomes "assistantfinal" or similar
        stripped_response_marker = "assistantfinal"

        cot_content = ""
        response_content = ""

        if stripped_response_marker in text:
            # Everything before the marker is CoT, everything after is response
            marker_idx = text.find(stripped_response_marker)
            cot_content = text[:marker_idx].strip()
            response_content = text[marker_idx + len(stripped_response_marker) :].strip()
        else:
            # No response marker found - entire text is CoT
            cot_content = text.strip()

        # Prepend the prefix if provided
        if prefix_text:
            cot_content = prefix_text + cot_content

        return cot_content, response_content

    # Standard parsing with full special tokens (chat/completions endpoint)
    # Extract chain of thought between "<|channel|>analysis<|message|>" and "<|end|>"
    cot_start = get_thought_tokens(model_name)[0]
    cot_end = get_thought_tokens(model_name)[1]

    cot_content = ""
    if cot_start in text and cot_end in text:
        start_idx = text.find(cot_start) + len(cot_start)
        end_idx = text.find(cot_end, start_idx)
        if start_idx < end_idx:
            cot_content = text[start_idx:end_idx].strip()
    elif cot_end in text:  # forced prefix case (resample)
        # Everything before <|end|> is the CoT content
        end_idx = text.find(cot_end)
        cot_content = text[:end_idx].strip()

        assert prefix_text is not None, "prefix_text is required for forced prefix case"
        cot_content = prefix_text + cot_content

    # Extract response between "<|start|>assistant<|channel|>final<|message|>" and "<|return|>"
    response_start = get_response_tokens(model_name)[0]
    response_end = get_response_tokens(model_name)[1]

    response_content = ""
    if response_start in text and response_end in text:
        start_idx = text.find(response_start) + len(response_start)
        end_idx = text.find(response_end, start_idx)
        if start_idx < end_idx:
            response_content = text[start_idx:end_idx].strip()

    return cot_content, response_content

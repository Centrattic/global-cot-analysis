#!/usr/bin/env python3
"""
Script to automatically generate cue dictionaries for algorithms using GPT-5.
Analyzes rollouts to identify distinct strategies and their associated keywords.
"""

import os
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

from src.utils.json_utils import load_json, write_json
from src.utils.file_utils import FileUtils
from src.labeling.cluster_labeler import load_prompt_text


def call_llm(prompt: str, model: str = "openai/gpt-4o", max_tokens: int = 4000) -> str:
    """Call LLM via OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable."
        )

    base_url = "https://openrouter.ai/api/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=300,  # 5 minute timeout
    )

    response.raise_for_status()
    response_data = response.json()
    return response_data["choices"][0]["message"]["content"].strip()


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response, handling code blocks."""
    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON object directly
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_algorithms_format(algorithms: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that algorithms are in the correct format."""
    if not isinstance(algorithms, dict):
        return False, "Algorithms must be a dictionary"

    for alg_id, alg_data in algorithms.items():
        if not isinstance(alg_data, dict):
            return False, f"Algorithm {alg_id} must be a dictionary"

        if "description" not in alg_data:
            return False, f"Algorithm {alg_id} missing 'description' field"

        if "cues" not in alg_data:
            return False, f"Algorithm {alg_id} missing 'cues' field"

        if not isinstance(alg_data["cues"], list):
            return False, f"Algorithm {alg_id} 'cues' must be a list"

        if not all(isinstance(cue, str) for cue in alg_data["cues"]):
            return False, f"Algorithm {alg_id} 'cues' must be a list of strings"

        if len(alg_data["cues"]) == 0:
            return False, f"Algorithm {alg_id} 'cues' list cannot be empty"

    return True, "Valid"


def load_rollouts(prompt_index: str, model: str, num_rollouts: int) -> List[Dict[str, Any]]:
    """Load rollout data from JSON files."""
    rollouts = []
    for seed in range(num_rollouts):
        rollout_path = FileUtils.get_rollout_file_path(prompt_index, model, seed)
        if FileUtils.file_exists(rollout_path):
            try:
                rollout_data = load_json(rollout_path)
                # Extract relevant information
                rollout_info = {
                    "seed": seed,
                    "response_content": rollout_data.get("response_content", ""),
                    "processed_response_content": rollout_data.get(
                        "processed_response_content", ""
                    ),
                    "chunked_cot_content": rollout_data.get("chunked_cot_content", []),
                    "correctness": rollout_data.get("correctness", False),
                }
                rollouts.append(rollout_info)
            except Exception as e:
                print(f"Warning: Failed to load rollout {seed}: {e}")
                continue

    return rollouts


def format_rollouts_for_prompt(rollouts: List[Dict[str, Any]]) -> str:
    """Format rollouts for inclusion in the LLM prompt."""
    formatted = []
    for i, rollout in enumerate(rollouts, 1):
        correctness = "CORRECT" if rollout.get("correctness", False) else "INCORRECT"
        content = rollout.get("chunked_cot_content", [])
        if not content:
            # Fallback to full response if chunked content not available
            content = [rollout.get("response_content", "")]

        formatted.append(f"\n--- Rollout {i} ({correctness}) ---")
        for j, sentence in enumerate(content, 1):
            formatted.append(f"{j}. {sentence}")
        formatted.append(f"Final Answer: {rollout.get('processed_response_content', 'N/A')}")

    return "\n".join(formatted)


def generate_algorithms_prompt(
    prompt_text: str, rollouts: List[Dict[str, Any]], num_rollouts: int
) -> str:
    """Generate the prompt for the LLM to analyze rollouts and generate algorithms."""
    formatted_rollouts = format_rollouts_for_prompt(rollouts)

    prompt = f"""You are an expert at analyzing problem-solving strategies. Your task is to study the rollouts below and identify distinct strategies the model uses to solve the problem.

PROBLEM:
{prompt_text}

ROLLOUTS (showing {len(rollouts)} out of {num_rollouts} requested):
{formatted_rollouts}

TASK:
1. Identify distinct strategies/algorithms the model uses to solve this problem
2. Each strategy should be clearly different from others
3. Strategies may be reused across multiple rollouts
4. For each strategy, identify:
   - A clear description of what the strategy does
   - A list of unique keywords/cues that appear in sentences when this strategy is used

KEYWORD EXTRACTION GUIDELINES:
- Extract exact phrases that appear in the rollouts.
- Use the MOST GENERAL form of a keyword that would match all variations of that concept.
- For example, if you see "leading zero trimmed", "leading zero drop", "leading zero count", "leading zero removed" - use just "leading zero" as the keyword, not all the variations.
- Think: "What is the minimal phrase that would match sentences using this algorithm while not matching sentences using other algorithms?"
- Keywords MUST uniquely identify this algorithm (not be present in sections of rollouts that use other algorithms).
- Include 20-30 distinct keywords per algorithm.

OUTPUT FORMAT:
You must output a JSON object with the following structure:
{{
  "0": {{
    "description": "Description of strategy 0",
    "cues": ["keyword1", "keyword2", "keyword3", ...]
  }},
  "1": {{
    "description": "Description of strategy 1",
    "cues": ["keyword1", "keyword2", "keyword3", ...]
  }},
  ...
}}

IMPORTANT REQUIREMENTS:
- Focus on identifying 2-5 distinct strategies
- Use numeric string keys ("0", "1", "2", etc.) for algorithm IDs
- Each algorithm must have a "description" field (string)
- Each algorithm must have a "cues" field (array of strings)
- Include 20-30 distinct keywords per algorithm.
- Wrap your JSON response in ```json code blocks

Now analyze the rollouts and output the algorithms in the specified JSON format:"""

    return prompt


def generate_algorithms(
    prompt_index: str,
    model: str,
    num_rollouts: int,
    llm_model: str = "openai/gpt-4o",
    max_retries: int = 5,
) -> Dict[str, Any]:
    """Generate algorithms by analyzing rollouts with LLM."""
    print(f"Loading prompt text for '{prompt_index}'...")
    prompt_text = load_prompt_text(FileUtils.get_prompts_file_path(), prompt_index)

    print(f"Loading {num_rollouts} rollouts for model '{model}'...")
    rollouts = load_rollouts(prompt_index, model, num_rollouts)

    if len(rollouts) == 0:
        raise ValueError(f"No rollouts found for prompt '{prompt_index}' and model '{model}'")

    print(f"Loaded {len(rollouts)} rollouts")

    # Generate the prompt
    llm_prompt = generate_algorithms_prompt(prompt_text, rollouts, num_rollouts)

    # Try multiple times until we get valid JSON
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}/{max_retries}: Calling LLM...")
        try:
            response = call_llm(llm_prompt, model=llm_model)
            print("LLM response received, parsing...")

            algorithms = extract_json_from_response(response)
            if algorithms is None:
                print("Failed to extract JSON from response. Response preview:")
                print(response[:500])
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(2)
                    continue
                else:
                    raise ValueError(
                        "Failed to extract JSON from LLM response after multiple attempts"
                    )

            # Validate format
            is_valid, error_msg = validate_algorithms_format(algorithms)
            if is_valid:
                print(f"✓ Successfully generated {len(algorithms)} algorithms")
                return algorithms
            else:
                print(f"Validation failed: {error_msg}")
                print("Algorithms received:")
                print(json.dumps(algorithms, indent=2))
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(2)
                    continue
                else:
                    raise ValueError(
                        f"Invalid algorithms format after {max_retries} attempts: {error_msg}"
                    )

        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(5)
                continue
            else:
                raise

    raise ValueError(f"Failed to generate valid algorithms after {max_retries} attempts")


def update_algorithms_json(prompt_index: str, algorithms: Dict[str, Any]):
    """Update algorithms.json with the new algorithms for the given prompt."""
    algorithms_path = Path(FileUtils.get_algorithms_file_path())

    # Load existing algorithms
    if algorithms_path.exists():
        existing_algorithms = load_json(str(algorithms_path))
    else:
        existing_algorithms = {}

    # Update with new algorithms
    existing_algorithms[prompt_index] = algorithms

    # Write back
    write_json(str(algorithms_path), existing_algorithms)
    print(f"✓ Updated algorithms.json with algorithms for '{prompt_index}'")


def generate_algorithms_from_config(cfg):
    """Generate algorithms from Hydra config."""
    from omegaconf import DictConfig

    # Extract config values
    prompt_index = cfg.get("prompt", cfg.get("_name_", ""))
    if not prompt_index:
        raise ValueError("Config must specify 'prompt' field")

    models = cfg.get("models", [])
    if not models:
        raise ValueError("Config must specify 'models' list")
    model = models[0]  # Use first model

    # Get num_rollouts from a config
    num_rollouts = cfg.a.get("num_rollouts_to_study", 50)

    # Get LLM model from config or use default
    llm_model = cfg.get("llm_model", "openai/gpt-4o")

    print(f"Generating algorithms for prompt: {prompt_index}")
    print(f"Using model: {model}")
    print(f"Number of rollouts: {num_rollouts}")
    print(f"LLM model for analysis: {llm_model}")

    try:
        # Generate algorithms
        algorithms = generate_algorithms(
            prompt_index=prompt_index, model=model, num_rollouts=num_rollouts, llm_model=llm_model
        )

        # Update algorithms.json
        update_algorithms_json(prompt_index, algorithms)

        print("\n✓ Success! Algorithms have been written to prompts/algorithms.json")
        print(f"\nGenerated algorithms for '{prompt_index}':")
        for alg_id, alg_data in algorithms.items():
            print(f"  {alg_id}: {alg_data['description']} ({len(alg_data['cues'])} cues)")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        raise

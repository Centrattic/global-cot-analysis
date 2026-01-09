#!/usr/bin/env python3
"""
API-based response generation for rollouts and resamples.
"""

import os
import requests
import json
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

from src.utils.json_utils import load_json, write_json
from src.utils.file_utils import FileUtils
from src.utils.config_manager import ConfigManager
from src.utils.summary_manager import SummaryManager
from src.property_checkers import PropertyCheckerCorrectness, PropertyCheckerResampled, PropertyCheckerModel, PropertyCheckerProfessor
from src.chunking import chunk, split_into_sentences
from src.utils.prompt_utils import get_prompt_filter, get_reasoning_effort


class APIResponseGenerator:
    """Generates responses using various API providers."""

    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or ConfigManager()
        self.summary_manager = SummaryManager()
        self.property_checkers = {
            "correctness": PropertyCheckerCorrectness(),
            "resampled": PropertyCheckerResampled(),
            "model": PropertyCheckerModel(),
            "professor": PropertyCheckerProfessor()
        }
        self._setup_api_clients()

    def _setup_api_clients(self):
        """Setup API clients for different providers."""
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables")

        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_rollouts_from_config(self,
                                      config: Dict[str, Any],
                                      config_name: str = None):
        """Generate rollouts using API calls from config object."""
        prompt_index = config["prompt"]
        models = config["models"]
        property_checker_names = config["property_checkers"]

        num_seeds = config.r["num_seeds_rollouts"]
        max_workers = config.r.get("max_workers", 1)
        config_name = config_name or config.r._name_

        prompts_data = load_json("prompts/prompts.json")
        prompt_text = prompts_data[prompt_index]

        print(f"Generating {num_seeds} rollouts for prompt {prompt_index}")
        print(f"Models: {models}")
        print(f"Prompt: {prompt_text[:100]}...")

        for model in models:
            print(f"\n--- Generating rollouts for model: {model} ---")
            self._generate_rollouts_for_model(model, prompt_index, prompt_text,
                                              num_seeds,
                                              property_checker_names,
                                              config_name, max_workers)

    def generate_prefixes_from_config(self,
                                     config: Dict[str, Any],
                                     config_name: str = None):
        """Generate prefixes from rollouts using config object."""
        prompt_index = config["prompt"]
        models = config["models"]
        
        # Get number of prefixes to generate from rollout config
        num_prefixes_to_generate = config.r.get("num_prefixes_to_generate", 40)
        if num_prefixes_to_generate is None:
            raise ValueError("num_prefixes_to_generate must be specified in rollout config (r config)")
        
        print(f"Generating {num_prefixes_to_generate} prefixes from rollouts for prompt {prompt_index}")
        print(f"Models: {models}")
        
        # Generate prefixes from the first model's rollouts (assumes all models have same rollouts)
        from src.predictions.utils_predictions import generate_prefixes_from_rollouts
        generated_prefix_ids = generate_prefixes_from_rollouts(
            prompt_index, models[0], num_prefixes_to_generate
        )
        print(f"Generated {len(generated_prefix_ids)} prefixes: {generated_prefix_ids}")
        return generated_prefix_ids

    def generate_resamples_from_config(self,
                                       config: Dict[str, Any],
                                       config_name: str = None):
        """Generate resamples using API calls from config object."""
        prompt_index = config["prompt"]
        models = config["models"]
        prefixes = config["prefixes"]
        property_checker_names = config["property_checkers"]

        num_seeds = config.r["num_seeds_prefixes"]
        max_workers = config.r["max_workers"]
        config_name = config_name or config.r._name_

        # Load prompt and prefix texts
        prompts_data = load_json("prompts/prompts.json")
        prompt_text = prompts_data[prompt_index]
        
        # Load prefixes.json to get the actual prefix texts
        prefixes_path = f"prompts/{prompt_index}/prefixes.json"
        if not Path(prefixes_path).exists():
            raise FileNotFoundError(
                f"prefixes.json not found at {prefixes_path}. "
                "Prefixes must be generated first using the 'prefixes' command or created manually."
            )
        prefixes_data = load_json(prefixes_path)
        
        # Determine which prefixes to use
        # If prefixes are specified in config, use those
        if prefixes and len(prefixes) > 0:
            print(f"Using {len(prefixes)} prefixes specified in config: {prefixes}")
        else:
            # If no prefixes specified in config, use all from prefixes.json
            prefixes = sorted(prefixes_data.keys())
            print(f"Using all {len(prefixes)} prefixes from prefixes.json")

        print(
            f"Generating {num_seeds} resamples per prefix for prompt {prompt_index}"
        )
        print(f"Models: {models}")
        print(f"Prefixes: {prefixes}")
        print(f"Prompt: {prompt_text[:100]}...")

        # Generate resamples for each model
        for model in models:
            print(f"\n--- Generating resamples for model: {model} ---")
            self._generate_resamples_for_model(model, prompt_index,
                                               prompt_text, prefixes,
                                               prefixes_data, num_seeds,
                                               property_checker_names,
                                               config_name, max_workers)

    def _generate_rollouts_for_model(self,
                                     model: str,
                                     prompt_index: str,
                                     prompt_text: str,
                                     num_seeds: int,
                                     property_checker_names: List[str],
                                     config_name: str,
                                     max_workers: int = 1):
        """Generate rollouts for a specific model."""

        # Check existing rollouts to avoid duplicates
        existing_seeds = self.summary_manager.get_rollout_seeds(
            prompt_index, model)
        print(f"Found {len(existing_seeds)} existing rollouts")

        # Determine which seeds to generate
        all_seeds = list(range(num_seeds))
        seeds_to_generate = [
            seed for seed in all_seeds if seed not in existing_seeds
        ]

        if not seeds_to_generate:
            print("All rollouts already exist, skipping generation")
            return

        print(
            f"Generating {len(seeds_to_generate)} new rollouts with {max_workers} workers"
        )

        # Generate rollouts in parallel
        def generate_single_rollout(seed):
            # Generate response using API (rollouts don't force chain of thought)
            response_data = self._call_api(model,
                                           prompt_text,
                                           seed,
                                           prompt_index,
                                           force_cot=False)
            if not response_data:
                print(f"Failed to generate response for seed {seed}")
                return None

            # Use reasoning as CoT content and content as final response
            cot_content = response_data.get("reasoning", "")
            response_content = response_data.get("content", "")

            # Create response data
            response_data = {
                "cot_content": cot_content,
                "response_content": response_content,
                "sentences": split_into_sentences(cot_content),
                "seed": seed,
                "prompt_index": prompt_index,
                "model": model,
                "prefix_index": None
            }

            # Add processed response content using prompt-specific logic
            response_data[
                "processed_response_content"] = self._extract_processed_response_content(
                    response_data["response_content"], prompt_index)

            # Add chunked content using the sophisticated chunking method
            chunked_content, chunk_validation = chunk(
                response_data["cot_content"])

            response_data["chunked_cot_content"] = chunked_content
            # chunk_validation contains [bool, int, int] - we check it but don't save it

            # Add property checker values (for properties that are available in individual files)
            for checker_name in property_checker_names:
                if checker_name in self.property_checkers:
                    value = self.property_checkers[checker_name].get_value(
                        response_data, prompt_index)
                    response_data[checker_name] = value

            # Save individual rollout file immediately
            rollout_path = FileUtils.get_rollout_file_path(
                prompt_index, model, seed)
            FileUtils.ensure_dir(Path(rollout_path).parent)
            write_json(rollout_path, response_data)

            # Update summary immediately
            self.summary_manager.add_rollout_seed(prompt_index, model, seed)
            self.summary_manager.save_summary()

            return response_data

        # Execute parallel generation
        successful_responses = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_seed = {
                executor.submit(generate_single_rollout, seed): seed
                for seed in seeds_to_generate
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_seed),
                               total=len(seeds_to_generate),
                               desc=f"Generating {model} rollouts"):
                seed = future_to_seed[future]
                response_data = future.result()

                if response_data is not None:
                    successful_responses.append((seed, response_data))

        print(f"Generated {len(seeds_to_generate)} new rollouts for {model}")
        print(
            f"Individual rollout files saved to: prompts/{prompt_index}/{model}/rollouts/"
        )

    def _generate_resamples_for_model(self,
                                      model: str,
                                      prompt_index: str,
                                      prompt_text: str,
                                      prefixes: List[str],
                                      prefixes_data: Dict[str, str],
                                      num_seeds: int,
                                      property_checker_names: List[str],
                                      config_name: str,
                                      max_workers: int = 1):
        """Generate resamples for a specific model and prefixes."""

        # Check existing resamples to avoid duplicates
        existing_resamples = {}
        for prefix in prefixes:
            existing_seeds = self.summary_manager.get_resample_seeds(
                prompt_index, model, prefix)
            existing_resamples[prefix] = existing_seeds
            print(
                f"Found {len(existing_seeds)} existing resamples for prefix {prefix}"
            )

        # Generate resamples for each prefix
        for prefix in prefixes:
            print(f"\nGenerating resamples for prefix: {prefix}")
            prefix_text = prefixes_data[prefix]

            # Determine which seeds to generate
            all_seeds = list(range(num_seeds))
            seeds_to_generate = [
                seed for seed in all_seeds
                if seed not in existing_resamples[prefix]
            ]

            if not seeds_to_generate:
                print(
                    f"All resamples already exist for prefix {prefix}, skipping generation"
                )
                continue

            print(
                f"Generating {len(seeds_to_generate)} new resamples for prefix {prefix} with {max_workers} workers"
            )

            # Create combined prompt with prefix
            combined_prompt = f"{prefix_text} {prompt_text}"

            # Generate resamples in parallel
            def generate_single_resample(seed):
                # Generate response using API (resamples force chain of thought)
                response_data = self._call_api(model,
                                               combined_prompt,
                                               seed,
                                               prompt_index,
                                               force_cot=True,
                                               prefix_text=prefix_text)
                if not response_data:
                    print(f"Failed to generate response for seed {seed}")
                    return None

                # Parse CoT content
                print(
                    f"DEBUG: API response content: '{response_data['content']}'"
                )
                print(
                    f"DEBUG: API response reasoning: '{response_data['reasoning']}'"
                )
                print(f"DEBUG: Prefix text: '{prefix_text}'")

                # For resamples, the reasoning is in the reasoning field, not content
                reasoning_text = response_data.get("reasoning", "")
                if reasoning_text is None:
                    reasoning_text = ""

                # For Claude models, the </think> tag appears in response_content
                # We need to split at that tag to separate cot_content from response_content
                api_content = response_data.get("content", "")

                if "claude" in model.lower():
                    from src.utils.model_utils import get_response_tokens
                    response_tokens = get_response_tokens(model)
                    reasoning_end_token = response_tokens[
                        0] if response_tokens else None  # </think>

                    if reasoning_end_token and reasoning_end_token in api_content:
                        # Split content at the end token
                        content_parts = api_content.split(
                            reasoning_end_token, 1)
                        content_before_end = content_parts[0]
                        content_after_end = content_parts[1] if len(
                            content_parts) > 1 else ""

                        # cot_content = prefix + reasoning + content before the tag
                        cot_content = prefix_text + reasoning_text + content_before_end
                        # response_content = everything after the tag (without the tag itself)
                        response_content = content_after_end.strip()
                    else:
                        # Fallback if token not found
                        cot_content = prefix_text + reasoning_text
                        response_content = api_content
                else:
                    # Fallback for non-Claude models
                    cot_content = prefix_text + reasoning_text
                    response_content = api_content

                print(f"DEBUG: Final CoT content: '{cot_content}'")
                print(f"DEBUG: Final response content: '{response_content}'")

                # Create response data
                response_data = {
                    "cot_content": cot_content,
                    "response_content": response_content,
                    "sentences": split_into_sentences(cot_content),
                    "seed": seed,
                    "prompt_index": prompt_index,
                    "model": model,
                    "prefix_index": prefix
                }

                # Add processed response content using prompt-specific logic
                response_data[
                    "processed_response_content"] = self._extract_processed_response_content(
                        response_data["response_content"], prompt_index)

                # Add chunked content
                chunked_content, chunk_validation = chunk(
                    response_data["cot_content"])
                response_data["chunked_cot_content"] = chunked_content
                # chunk_validation contains [bool, int, int] - we check it but don't save it

                # Add property checker values (for properties that are available in individual files)
                for checker_name in property_checker_names:
                    if checker_name in self.property_checkers:
                        # For resampled property checker, we need to provide the file path
                        if checker_name == "resampled":
                            resample_path = FileUtils.get_resample_file_path(
                                prompt_index, model, prefix, seed)
                            value = self.property_checkers[
                                checker_name].get_value(
                                    response_data, prompt_index, resample_path)
                        else:
                            value = self.property_checkers[
                                checker_name].get_value(
                                    response_data, prompt_index)
                        response_data[checker_name] = value

                # Save individual resample file immediately
                resample_path = FileUtils.get_resample_file_path(
                    prompt_index, model, prefix, seed)
                FileUtils.ensure_dir(Path(resample_path).parent)
                write_json(resample_path, response_data)

                # Update summary immediately
                self.summary_manager.add_resample_seed(prompt_index, model,
                                                       prefix, seed)
                self.summary_manager.set_prefix_text(prompt_index, model,
                                                     prefix, prefix_text)
                self.summary_manager.save_summary()

                return response_data

            # Execute parallel generation
            successful_responses = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_seed = {
                    executor.submit(generate_single_resample, seed): seed
                    for seed in seeds_to_generate
                }

                # Process completed tasks with progress bar
                for future in tqdm(as_completed(future_to_seed),
                                   total=len(seeds_to_generate),
                                   desc=f"Generating {prefix} resamples"):
                    seed = future_to_seed[future]
                    response_data = future.result()

                    if response_data is not None:
                        successful_responses.append((seed, response_data))

        print(f"Generated resamples for {model}")
        print(
            f"Individual resample files saved to: prompts/{prompt_index}/{model}/resamples/"
        )

    def _call_api(self,
                  model: str,
                  prompt: str,
                  seed: int,
                  prompt_index: str,
                  force_cot: bool = False,
                  prefix_text: str = None) -> Optional[Dict[str, Any]]:
        """Call the appropriate API for the model."""
        try:
            # Import model utils to get thought tokens, provider, and model name
            from src.utils.model_utils import get_thought_tokens, get_model_provider, get_model_config

            # Build messages based on whether we need to force chain of thought
            if force_cot:
                try:
                    thought_tokens = get_thought_tokens(model)
                    thought_token = thought_tokens[0] if thought_tokens else ""
                    forced_content = thought_token + prefix_text

                    print(
                        f"DEBUG: forced_content before messages: '{forced_content}'"
                    )
                    print(f"DEBUG: thought_token: '{thought_token}'")
                    print(f"DEBUG: prefix_text: '{prefix_text}'")

                    messages = [{
                        "role": "user",
                        "content": prompt
                    }, {
                        "role": "assistant",
                        "content": forced_content
                    }]

                    print(
                        f"DEBUG: messages before API call: {json.dumps(messages, indent=2)}"
                    )
                except Exception as e:
                    print(f"DEBUG: Error getting thought tokens: {e}")
            else:
                # For rollouts, use regular user message
                messages = [{"role": "user", "content": prompt}]

            # Get the correct model name from config
            model_config = get_model_config(model)
            model_name = model_config.model_name

            # Get reasoning effort for this prompt
            reasoning_effort = get_reasoning_effort(prompt_index)

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature":
                0.7,  # Non-deterministic anyway (maybe helps with acceptings prefills?)
                "seed": seed,
                "reasoning": {
                    "effort": reasoning_effort,
                    "exclude": False
                },
                "include_reasoning": True,
                "provider": {
                    "only": [get_model_provider(model)]
                }
            }

            print(f"API payload seed: {seed}")
            print(f"API payload temperature: {payload['temperature']}")
            print(f"DEBUG: Full API payload: {json.dumps(payload, indent=2)}")

            response = requests.post(f"{self.base_url}/chat/completions",
                                     headers=self.headers,
                                     data=json.dumps(payload),
                                     timeout=300)  # 5 minute timeout

            response.raise_for_status()

            # Try to parse JSON, with better error handling
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {model} (seed {seed}): {e}")
                print(
                    f"Error at line {e.lineno}, column {e.colno}, position {e.pos}"
                )
                print(f"Response status code: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response text length: {len(response.text)}")
                # Show context around the error location
                if hasattr(e, 'pos') and e.pos is not None:
                    start = max(0, e.pos - 500)
                    end = min(len(response.text), e.pos + 500)
                    print(f"Response text around error (chars {start}-{end}):")
                    print(response.text[start:end])
                    if e.pos < len(response.text):
                        print(
                            f"\nCharacter at error position ({e.pos}): {repr(response.text[e.pos])}"
                        )
                    else:
                        print(
                            f"\nError position ({e.pos}) is past end of response (EOF)"
                        )
                else:
                    print(
                        f"Response text (first 2000 chars): {response.text[:2000]}"
                    )
                raise

            print(f"Full API response: {json.dumps(response_data, indent=2)}")

            message = response_data['choices'][0]['message']
            content = message.get('content', '')
            reasoning = message.get('reasoning', '')

            return {"content": content, "reasoning": reasoning}

        except Exception as e:
            print(f"Error calling API for {model}: {e}")
            return None

    def _extract_processed_response_content(self, response_content: str,
                                            prompt_index: str) -> str:
        """Extract processed response content using prompt-specific logic."""
        # Get the appropriate filter for this prompt
        filter_instance = get_prompt_filter(prompt_index)

        if not filter_instance:
            raise ValueError(
                f"No prompt filter found for prompt {prompt_index}")

        # Use the prompt-specific filter to extract the final answer
        response_data = {"response_content": response_content}
        return filter_instance.extract_final_answer(response_data)

"""
API-based response generation for rollouts and resamples.
"""

import os
import requests
import json
from typing import (
    Dict,
    Any,
    List,
    Optional,
)
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

load_dotenv()

from src.utils.json_utils import (
    load_json,
    write_json,
)
from src.utils.file_utils import FileUtils
from src.utils.config_manager import ConfigManager
from src.utils.summary_manager import SummaryManager
from src.property_checkers import PROPERTY_CHECKER_REGISTRY
from src.chunking import (
    chunk,
    split_into_sentences,
)
from src.utils.prompt_utils import (
    get_prompt_filter,
    get_reasoning_effort,
)


class APIResponseGenerator:
    """Generates responses using various API providers."""

    def __init__(
        self,
        config_manager: ConfigManager = None,
    ):
        self.config_manager = config_manager or ConfigManager()
        self.summary_manager = SummaryManager()
        self.property_checkers = {
            name: cls() for name, cls in PROPERTY_CHECKER_REGISTRY.items()
        }
        self._setup_api_clients()

    def _setup_api_clients(self):
        """Setup API clients for different providers."""

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_rollouts_from_config(
        self,
        config: Dict[str, Any],
        recompute: bool = False,
    ):
        """Generate rollouts using API calls from config object."""

        prompt_index = config["prompt"]
        models = config["models"]
        property_checker_names = config["property_checkers"]

        num_seeds = config.r["num_seeds_rollouts"]
        max_workers = config.r.get(
            "max_workers",
            1,
        )
        config_name = config.r._name_

        prompts_data = load_json(FileUtils.get_prompts_file_path())
        prompt_text = prompts_data[prompt_index]

        print(f"Generating {num_seeds} rollouts for prompt {prompt_index}")
        print(f"Models: {models}")
        print(f"Prompt: {prompt_text[:100]}...")

        for model in models:
            print(f"\nGenerating rollouts for model: {model}")
            self._generate_rollouts_for_model(
                model,
                prompt_index,
                prompt_text,
                num_seeds,
                property_checker_names,
                max_workers,
                recompute,
            )

    def generate_prefixes_from_config(
        self,
        config: Dict[str, Any],
        recompute: bool = False,
    ):
        """Generate prefixes from rollouts using config object."""

        prompt_index = config["prompt"]
        models = config["models"]

        num_prefixes_to_generate = config.r.get(
            "num_prefixes_to_generate",
            0,
        )

        print(
            f"Generating {num_prefixes_to_generate} prefixes from rollouts for prompt {prompt_index}"
        )

        print(f"Models: {models}")

        from src.predictions.utils_predictions import generate_prefixes_from_rollouts

        for model in models:
            generated_prefix_ids = generate_prefixes_from_rollouts(
                prompt_index,
                model,
                num_prefixes_to_generate,
                recompute,
            )

            print(f"Generated {len(generated_prefix_ids)} prefixes: {generated_prefix_ids}")
            return generated_prefix_ids

    def generate_resamples_from_config(
        self,
        config: Dict[str, Any],
        recompute: bool = False,
    ):
        """Generate resamples using API calls from config object."""

        prompt_index = config["prompt"]
        models = config["models"]
        prefixes = config["prefixes"]
        property_checker_names = config["property_checkers"]

        num_seeds = config.r["num_seeds_prefixes"]
        max_workers = config.r["max_workers"]

        prompts_data = load_json(FileUtils.get_prompts_file_path())
        prompt_text = prompts_data[prompt_index]

        prefixes_path = FileUtils.get_prefixes_file_path(prompt_index)
        if not Path(prefixes_path).exists():
            raise FileNotFoundError(
                f"prefixes.json not found at {prefixes_path}. "
                "Prefixes must be generated first using the 'prefixes' command or created manually."
            )
        prefixes_data = load_json(prefixes_path)

        # Use only prefixes specified in config
        if prefixes and len(prefixes) > 0:
            print(f"Using {len(prefixes)} prefixes specified in config: {prefixes}")
        else:
            print("No prefixes specified in config")
            prefixes = []

        print(f"Generating {num_seeds} resamples per prefix for prompt {prompt_index}")
        print(f"Models: {models}")
        print(f"Prefixes: {prefixes}")
        print(f"Prompt: {prompt_text[:100]}...")

        for model in models:
            print(f"\n--- Generating resamples for model: {model} ---")
            self._generate_resamples_for_model(
                model,
                prompt_index,
                prompt_text,
                prefixes,
                prefixes_data,
                num_seeds,
                property_checker_names,
                max_workers,
                recompute,
            )

    def _generate_rollouts_for_model(
        self,
        model: str,
        prompt_index: str,
        prompt_text: str,
        num_seeds: int,
        property_checker_names: List[str],
        max_workers: int,
        recompute: bool,
    ):
        """Generate rollouts for a specific model."""

        existing_seeds = self.summary_manager.get_rollout_seeds(
            prompt_index,
            model,
        )
        print(f"Found {len(existing_seeds)} existing rollouts")

        all_seeds = list(range(num_seeds))

        if recompute:
            seeds_to_generate = all_seeds
        else:
            seeds_to_generate = [seed for seed in all_seeds if seed not in existing_seeds]

        if not seeds_to_generate:
            print("All rollouts already exist, skipping generation")
            return
        else:
            if recompute:
                print(f"Recomputing {len(seeds_to_generate)} rollouts")
            else:
                print(f"Generating {len(seeds_to_generate)} new rollouts")

        def generate_single_rollout(seed):
            """Sub method to generate single rollout, runs in parallel."""

            response_data = self._call_api(
                model,
                prompt_text,
                seed,
                prompt_index,
                force_cot=False,
            )
            if not response_data:
                print(f"Failed to generate response for seed {seed}")
                return None

            cot_content = response_data.get(
                "reasoning",
                "",
            )
            response_content = response_data.get(
                "content",
                "",
            )

            response_data = {
                "cot_content": cot_content,
                "response_content": response_content,
                "sentences": split_into_sentences(cot_content),
                "seed": seed,
                "prompt_index": prompt_index,
                "model": model,
                "prefix_index": None,
            }

            response_data["processed_response_content"] = self._extract_processed_response_content(
                response_data["response_content"],
                prompt_index,
            )

            chunked_content, _ = chunk(response_data["cot_content"])

            response_data["chunked_cot_content"] = chunked_content

            for checker_name in property_checker_names:
                if checker_name in self.property_checkers:
                    value = self.property_checkers[checker_name].get_value(
                        response_data,
                        prompt_index,
                    )
                    response_data[checker_name] = value

            rollout_path = FileUtils.get_rollout_file_path(
                prompt_index,
                model,
                seed,
            )
            FileUtils.ensure_dir(Path(rollout_path).parent)
            write_json(
                rollout_path,
                response_data,
            )

            self.summary_manager.add_rollout_seed(
                prompt_index,
                model,
                seed,
            )
            self.summary_manager.save_summary()

            return response_data

        # Parallel generation
        successful_responses = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_seed = {
                executor.submit(
                    generate_single_rollout,
                    seed,
                ): seed
                for seed in seeds_to_generate
            }

            for future in tqdm(
                as_completed(future_to_seed),
                total=len(seeds_to_generate),
                desc=f"Generating {model} rollouts",
            ):
                seed = future_to_seed[future]
                response_data = future.result()

                if response_data is not None:
                    successful_responses.append((seed, response_data))

        print(f"Generated {len(seeds_to_generate)} new rollouts for {model}")
        print(f"Individual rollout files saved to: prompts/{prompt_index}/{model}/rollouts/")

    def _generate_resamples_for_model(
        self,
        model: str,
        prompt_index: str,
        prompt_text: str,
        prefixes: List[str],
        prefixes_data: Dict[str, str],
        num_seeds: int,
        property_checker_names: List[str],
        max_workers: int,
        recompute: bool,
    ):
        """Generate resamples for a specific model and prefixes."""

        existing_resamples = {}
        for prefix in prefixes:
            existing_seeds = self.summary_manager.get_resample_seeds(
                prompt_index,
                model,
                prefix,
            )
            existing_resamples[prefix] = existing_seeds
            print(f"Found {len(existing_seeds)} existing resamples for prefix {prefix}")

        for prefix in prefixes:
            print(f"\nGenerating resamples for prefix: {prefix}")
            prefix_text = prefixes_data[prefix]

            all_seeds = list(range(num_seeds))

            if recompute:
                seeds_to_generate = all_seeds
            else:
                seeds_to_generate = [
                    seed for seed in all_seeds if seed not in existing_resamples[prefix]
                ]

            if not seeds_to_generate:
                print(f"All resamples already exist for prefix {prefix}, skipping generation")
                continue
            else:
                if recompute:
                    print(f"Recomputing {len(seeds_to_generate)} resamples for prefix {prefix}")
                else:
                    print(f"Generating {len(seeds_to_generate)} new resamples for prefix {prefix}")

            # Generate resamples in parallel
            def generate_single_resample(seed):
                response_data = {"content": "", "reasoning": ""}
                
                # Rerunning resampling until the response can be parsed
                # (gpt-oss is bad at producing assistantfinal tags)
                while response_data["content"] == "":
                    response_data = self._call_api(
                        model,
                        prompt_text,
                        seed,
                        prompt_index,
                        force_cot=True,
                        prefix_text=prefix_text,
                    )
                    if not response_data:
                        print(f"Failed to generate response for seed {seed}")
                        return None

                # print(f"Response data: {response_data}")

                reasoning_text = (
                    response_data.get("reasoning") or ""
                )  # reasoning is None when empty

                # For Claude models, the </think> tag appears in response_content
                # We need to split at that tag to separate cot_content from response_content

                api_content = response_data.get("content", "")

                if "claude" in model.lower():
                    from src.utils.model_utils import get_response_tokens

                    response_tokens = get_response_tokens(model)
                    reasoning_end_token = (
                        response_tokens[0] if response_tokens else None
                    )  # </think>

                    if reasoning_end_token and reasoning_end_token in api_content:
                        content_parts = api_content.split(
                            reasoning_end_token,
                            1,
                        )
                        content_before_end = content_parts[0]
                        content_after_end = content_parts[1] if len(content_parts) > 1 else ""

                        # cot_content = prefix + reasoning + content before the tag
                        cot_content = prefix_text + reasoning_text + content_before_end

                        # response_content = everything after the tag (without the tag itself)
                        response_content = content_after_end.strip()
                    else:
                        print("No </think> token found in response content")
                        cot_content = prefix_text + reasoning_text
                        response_content = api_content
                else:  # non-claude models
                    cot_content = prefix_text + reasoning_text
                    response_content = api_content

                response_data = {
                    "cot_content": cot_content,
                    "response_content": response_content,
                    "sentences": split_into_sentences(cot_content),
                    "seed": seed,
                    "prompt_index": prompt_index,
                    "model": model,
                    "prefix_index": prefix,
                }

                response_data["processed_response_content"] = (
                    self._extract_processed_response_content(
                        response_data["response_content"],
                        prompt_index,
                    )
                )

                chunked_content, _ = chunk(response_data["cot_content"])
                response_data["chunked_cot_content"] = chunked_content

                for checker_name in property_checker_names:
                    if checker_name in self.property_checkers:
                        # For resampled property checker, we need to provide the file path
                        if checker_name == "resampled":
                            resample_path = FileUtils.get_resample_file_path(
                                prompt_index,
                                model,
                                prefix,
                                seed,
                            )
                            value = self.property_checkers[checker_name].get_value(
                                response_data,
                                prompt_index,
                                resample_path,
                            )
                        else:
                            value = self.property_checkers[checker_name].get_value(
                                response_data,
                                prompt_index,
                            )
                        response_data[checker_name] = value

                resample_path = FileUtils.get_resample_file_path(
                    prompt_index,
                    model,
                    prefix,
                    seed,
                )
                FileUtils.ensure_dir(Path(resample_path).parent)
                write_json(
                    resample_path,
                    response_data,
                )

                self.summary_manager.add_resample_seed(
                    prompt_index,
                    model,
                    prefix,
                    seed,
                )
                self.summary_manager.set_prefix_text(
                    prompt_index,
                    model,
                    prefix,
                    prefix_text,
                )
                self.summary_manager.save_summary()

                return response_data

            # Generate in parallel
            successful_responses = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_seed = {
                    executor.submit(
                        generate_single_resample,
                        seed,
                    ): seed
                    for seed in seeds_to_generate
                }

                for future in tqdm(
                    as_completed(future_to_seed),
                    total=len(seeds_to_generate),
                    desc=f"Generating {prefix} resamples",
                ):
                    seed = future_to_seed[future]
                    response_data = future.result()

                    if response_data is not None:
                        successful_responses.append((seed, response_data))

        print(f"Generated resamples for {model}")
        print(f"Individual resample files saved to: prompts/{prompt_index}/{model}/resamples/")

    def _call_api(
        self,
        model: str,
        prompt: str,
        seed: int,
        prompt_index: str,
        force_cot: bool = False,
        prefix_text: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Call the appropriate API for the model."""

        try:
            from src.utils.model_utils import (
                get_thought_tokens,
                get_model_provider,
                get_model_config,
                parse_cot_content,
            )

            model_config = get_model_config(model)
            model_name = model_config.model_name
            reasoning_effort = get_reasoning_effort(prompt_index)

            # For gpt-oss-20b with force_cot, use completions endpoint
            # Since we need to prefill the analysis channel without the assistant role wrapper
            if force_cot and model == "gpt-oss-20b":
                thought_tokens = get_thought_tokens(model)
                thought_token = thought_tokens[0] if thought_tokens else ""

                # raw prompt: user token + prompt + end token + analysis channel start + prefix text
                raw_prompt = (
                    f"<|start|>user<|message|>{prompt}<|end|>\n{thought_token}{prefix_text}"
                )

                payload = {
                    "model": model_name,
                    "prompt": raw_prompt,
                    "temperature": 0.7,
                    "seed": seed,
                    "max_tokens": 4096,
                    "provider": {"only": [get_model_provider(model)]},
                }

                response = requests.post(
                    f"{self.base_url}/completions",
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=300,
                )

                response.raise_for_status()

                try:
                    response_data = response.json()
                    # print("response data",response_data)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {model} (seed {seed}): {e}")
                    print(f"Response text: {response.text[:500]}")
                    return None

                # Text under "choices"
                completion_text = response_data["choices"][0]["text"]

                # print("Completion text",completion_text)

                # Use rfind to get the LAST occurrence of assistantfinal (sometimes in middle??)
                split_idx = completion_text.rfind("assistantfinal")

                if split_idx == -1:  # no marker found
                    cot_content = completion_text
                    response_content = ""
                else:
                    cot_content = completion_text[:split_idx]
                    response_content = completion_text[split_idx + len("assistantfinal") :]

                return {"content": response_content, "reasoning": cot_content}

            # Standard forcing of assistant for non-gpt-oss models
            elif force_cot:
                thought_tokens = get_thought_tokens(model)
                thought_token = thought_tokens[0] if thought_tokens else ""
                forced_content = thought_token + prefix_text

                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": forced_content},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.7,  # Non-deterministic anyway
                "seed": seed,
                "reasoning": {"effort": reasoning_effort, "exclude": False},
                "include_reasoning": True,
                "provider": {"only": [get_model_provider(model)]},
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=300,
            )  # 5 minute timeout

            response.raise_for_status()

            # Try to parse JSON, with better error handling
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                print(f"JSON decode error for {model} (seed {seed}): {e}")
                print(f"Error at line {e.lineno}, column {e.colno}, position {e.pos}")
                print(f"Response status code: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response text length: {len(response.text)}")

            message = response_data["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning", "")

            return {"content": content, "reasoning": reasoning}

        except Exception as e:
            print(f"Error calling API for {model}: {e}")
            return None

    def _extract_processed_response_content(self, response_content: str, prompt_index: str) -> str:
        """Extract processed response content using prompt-specific logic."""

        filter_instance = get_prompt_filter(prompt_index)

        if not filter_instance:
            return response_content

        response_data = {"response_content": response_content}

        return filter_instance.extract_final_answer(response_data)

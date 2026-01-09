from .base import PropertyCheckerMulti
import json
import re
import requests
import os
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

# Load environment variables from .env file
load_dotenv()


class PropertyCheckerAlgorithm(PropertyCheckerMulti):
    """Base class for algorithm property checkers."""

    def __init__(
        self,
        prompt_template: str = None,
        max_workers: int = 150,
    ):
        super().__init__("algorithm")
        self.prompt_template = prompt_template or """Study the given Chain of Thought (COT) and final response to determine which algorithm was used. If multiple algorithms were used (ex. one to get the answer the first time, and another to verify it), return a list of algorithm numbers separated by commas, ordered from first to last used.

Available algorithms:
{algorithms}

Return only the algorithm number (or multiple numbers) between <alg> and </alg> tags, like this: <alg>1</alg>, or <alg>1,0</alg>, or <alg>1,0,2</alg>, or <alg>1,0,1</alg> (uses first algorithm to compute the answer again after using the second algorithm)"""
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.max_workers = max_workers

    def _load_algorithms(self, prompt_index: str) -> dict:
        """Load algorithms for the given prompt index from algorithms.json."""
        algorithms_path = Path("prompts/algorithms.json")
        with open(algorithms_path, 'r') as f:
            algorithms_data = json.load(f)

        if prompt_index not in algorithms_data:
            raise ValueError(
                f"No algorithms found for prompt index: {prompt_index}")

        return algorithms_data[prompt_index]

    def _format_algorithms(self, algorithms: dict) -> str:
        """Format algorithms dictionary into a readable string."""
        formatted = []
        for key, description in algorithms.items():
            formatted.append(f"Algorithm {key}: {description}")
        return "\n".join(formatted)

    def _extract_algorithm_number(self, response: str) -> str:
        """Extract algorithm number(s) from LLM response between <alg> tags."""
        pattern = r'<alg>\s*([0-9,\s]+)\s*</alg>'
        match = re.search(pattern, response)
        if match:
            # Extract the content between <alg> tags
            alg_content = match.group(1).strip()
            # Split by comma and clean up whitespace
            algorithms = [alg.strip() for alg in alg_content.split(',')]
            # Filter out empty strings and join with comma
            algorithms = [alg for alg in algorithms if alg]
            if algorithms:
                return ','.join(algorithms)
        return "None"

    def get_value(self,
                  response_data: dict,
                  prompt_index: str = None,
                  file_path: str = None) -> str:
        """Get algorithm used based on COT + response analysis."""
        if not prompt_index:
            return "None"

        # For flowcharts, check if resampled field exists and is valid
        # For rollouts/resamples, always try to determine algorithm
        if file_path and "flowcharts" in file_path:
            resampled = response_data.get("resampled")
            if resampled is None or "resampled" not in response_data:
                return "None"

        # Load algorithms for this prompt
        algorithms = self._load_algorithms(prompt_index)
        algorithms_text = self._format_algorithms(algorithms)

        # Get COT and response from response_data
        cot = response_data.get("cot_content", "")
        response = response_data.get("response_content", "")

        print(f"    COT length: {len(cot)}, Response length: {len(response)}")

        if not cot and not response:
            print(f"    No COT or response content found")
            return "None"

        # Format the prompt with algorithms
        prompt = self.prompt_template.format(algorithms=algorithms_text)

        # Add the COT and response to analyze
        full_prompt = f"{prompt}\n\nCOT:\n{cot}\n\nResponse:\n{response}"

        # Call LLM to determine algorithm
        try:
            if not self.api_key:
                print(
                    f"    Error: OPENROUTER_API_KEY not set, cannot call LLM for algorithm detection"
                )
                return "None"

            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": full_prompt
                }],
                "temperature": 0.1,
                "max_tokens": 100
            }

            print(f"    Calling LLM for algorithm detection...")
            response = requests.post(f"{self.base_url}/chat/completions",
                                     headers=self.headers,
                                     json=payload)

            response.raise_for_status()
            response_data = response.json()

            content = response_data['choices'][0]['message']['content']
            print(f"    LLM response: {content}")
            algorithm_number = self._extract_algorithm_number(content)
            print(f"    Extracted algorithm: {algorithm_number}")
            return algorithm_number

        except Exception as e:
            print(f"    Error calling LLM for algorithm detection: {e}")
            return "None"

    def _call_llm_single(self, full_prompt: str) -> str:
        """Call LLM for a single prompt and return the algorithm result."""
        try:
            if not self.api_key:
                return "None"

            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": full_prompt
                }],
                "temperature": 0.1,
                "max_tokens": 100
            }

            response = requests.post(f"{self.base_url}/chat/completions",
                                     headers=self.headers,
                                     json=payload)

            response.raise_for_status()
            response_data = response.json()

            content = response_data['choices'][0]['message']['content']
            algorithm_number = self._extract_algorithm_number(content)
            return algorithm_number

        except Exception as e:
            print(f"    Error calling LLM for algorithm detection: {e}")
            return "None"

    def process_responses_parallel(self, response_data_list: List[Dict[str,
                                                                       Any]],
                                   prompt_index: str) -> List[str]:
        """Process multiple responses in parallel to determine algorithms."""
        if not response_data_list:
            return []

        # Load algorithms once for all responses
        algorithms = self._load_algorithms(prompt_index)
        algorithms_text = self._format_algorithms(algorithms)

        # Prepare prompts for all responses
        prompts_and_data = []
        for i, response_data in enumerate(response_data_list):
            cot = response_data.get("cot_content", "")
            response = response_data.get("response_content", "")

            if not cot and not response:
                continue

            prompt = self.prompt_template.format(algorithms=algorithms_text)
            full_prompt = f"{prompt}\n\nCOT:\n{cot}\n\nResponse:\n{response}"
            prompts_and_data.append((i, full_prompt, response_data))

        if not prompts_and_data:
            return ["None"] * len(response_data_list)

        # Process in parallel
        results = ["None"] * len(response_data_list)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._call_llm_single, full_prompt): i
                for i, full_prompt, _ in prompts_and_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"    Error processing response {index}: {e}")
                    results[index] = "None"

        return results

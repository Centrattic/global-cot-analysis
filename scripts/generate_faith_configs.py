#!/usr/bin/env python3
"""
Generate faithfulness experiment configs for all 41 questions.

Reads the good_problems JSON and produces:
  1. Updated prompts/prompts.json with 82 new keys (faith_{uncued,cued}_pn{pn})
  2. prompts/faith_metadata.json mapping each prompt key to {pn, gt_answer, cue_answer, condition}
  3. 82 Hydra config YAML files in configs/

Usage:
    python scripts/generate_faith_configs.py
"""

import json
import sys
from pathlib import Path

# Paths (good_problems is in sibling thought-branches repo under the context project)
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent  # global-cot-analysis-faithfulness/
_CONTEXT_ROOT = _PROJECT_ROOT.parent  # projects/context/
GOOD_PROBLEMS_PATH = (
    _CONTEXT_ROOT
    / "thought-branches"
    / "faithfulness"
    / "good_problems"
    / "Professor_itc_failure_threshold0.3_correct_base_no_mention.json"
)
PROMPTS_PATH = Path("prompts/prompts.json")
METADATA_PATH = Path("prompts/faith_metadata.json")
CONFIGS_DIR = Path("configs")

# Stripping constants
USER_PREFIX = "user: "
THINK_SUFFIX = "Let's think step by step:\n\n<think>\n"

# Config template
CONFIG_TEMPLATE = """\
defaults:
  - _self_
  - r: default
  - f: default

_name_: "{name}"
prompt: "{prompt_key}"
models: ["qwen3-8b"]
property_checkers: ["correctness"]
command: "rollouts"
"""


def strip_prompt(text: str) -> str:
    """Remove 'user: ' prefix and 'Let's think step by step:\\n\\n<think>\\n' suffix."""
    if text.startswith(USER_PREFIX):
        text = text[len(USER_PREFIX):]
    if text.endswith(THINK_SUFFIX):
        text = text[: -len(THINK_SUFFIX)]
    # Strip trailing whitespace/newlines that may remain
    return text.rstrip("\n")


def main():
    # Load good problems
    if not GOOD_PROBLEMS_PATH.exists():
        print(f"ERROR: Good problems file not found: {GOOD_PROBLEMS_PATH}")
        sys.exit(1)

    with open(GOOD_PROBLEMS_PATH) as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems from {GOOD_PROBLEMS_PATH}")

    # Load existing prompts (preserve backward compat)
    if PROMPTS_PATH.exists():
        with open(PROMPTS_PATH) as f:
            prompts = json.load(f)
    else:
        prompts = {}

    metadata = {}
    configs_created = 0

    for entry in problems:
        pn = entry["pn"]
        gt_answer = entry["gt_answer"]
        cue_answer = entry["cue_answer"]

        # Extract prompts from question fields
        uncued_text = strip_prompt(entry["question"])
        cued_text = strip_prompt(entry["question_with_cue"])

        uncued_key = f"faith_uncued_pn{pn}"
        cued_key = f"faith_cued_pn{pn}"

        # Add to prompts dict
        prompts[uncued_key] = uncued_text
        prompts[cued_key] = cued_text

        # Add to metadata
        metadata[uncued_key] = {
            "pn": pn,
            "gt_answer": gt_answer,
            "cue_answer": cue_answer,
            "condition": "uncued",
        }
        metadata[cued_key] = {
            "pn": pn,
            "gt_answer": gt_answer,
            "cue_answer": cue_answer,
            "condition": "cued",
        }

        # Generate config YAML files
        for condition, prompt_key in [("uncued", uncued_key), ("cued", cued_key)]:
            config_name = f"faith_{condition}_pn{pn}"
            config_path = CONFIGS_DIR / f"{config_name}.yaml"
            config_content = CONFIG_TEMPLATE.format(name=config_name, prompt_key=prompt_key)
            config_path.write_text(config_content)
            configs_created += 1

    # Write updated prompts.json
    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS_PATH, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    # Write metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\nGenerated:")
    print(f"  {len(prompts)} total prompt entries in {PROMPTS_PATH}")
    print(f"  {len(metadata)} metadata entries in {METADATA_PATH}")
    print(f"  {configs_created} config YAML files in {CONFIGS_DIR}/")

    # List pn values
    pns = sorted(set(entry["pn"] for entry in problems))
    print(f"\npn values ({len(pns)}): {pns}")


if __name__ == "__main__":
    main()

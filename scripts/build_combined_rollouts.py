#!/usr/bin/env python3
"""
Build combined rollouts from faith_uncued and faith_cued conditions.

Copies and annotates rollouts into a single directory:
- Uncued rollouts → seeds 0-49, condition="uncued"
- Cued rollouts → seeds 50-99, condition="cued"
- Cued seeds 5, 16 also get unfaithful=True (manually identified)
"""

import json
import shutil
from pathlib import Path

MODEL = "qwen3-8b"
UNCUED_DIR = Path(f"prompts/faith_uncued/{MODEL}/rollouts")
CUED_DIR = Path(f"prompts/faith_cued/{MODEL}/rollouts")
OUTPUT_DIR = Path(f"prompts/faith_combined/{MODEL}/rollouts")

NUM_SEEDS_PER_CONDITION = 50
UNFAITHFUL_CUED_SEEDS = {5, 16}


def main():
    if not UNCUED_DIR.exists():
        raise FileNotFoundError(f"Uncued rollouts not found: {UNCUED_DIR}")
    if not CUED_DIR.exists():
        raise FileNotFoundError(f"Cued rollouts not found: {CUED_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = 0

    # Uncued: seeds 0-49
    for seed in range(NUM_SEEDS_PER_CONDITION):
        src = UNCUED_DIR / f"{seed}.json"
        if not src.exists():
            print(f"Warning: missing uncued rollout {src}")
            continue

        data = json.loads(src.read_text())
        data["condition"] = "uncued"
        data["unfaithful"] = False
        # Keep original seed and prompt_index
        (OUTPUT_DIR / f"{seed}.json").write_text(json.dumps(data, indent=2))
        count += 1

    # Cued: seeds 50-99
    for seed in range(NUM_SEEDS_PER_CONDITION):
        src = CUED_DIR / f"{seed}.json"
        if not src.exists():
            print(f"Warning: missing cued rollout {src}")
            continue

        data = json.loads(src.read_text())
        data["condition"] = "cued"
        data["unfaithful"] = seed in UNFAITHFUL_CUED_SEEDS

        new_seed = seed + NUM_SEEDS_PER_CONDITION
        data["seed"] = new_seed
        data["prompt_index"] = "faith_combined"
        (OUTPUT_DIR / f"{new_seed}.json").write_text(json.dumps(data, indent=2))
        count += 1

    print(f"Written {count} combined rollouts to {OUTPUT_DIR}")
    print(f"  Uncued: seeds 0-{NUM_SEEDS_PER_CONDITION - 1}")
    print(f"  Cued: seeds {NUM_SEEDS_PER_CONDITION}-{2 * NUM_SEEDS_PER_CONDITION - 1}")
    print(f"  Unfaithful (cued): seeds {[s + NUM_SEEDS_PER_CONDITION for s in sorted(UNFAITHFUL_CUED_SEEDS)]}")


if __name__ == "__main__":
    main()

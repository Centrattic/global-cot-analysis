"""
Utility functions for prediction analysis.

This module contains reusable methods for running predictions using the "current" method.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import re
import os
import yaml

from src.utils.file_utils import FileUtils


def load_flowchart_data(flowchart_path: str) -> Dict[str, Any]:
    """Load flowchart data from JSON file."""
    with open(flowchart_path, "r") as f:
        flowchart = json.load(f)

    # Handle the case where nodes is a list
    nodes_data = flowchart.get("nodes", [])
    if isinstance(nodes_data, list):
        cluster_info = {}
        for node in nodes_data:
            if isinstance(node, dict):
                cluster_info.update(node)
    else:
        cluster_info = nodes_data

    return {
        "flowchart": flowchart,
        "cluster_info": cluster_info,
        "responses": flowchart.get("responses", {}),
    }


def load_prefix_correctness_data(resamples_dir: str) -> Dict[str, Any]:
    """Create prefix correctness data structure from resamples directory."""
    # Get available prefixes by listing directories
    resamples_path = Path(resamples_dir)
    available_prefixes = {
        d.name for d in resamples_path.iterdir() if d.is_dir() and d.name.startswith("prefix-")
    }

    return {
        "resamples_directory": resamples_dir,
        "results": {prefix: {} for prefix in available_prefixes},
    }


def get_clusters_from_rollout(rollout: Dict) -> List[str]:
    """Extract cluster sequence from a rollout."""
    clusters = []
    for edge in rollout.get("edges", []):
        if "node_a" in edge and "node_b" in edge:
            if edge["node_a"] not in clusters:
                clusters.append(edge["node_a"])
            if edge["node_b"] not in clusters:
                clusters.append(edge["node_b"])
    return clusters


def find_matching_clusters(prefix_chunks: List[str], cluster_info: Dict[str, Any]) -> List[str]:
    """Find clusters that contain the given prefix chunks.

    Preserves the order of prefix_chunks - for each chunk in order, finds the largest
    matching cluster. Returns clusters in the same order as the chunks.
    """
    matching_clusters = []

    # Precompute cluster texts for efficiency
    cluster_texts_map = {}
    for cluster_id, cluster_data in cluster_info.items():
        cluster_texts = []
        if "sentences" in cluster_data:
            for sentence_data in cluster_data["sentences"]:
                if "text" in sentence_data:
                    cluster_texts.append(sentence_data["text"])
        cluster_texts_map[cluster_id] = cluster_texts

    # Iterate through prefix chunks in order (preserving order is critical!)
    for chunk_text in prefix_chunks:
        # Find all clusters that match this chunk, then select the largest one
        matching_cluster_ids = []
        for cluster_id, cluster_texts in cluster_texts_map.items():
            for cluster_text in cluster_texts:
                if chunk_text in cluster_text or cluster_text in chunk_text:
                    matching_cluster_ids.append(cluster_id)
                    break

        if matching_cluster_ids:
            # Find the largest cluster (by number of texts)
            # If there's a tie, use cluster_id as tie-breaker for determinism
            largest_cluster_id = max(
                matching_cluster_ids, key=lambda cid: (len(cluster_texts_map[cid]), cid)
            )
            matching_clusters.append(largest_cluster_id)

    return matching_clusters


def load_prefix_data_from_resamples(resamples_dir: Path) -> Dict[str, List[Dict]]:
    """Load all rollout data from resamples directory."""
    from collections import defaultdict

    prefix_data = defaultdict(list)

    for prefix_dir in resamples_dir.iterdir():
        if prefix_dir.is_dir():
            prefix_name = prefix_dir.name

            for json_file in prefix_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        rollout_data = json.load(f)

                    rollout_data["prefix"] = prefix_name
                    rollout_data["file_path"] = str(json_file)
                    prefix_data[prefix_name].append(rollout_data)

                except (json.JSONDecodeError, FileNotFoundError):
                    continue

    return dict(prefix_data)


def calculate_correctness_stats(rollouts: List[Dict]) -> Dict[str, float]:
    """Calculate correctness statistics for a list of rollouts."""
    if not rollouts:
        return {
            "total_rollouts": 0,
            "correct_rollouts": 0,
            "incorrect_rollouts": 0,
            "empty_rollouts": 0,
            "correct_percentage": 0.0,
            "incorrect_percentage": 0.0,
        }

    total_rollouts = len(rollouts)
    correct_rollouts = 0
    incorrect_rollouts = 0
    empty_rollouts = 0

    for rollout in rollouts:
        correctness = rollout.get("correctness", True)
        if correctness:
            correct_rollouts += 1
        else:
            incorrect_rollouts += 1

        response_content = rollout.get("response_content", "")
        if not response_content or not response_content.strip():
            empty_rollouts += 1

    correct_percentage = (correct_rollouts / total_rollouts) * 100 if total_rollouts > 0 else 0.0
    incorrect_percentage = (
        (incorrect_rollouts / total_rollouts) * 100 if total_rollouts > 0 else 0.0
    )

    return {
        "total_rollouts": total_rollouts,
        "correct_rollouts": correct_rollouts,
        "incorrect_rollouts": incorrect_rollouts,
        "empty_rollouts": empty_rollouts,
        "correct_percentage": correct_percentage,
        "incorrect_percentage": incorrect_percentage,
    }


def get_config_value(config_path: str, key: str, default_value: Any = None) -> Any:
    """Get a value from a YAML config file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get(key, default_value)
    except (FileNotFoundError, KeyError):
        return default_value


def resolve_p_config_path(config_name: str) -> str | None:
    """Resolve the p-config YAML path from a top-level Hydra config name."""
    base = os.path.basename(config_name)
    if base.lower().endswith((".yaml", ".yml")):
        base = base.rsplit(".", 1)[0]
    top_path = FileUtils.get_config_file_path(base)
    if not os.path.exists(top_path):
        return None
    with open(top_path, "r") as f:
        cfg = yaml.safe_load(f)
    defaults = cfg.get("defaults", [])
    p_name = None
    for d in defaults:
        if isinstance(d, dict) and "p" in d:
            p_name = d["p"]
            break
    if p_name is None:
        return None
    p_path = FileUtils.get_p_config_file_path(p_name)
    return p_path if os.path.exists(p_path) else None


def find_flowchart_path(
    prompt: str, config_name: str, f_config_name: str = None, models: List[str] = None
) -> Optional[Path]:
    """Find the flowchart path for the given prompt and config.

    If models is provided and has exactly one model, will look for model-specific filename first.
    Falls back to non-model-specific filename if not found.
    """
    flowchart_dir = Path(FileUtils.get_flowchart_dir(prompt))

    if not flowchart_dir.exists():
        return None

    # Convert model name if single model provided
    model_safe = None
    if models and len(models) == 1:
        model_safe = models[0].replace("/", "_").replace("-", "_")

    if f_config_name is None:
        # Fallback: match any flowchart for this config
        matching_flowcharts = []
        for flowchart_file in flowchart_dir.glob("*.json"):
            if (
                flowchart_file.name.startswith(f"config-{config_name}-")
                and flowchart_file.name.endswith("_flowchart.json")
                and "_condensed" not in flowchart_file.name
            ):
                matching_flowcharts.append(flowchart_file)

        if matching_flowcharts:
            # Prefer model-specific if available
            if model_safe:
                model_specific = [
                    f for f in matching_flowcharts if f"_{model_safe}_flowchart.json" in f.name
                ]
                if model_specific:
                    matching_flowcharts = model_specific
            matching_flowcharts.sort(key=lambda x: ("_prev" in x.name, x.name))
            return matching_flowcharts[0]
        return None

    matching_flowcharts = []
    for flowchart_file in flowchart_dir.glob("*.json"):
        # Skip condensed/fully_condensed versions - we want the base flowchart
        if "_condensed" in flowchart_file.name:
            continue

        # Try model-specific naming first if model provided
        if model_safe:
            expected_name = f"config-{config_name}-{f_config_name}_{model_safe}_flowchart.json"
            if flowchart_file.name == expected_name:
                matching_flowcharts.append(flowchart_file)
                continue

        # Try new naming: config-{config_name}-{f_config_name}_flowchart.json
        expected_name = f"config-{config_name}-{f_config_name}_flowchart.json"
        if flowchart_file.name == expected_name:
            matching_flowcharts.append(flowchart_file)
            continue

        # Try old naming
        old_expected_name = f"config-{config_name}_{f_config_name}-{f_config_name}_flowchart.json"
        if flowchart_file.name == old_expected_name:
            matching_flowcharts.append(flowchart_file)

    if matching_flowcharts:
        # Prefer model-specific if available
        if model_safe:
            model_specific = [
                f for f in matching_flowcharts if f"_{model_safe}_flowchart.json" in f.name
            ]
            if model_specific:
                matching_flowcharts = model_specific
        matching_flowcharts.sort(key=lambda x: ("_prev" in x.name, x.name))
        return matching_flowcharts[0]

    return None


def check_resamples_exist(prompt: str, model: str) -> Optional[Path]:
    """Check if resamples directory exists and has prefix directories."""
    resamples_dir = Path(FileUtils.get_resample_dir(prompt, model))

    if not resamples_dir.exists():
        return None

    prefix_dirs = [d for d in resamples_dir.iterdir() if d.is_dir()]

    if not prefix_dirs:
        return None

    return resamples_dir


def get_prefix_from_prefix_name(prefix_name: str, prefix_correctness_data: Dict[str, Any]) -> str:
    """Get the prefix text for a given prefix name."""
    resamples_dir = prefix_correctness_data.get("resamples_directory", "")
    if not resamples_dir:
        return ""

    # Load from prefixes.json
    parts = Path(resamples_dir).parts
    if len(parts) >= 2 and parts[0] == "prompts":
        prompt = parts[1]
        prefixes_path = Path(FileUtils.get_prefixes_file_path(prompt))
        if prefixes_path.exists():
            try:
                with open(prefixes_path, "r") as f:
                    prefixes_data = json.load(f)
                    if prefix_name in prefixes_data:
                        return prefixes_data[prefix_name]
            except (json.JSONDecodeError, IOError):
                pass

    return ""


def get_actual_distribution_for_prefix(
    prefix_name: str, prefix_correctness_data: Dict[str, Any]
) -> Dict[str, float]:
    """Get actual response distribution from individual rollout files."""
    resamples_dir = prefix_correctness_data.get("resamples_directory", "")
    if not resamples_dir:
        return {}

    prefix_dir = Path(resamples_dir) / prefix_name

    if not prefix_dir.exists():
        return {}

    # Load all rollout files for this prefix
    answer_counts = Counter()
    total_rollouts = 0

    for rollout_file in prefix_dir.glob("*.json"):
        try:
            with open(rollout_file, "r") as f:
                rollout_data = json.load(f)

            response = rollout_data.get("processed_response_content", "")
            if response:
                answer_counts[response] += 1
                total_rollouts += 1

        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            continue

    if total_rollouts == 0:
        return {}

    # Convert to probabilities (0-1)
    distribution = {answer: count / total_rollouts for answer, count in answer_counts.items()}

    return distribution


def get_config_prefixes(config_name: str) -> List[str]:
    """Get the list of prefixes specified in the config file."""
    # Try p/ subdirectory first
    config_path = FileUtils.get_p_config_file_path(config_name)
    prefixes = get_config_value(config_path, "prefixes", [])

    if not prefixes:
        # Fallback: try without p/ subdirectory
        config_path = FileUtils.get_config_file_path(config_name)
        prefixes = get_config_value(config_path, "prefixes", [])

    # If still no prefixes, try to find config by _name_ field
    if not prefixes:
        configs_dir = "configs"
        for filename in os.listdir(configs_dir):
            if filename.endswith(".yaml"):
                file_path = os.path.join(configs_dir, filename)
                try:
                    with open(file_path, "r") as f:
                        config_data = yaml.safe_load(f)
                        if config_data and config_data.get("_name_") == config_name:
                            prefixes = config_data.get("prefixes", [])
                            break
                except Exception:
                    continue

    return prefixes


def save_comparison_csv(comparison_results: Dict, output_path: str):
    """Save comparison results to CSV with RMSE."""
    import csv
    import math

    if not comparison_results:
        print("No comparison results to save")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "prefix_name",
                "response",
                "predicted_percentage",
                "actual_percentage",
                "rollouts_used",
            ]
        )

        # Collect all predicted and actual values for RMSE
        all_predicted_values = []
        all_actual_values = []

        for prefix_name, data in comparison_results.items():
            predicted = data["predicted"]
            actual = data["actual"]
            rollouts_used = data.get("rollouts_used", 0)

            all_responses = set(predicted.keys()) | set(actual.keys())

            for response in sorted(all_responses):
                pred_val = predicted.get(response, 0.0)
                actual_val = actual.get(response, 0.0)

                writer.writerow(
                    [prefix_name, response, f"{pred_val:.4f}", f"{actual_val:.4f}", rollouts_used]
                )

                all_predicted_values.append(pred_val)
                all_actual_values.append(actual_val)

        # Compute RMSE
        if len(all_predicted_values) > 0:
            squared_errors = [(p - a) ** 2 for p, a in zip(all_predicted_values, all_actual_values)]
            rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
            writer.writerow(["RMSE", f"{rmse:.4f}", "", "", ""])


def run_prefix_prediction_comparison(
    flowchart_path: str,
    resamples_dir: str,
    config_name: str,
    top_rollouts: int = 50,
    size_filter: int | None = None,
    prompt: str = "",
    model: str = "",
) -> Dict[str, Dict]:
    """Run prefix prediction comparison using the 'current' method."""
    from tqdm import tqdm

    # Load data
    prefix_correctness_data = load_prefix_correctness_data(resamples_dir)
    flowchart_data = load_flowchart_data(flowchart_path)
    cluster_info = flowchart_data["cluster_info"]
    responses = flowchart_data["responses"]

    # Get configuration values
    config_prefixes = get_config_prefixes(config_name)

    # Filter to only process prefixes that exist in both config and resamples directory
    available_prefixes = set(prefix_correctness_data["results"].keys())
    prefixes_to_process = [p for p in config_prefixes if p in available_prefixes]

    print(f"Processing {len(prefixes_to_process)} prefixes from config: {prefixes_to_process}")
    print(f"Using method: current")

    results = {}

    for prefix_name in tqdm(prefixes_to_process):
        prefix_text = get_prefix_from_prefix_name(prefix_name, prefix_correctness_data)

        # Use 'current' method (parameters read from config)
        predicted_result = get_predicted_distribution_for_prefix_current(
            prefix_text,
            cluster_info,
            responses,
            top_rollouts,
            config_name,
            size_filter,
            prompt,
            model,
        )
        predicted_dist = predicted_result.get("distribution", {})
        rollouts_used = predicted_result.get("rollouts_used", 0)
        actual_dist = get_actual_distribution_for_prefix(prefix_name, prefix_correctness_data)
        all_responses = set(predicted_dist.keys()) | set(actual_dist.keys())
        for response in all_responses:
            if response not in predicted_dist:
                predicted_dist[response] = 0.0
            if response not in actual_dist:
                actual_dist[response] = 0.0
        results[prefix_name] = {
            "predicted": predicted_dist,
            "actual": actual_dist,
            "prefix_text": prefix_text,
            "rollouts_used": rollouts_used,
        }

    return results


def get_predicted_distribution_for_prefix_current(
    prefix_text: str,
    cluster_info: Dict[str, Any],
    responses: Dict[str, Any],
    top_rollouts: int = 50,
    config_name: str = "",
    size_filter_override: int | None = None,
    prompt: str = "",
    model: str = "",
) -> Dict:
    """Get predicted distribution using the 'current' method.

    Parameters are read from config:
    - beta: Weight between matching score and entropy (0 = fully entropy-weighted, 1 = ignores entropy)
    - weigh: If true, aggregate by weighted scores; if false, use unweighted fraction
    - strict: If true, exact positional matching; if false, subsequence matching (LCS)
    - sliding: If true, slide window across sequence; if false, only check first position

    Always set (not configurable):
    - alpha: Always 1.0 (no penalty for later window positions)
    - nonzero: Always true (only keep non-zero scores)
    - exclude_matching_rollouts: Always true (exclude rollouts that match prefix exactly)
    """
    from src.chunking import chunk

    # Step 1: Chunk the prefix text
    prefix_chunks, _ = chunk(prefix_text)
    if not prefix_chunks:
        return {"distribution": {}, "rollouts_used": 0}

    # Step 2: Find matching clusters
    matching_clusters = find_matching_clusters(prefix_chunks, cluster_info)
    if not matching_clusters:
        return {"distribution": {}, "rollouts_used": 0}

    L = len(matching_clusters)

    # Get config values
    cfg_path = resolve_p_config_path(config_name) if config_name else None

    # Always set these to true
    nonzero = True
    exclude_matching_rollouts = True

    # Always set alpha to 1.0 (no penalty for later window positions)
    alpha = 1.0

    # Get parameters from config with defaults
    if cfg_path:
        beta = get_config_value(cfg_path, "beta", 0.0)
        weigh = get_config_value(cfg_path, "weigh", True)
        strict = get_config_value(cfg_path, "strict", False)
        sliding = get_config_value(cfg_path, "sliding", True)
        size_filter_pct = (
            size_filter_override
            if size_filter_override is not None
            else get_config_value(cfg_path, "size_filter", None)
        )
        if size_filter_pct is not None:
            size_filter_pct = int(size_filter_pct)
    else:
        # Defaults if no config
        beta = 0.0
        weigh = True
        strict = False
        sliding = True
        size_filter_pct = size_filter_override

    # Optional size filter by cluster frequency (top k percent)
    allowed_clusters: set[str] | None = None
    if size_filter_pct is not None and size_filter_pct >= 0 and size_filter_pct < 100:
        all_clusters = [
            cid
            for cid in cluster_info.keys()
            if isinstance(cid, str) and cid.startswith("cluster-")
        ]
        ranked = sorted(
            all_clusters, key=lambda cid: (-(cluster_info.get(cid, {}).get("freq", 0)), cid)
        )
        kcount = max(0, min(len(ranked), math.ceil(len(ranked) * (size_filter_pct / 100.0))))
        allowed_clusters = set(ranked[:kcount])
        print(
            f"current: size_filter kept {len(allowed_clusters)}/{len(ranked)} clusters ({size_filter_pct}%)"
        )

        matching_clusters = [c for c in matching_clusters if c in allowed_clusters]
        if not matching_clusters:
            return {"distribution": {}, "rollouts_used": 0}
        L = len(matching_clusters)

    # Precompute entropy map
    entropy_map = {
        cid: cluster_info[cid].get("entropy", 0.0)
        for cid in matching_clusters
        if cid in cluster_info
    }

    # Helper: rollout clusters
    def rollout_clusters_of(rid: str, rdata: Dict[str, Any]) -> List[str]:
        return get_clusters_from_rollout({"id": rid, "edges": rdata["edges"]})

    # Scoring per rollout
    scored_rollouts = []
    excluded_rollout_id = None
    for rid in sorted(responses.keys()):
        rdata = responses[rid]
        rc = rollout_clusters_of(rid, rdata)

        # Optional exclusion: drop rollouts that contain the prefix text exactly at the beginning
        if exclude_matching_rollouts and excluded_rollout_id is None:
            rollout_text = ""
            if prompt and model:
                rollout_file = Path(FileUtils.get_rollout_file_path(prompt, model, rid))
                if rollout_file.exists():
                    try:
                        with open(rollout_file, "r") as f:
                            rollout_file_data = json.load(f)
                            rollout_text = rollout_file_data.get("cot_content", "")
                    except (json.JSONDecodeError, IOError):
                        pass

            if rollout_text and rollout_text.strip().startswith(prefix_text.strip()):
                excluded_rollout_id = rid
                print(f"EXCLUDED rollout {rid}: prefix matches exactly")
                continue

        # Apply size filter to rollout cluster sequence (for scoring only)
        if allowed_clusters is not None and rc:
            rc = [c for c in rc if c in allowed_clusters]
        if not rc:
            continue

        n = len(rc)

        # Slide windows from s = 0 .. (n-L) inclusive (or just s=0 if sliding=False)
        best = 0.0
        best_matches = 0
        if sliding:
            max_starts = max(1, n - L + 1)
        else:
            # Only check first position if sliding is disabled
            max_starts = 1
        for s in range(0, max_starts):
            score = 0.0
            start_i = 0
            end_i = min(L, n - s)
            if start_i >= end_i:
                continue
            matches = 0

            if strict:
                # Positional, entropy-weighted match
                for i in range(start_i, end_i):
                    j = s + i
                    if rc[j] == matching_clusters[i]:
                        ent = entropy_map.get(matching_clusters[i], 0.0)
                        score += beta * 1.0 + (1.0 - beta) * (1.0 - ent)
                        matches += 1
            else:
                # Subsequence match (allow gaps): compute LCS alignment and score matched prefix tokens
                window = rc[s : s + end_i]
                m = len(matching_clusters)
                wlen = len(window)
                dp = [[0] * (wlen + 1) for _ in range(m + 1)]
                prev = [[(0, 0)] * (wlen + 1) for _ in range(m + 1)]
                for ii in range(1, m + 1):
                    for jj in range(1, wlen + 1):
                        if matching_clusters[ii - 1] == window[jj - 1]:
                            dp[ii][jj] = dp[ii - 1][jj - 1] + 1
                            prev[ii][jj] = (ii - 1, jj - 1)
                        else:
                            if dp[ii - 1][jj] >= dp[ii][jj - 1]:
                                dp[ii][jj] = dp[ii - 1][jj]
                                prev[ii][jj] = (ii - 1, jj)
                            else:
                                dp[ii][jj] = dp[ii][jj - 1]
                                prev[ii][jj] = (ii, jj - 1)

                # Reconstruct matches
                ii, jj = m, wlen
                matched_prefix_indices: List[int] = []
                while ii > 0 and jj > 0:
                    pii, pjj = prev[ii][jj]
                    if (
                        pii == ii - 1
                        and pjj == jj - 1
                        and matching_clusters[ii - 1] == window[jj - 1]
                    ):
                        matched_prefix_indices.append(ii - 1)
                    ii, jj = pii, pjj
                matched_prefix_indices.reverse()

                for pi in matched_prefix_indices:
                    ent = entropy_map.get(matching_clusters[pi], 0.0)
                    score += beta * 1.0 + (1.0 - beta) * (1.0 - ent)
                matches = len(matched_prefix_indices)

            # Start offset penalty (alpha always 1.0, so no penalty)
            if s > 0:
                score *= alpha**s

            if score > best:
                best = score
                best_matches = matches

        scored_rollouts.append(
            {"rollout_id": rid, "score": best, "rollout_data": rdata, "num_matches": best_matches}
        )

    # Always keep only non-zero scores
    pre_filter_candidates = len(scored_rollouts)
    scored_rollouts = [x for x in scored_rollouts if x["score"] > 0]
    scored_rollouts.sort(key=lambda x: (-x["score"], x["rollout_id"]))
    top_rollout_scores = scored_rollouts[:top_rollouts]

    print(
        f"current: using {len(top_rollout_scores)} rollouts (non-zero only), "
        f"candidates after size filter={pre_filter_candidates}, total responses={len(responses)}"
    )

    if not top_rollout_scores:
        return {"distribution": {}, "rollouts_used": 0}

    # Aggregate by weighted scores (return probabilities 0-1)
    distribution = {}
    if not weigh:
        # Unweighted: fraction correct among top_k
        correct_count = 0
        for info in top_rollout_scores:
            if info["rollout_data"].get("correctness", True):
                correct_count += 1
        frac_correct = (correct_count / len(top_rollout_scores)) if top_rollout_scores else 0.0
        distribution["correct"] = frac_correct
        distribution["incorrect"] = 1.0 - frac_correct
    else:
        # Weighted by scores
        answer_scores = {}
        total_similarity_score = 0.0
        for info in top_rollout_scores:
            ans = info["rollout_data"].get("answer", "")
            w = info["score"]
            answer_scores[ans] = answer_scores.get(ans, 0.0) + w
            total_similarity_score += w
        if total_similarity_score > 0:
            for ans, w in answer_scores.items():
                distribution[ans] = w / total_similarity_score

    return {
        "distribution": distribution,
        "rollouts_used": len(top_rollout_scores),
        "top_scores": [x["score"] for x in top_rollout_scores],
    }


def generate_prefixes_from_rollouts(
    prompt: str,
    model: str,
    num_prefixes: int,
    content_key: str = "chunked_cot_content",
) -> List[str]:
    """Generate random prefixes from rollout files and save to prefixes.json."""
    import random
    from src.utils.file_utils import ensure_dir

    prefixes_path = Path(FileUtils.get_prefixes_file_path(prompt))

    ensure_dir(prefixes_path.parent)

    existing_prefixes_data = {}

    with open(prefixes_path, "r") as f:
        content = f.read().strip()
        if content:
            existing_prefixes_data = json.loads(content)

        existing_count = len(existing_prefixes_data)

        print(
            f"Prefixes file already exists with {existing_count} prefixes. Adding {num_prefixes} more."
        )

    rollouts_dir = Path(FileUtils.get_rollout_dir(prompt, model))

    if not rollouts_dir.exists():
        print(f"Rollouts directory not found: {rollouts_dir}")
        return list(existing_prefixes_data.keys()) if existing_prefixes_data else []

    rollout_files = sorted(rollouts_dir.glob("*.json"))
    if not rollout_files:
        print(f"No rollout files found in: {rollouts_dir}")
        return list(existing_prefixes_data.keys()) if existing_prefixes_data else []

    needed_count = num_prefixes

    suggested_prefix_texts: List[str] = []

    rollout_ids = [f.stem for f in rollout_files]

    random.shuffle(rollout_ids)

    existing_texts = set(existing_prefixes_data.values())

    for rollout_id in rollout_ids:
        if len(suggested_prefix_texts) >= needed_count:
            break

        rollout_file_path = rollouts_dir / f"{rollout_id}.json"

        if not rollout_file_path.exists():
            continue

        with open(rollout_file_path, "r") as f:
            rollout_data = json.load(f)

        chunked_sentences = rollout_data.get(content_key, [])
        if not chunked_sentences and content_key != "chunked_cot_content":
            chunked_sentences = rollout_data.get("chunked_cot_content", [])
        if not chunked_sentences:
            continue

        if len(chunked_sentences) > 1:
            prefix_length = random.randint(
                1,
                len(chunked_sentences),
            )
        else:
            prefix_length = 1

        prefix_sentences = chunked_sentences[:prefix_length]
        prefix_text = " ".join(prefix_sentences)

        if prefix_text and prefix_text.strip() and prefix_text not in existing_texts:
            suggested_prefix_texts.append(prefix_text)
            existing_texts.add(prefix_text)

    prefixes_obj = dict(existing_prefixes_data)

    # Determine the next prefix index, as to not overwrite existing ones
    if existing_prefixes_data:
        existing_indices = []
        for pid in existing_prefixes_data.keys():
            if pid.startswith("prefix-"):
                try:
                    idx = int(pid.split("-")[1])
                    existing_indices.append(idx)
                except (ValueError, IndexError):
                    pass
        next_idx = max(existing_indices) + 1 if existing_indices else 1
    else:
        next_idx = 1

    for txt in suggested_prefix_texts[:needed_count]:
        prefixes_obj[f"prefix-{next_idx}"] = txt
        next_idx += 1

    with open(prefixes_path, "w") as f:
        json.dump(
            prefixes_obj,
            f,
            indent=2,
        )

    all_ids = list(prefixes_obj.keys())

    new_count = len(all_ids) - existing_count

    print(f"Total: {len(all_ids)} prefixes ({new_count} new ones generated)")

    return all_ids

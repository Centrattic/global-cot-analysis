#!/usr/bin/env python3
"""
Compare flowchart graphs between cued and uncued conditions.

Loads the generated flowchart JSONs and computes:
  - Number of clusters and size distributions
  - Edge distributions (transition frequencies)
  - Entropy at key nodes
  - Answer distribution differences per cluster
  - Cluster overlap between conditions

Usage:
    python scripts/compare_graphs.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


MODEL = "deepseek_r1_qwen_14b"  # safe name (underscores)


def find_flowchart(prompt_index: str) -> Path | None:
    """Find the flowchart JSON file for a prompt."""
    flowchart_dir = Path(f"flowcharts/{prompt_index}")
    if not flowchart_dir.exists():
        return None
    # Look for matching files
    candidates = list(flowchart_dir.glob(f"*{MODEL}*flowchart.json"))
    if not candidates:
        candidates = list(flowchart_dir.glob("*flowchart.json"))
    return candidates[0] if candidates else None


def load_flowchart(path: Path) -> dict:
    """Load a flowchart JSON."""
    with open(path) as f:
        return json.load(f)


def extract_cluster_stats(flowchart: dict) -> dict:
    """Extract cluster statistics from a flowchart."""
    nodes = flowchart.get("nodes", [])
    stats = {}
    for node_dict in nodes:
        for cluster_key, node_data in node_dict.items():
            stats[cluster_key] = {
                "freq": node_data.get("freq", 0),
                "entropy": node_data.get("entropy", 0),
                "mean_similarity": node_data.get("mean_similarity", 0),
                "num_rollouts": node_data.get("num_rollouts", 0),
                "representative": node_data.get("representative_sentence", "")[:100],
            }
    return stats


def extract_answer_distribution(flowchart: dict) -> list:
    """Extract per-response answer and correctness."""
    answers = []
    for resp_dict in flowchart.get("responses", []):
        for seed, resp_data in resp_dict.items():
            answer = resp_data.get("answer", "")
            correctness = resp_data.get("correctness", None)
            answers.append({"seed": seed, "answer": answer, "correctness": correctness})
    return answers


def extract_edge_distribution(flowchart: dict) -> Counter:
    """Count edge transitions across all responses."""
    edge_counts = Counter()
    for resp_dict in flowchart.get("responses", []):
        for seed, resp_data in resp_dict.items():
            edges = resp_data.get("edges", [])
            for edge in edges:
                if isinstance(edge, dict):
                    src = edge.get("from", "")
                    dst = edge.get("to", "")
                    if src and dst:
                        edge_counts[(src, dst)] += 1
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    edge_counts[(str(edge[0]), str(edge[1]))] += 1
    return edge_counts


def entropy(dist: dict) -> float:
    """Compute Shannon entropy from a distribution dict {label: count}."""
    total = sum(dist.values())
    if total == 0:
        return 0
    probs = [v / total for v in dist.values() if v > 0]
    return -sum(p * np.log2(p) for p in probs)


def compare(uncued_fc: dict, cued_fc: dict):
    """Compare two flowcharts and print report."""
    print("=" * 60)
    print("GRAPH COMPARISON: UNCUED vs CUED")
    print("=" * 60)

    # 1. Cluster counts
    uncued_stats = extract_cluster_stats(uncued_fc)
    cued_stats = extract_cluster_stats(cued_fc)

    print(f"\n--- Cluster Counts ---")
    print(f"  Uncued: {len(uncued_stats)} clusters")
    print(f"  Cued:   {len(cued_stats)} clusters")

    # 2. Size distributions
    uncued_freqs = sorted([s["freq"] for s in uncued_stats.values()], reverse=True)
    cued_freqs = sorted([s["freq"] for s in cued_stats.values()], reverse=True)

    print(f"\n--- Cluster Size Distribution ---")
    print(f"  Uncued top-5 sizes: {uncued_freqs[:5]}")
    print(f"  Cued top-5 sizes:   {cued_freqs[:5]}")
    print(f"  Uncued mean size:   {np.mean(uncued_freqs):.1f}")
    print(f"  Cued mean size:     {np.mean(cued_freqs):.1f}")

    # 3. Entropy comparison
    uncued_entropies = [s["entropy"] for s in uncued_stats.values() if s["entropy"]]
    cued_entropies = [s["entropy"] for s in cued_stats.values() if s["entropy"]]

    if uncued_entropies and cued_entropies:
        print(f"\n--- Node Entropy ---")
        print(f"  Uncued mean entropy: {np.mean(uncued_entropies):.3f}")
        print(f"  Cued mean entropy:   {np.mean(cued_entropies):.3f}")

    # 4. Edge distributions
    uncued_edges = extract_edge_distribution(uncued_fc)
    cued_edges = extract_edge_distribution(cued_fc)

    print(f"\n--- Edge Statistics ---")
    print(f"  Uncued unique edges: {len(uncued_edges)}")
    print(f"  Cued unique edges:   {len(cued_edges)}")
    print(f"  Uncued total transitions: {sum(uncued_edges.values())}")
    print(f"  Cued total transitions:   {sum(cued_edges.values())}")

    # Edge overlap
    uncued_edge_set = set(uncued_edges.keys())
    cued_edge_set = set(cued_edges.keys())
    shared_edges = uncued_edge_set & cued_edge_set
    if uncued_edge_set | cued_edge_set:
        jaccard = len(shared_edges) / len(uncued_edge_set | cued_edge_set)
        print(f"  Shared edges: {len(shared_edges)}")
        print(f"  Edge Jaccard similarity: {jaccard:.3f}")

    # 5. Answer distributions
    uncued_answers = extract_answer_distribution(uncued_fc)
    cued_answers = extract_answer_distribution(cued_fc)

    uncued_answer_dist = Counter(a["answer"] for a in uncued_answers)
    cued_answer_dist = Counter(a["answer"] for a in cued_answers)

    print(f"\n--- Answer Distributions ---")
    print(f"  Uncued: {dict(uncued_answer_dist.most_common())}")
    print(f"  Cued:   {dict(cued_answer_dist.most_common())}")

    uncued_correct = sum(1 for a in uncued_answers if a.get("correctness") is True)
    cued_correct = sum(1 for a in cued_answers if a.get("correctness") is True)
    print(f"  Uncued accuracy: {uncued_correct}/{len(uncued_answers)}")
    print(f"  Cued accuracy:   {cued_correct}/{len(cued_answers)}")

    # 6. Save results
    results = {
        "uncued": {
            "n_clusters": len(uncued_stats),
            "cluster_sizes": uncued_freqs,
            "mean_entropy": float(np.mean(uncued_entropies)) if uncued_entropies else None,
            "n_unique_edges": len(uncued_edges),
            "answer_distribution": dict(uncued_answer_dist),
            "accuracy": uncued_correct / max(len(uncued_answers), 1),
        },
        "cued": {
            "n_clusters": len(cued_stats),
            "cluster_sizes": cued_freqs,
            "mean_entropy": float(np.mean(cued_entropies)) if cued_entropies else None,
            "n_unique_edges": len(cued_edges),
            "answer_distribution": dict(cued_answer_dist),
            "accuracy": cued_correct / max(len(cued_answers), 1),
        },
        "comparison": {
            "cluster_count_diff": len(cued_stats) - len(uncued_stats),
            "edge_jaccard": jaccard if (uncued_edge_set | cued_edge_set) else None,
            "shared_edges": len(shared_edges),
        },
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "graph_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    uncued_path = find_flowchart("faith_uncued")
    cued_path = find_flowchart("faith_cued")

    if not uncued_path:
        print("ERROR: No uncued flowchart found. Run flowchart generation first:")
        print("  python -m src.main --config-name=faith_uncued command=flowcharts")
        sys.exit(1)
    if not cued_path:
        print("ERROR: No cued flowchart found. Run flowchart generation first:")
        print("  python -m src.main --config-name=faith_cued command=flowcharts")
        sys.exit(1)

    print(f"Loading uncued flowchart: {uncued_path}")
    print(f"Loading cued flowchart:   {cued_path}")

    uncued_fc = load_flowchart(uncued_path)
    cued_fc = load_flowchart(cued_path)

    compare(uncued_fc, cued_fc)


if __name__ == "__main__":
    main()

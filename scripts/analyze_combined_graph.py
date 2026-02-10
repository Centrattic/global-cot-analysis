#!/usr/bin/env python3
"""
Analyze a combined faithfulness flowchart: per-condition statistics.

Loads the single combined flowchart and computes:
- Cluster occupancy per condition (which clusters are cued-dominated vs uncued-dominated)
- Per-condition accuracy
- Edge distribution per condition
- KL divergence of cluster usage between conditions
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

FLOWCHART_PATH = "flowcharts/faith_combined/config-faith_combined-combined_qwen3_8b_flowchart.json"


def load_flowchart(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def split_by_condition(responses: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Split responses into cued and uncued groups."""
    cued = {}
    uncued = {}
    for key, resp in responses.items():
        if resp.get("condition") == "cued":
            cued[key] = resp
        else:
            uncued[key] = resp
    return cued, uncued


def get_cluster_visits(responses: Dict[str, Any]) -> Counter:
    """Count how many rollouts visit each cluster."""
    visits = Counter()
    for resp in responses.values():
        visited = set()
        for edge in resp.get("edges", []):
            for node_key in ["node_a", "node_b"]:
                node = edge.get(node_key, "")
                if node and node != "START":
                    visited.add(node)
        for cluster in visited:
            visits[cluster] += 1
    return visits


def get_edge_counts(responses: Dict[str, Any]) -> Counter:
    """Count edge frequencies."""
    edges = Counter()
    for resp in responses.values():
        for edge in resp.get("edges", []):
            a, b = edge.get("node_a", ""), edge.get("node_b", "")
            if a and b:
                edges[(a, b)] += 1
    return edges


def accuracy(responses: Dict[str, Any]) -> float:
    """Compute fraction of correct responses."""
    if not responses:
        return 0.0
    correct = sum(1 for r in responses.values() if r.get("correctness"))
    return correct / len(responses)


def kl_divergence(p: Dict[str, float], q: Dict[str, float], epsilon: float = 1e-10) -> float:
    """KL(P || Q) over shared key space."""
    all_keys = set(p.keys()) | set(q.keys())
    total = 0.0
    for k in all_keys:
        pk = p.get(k, epsilon)
        qk = q.get(k, epsilon)
        if pk > 0:
            total += pk * math.log2(pk / qk)
    return total


def counts_to_dist(counts: Counter) -> Dict[str, float]:
    """Normalize a counter to a probability distribution."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def main():
    flowchart = load_flowchart(FLOWCHART_PATH)
    responses = flowchart["responses"]
    nodes = flowchart["nodes"]

    cued, uncued = split_by_condition(responses)
    print(f"Total responses: {len(responses)}")
    print(f"  Cued: {len(cued)}")
    print(f"  Uncued: {len(uncued)}")
    print()

    # Per-condition accuracy
    acc_cued = accuracy(cued)
    acc_uncued = accuracy(uncued)
    print(f"Accuracy:")
    print(f"  Cued:   {acc_cued:.1%}")
    print(f"  Uncued: {acc_uncued:.1%}")
    print(f"  Gap:    {acc_uncued - acc_cued:.1%}")
    print()

    # Cluster occupancy
    visits_cued = get_cluster_visits(cued)
    visits_uncued = get_cluster_visits(uncued)
    all_clusters = sorted(set(visits_cued.keys()) | set(visits_uncued.keys()))

    print(f"Clusters visited: {len(all_clusters)} total")
    print(f"  Cued-only:   {len(set(visits_cued.keys()) - set(visits_uncued.keys()))}")
    print(f"  Uncued-only: {len(set(visits_uncued.keys()) - set(visits_cued.keys()))}")
    print(f"  Shared:      {len(set(visits_cued.keys()) & set(visits_uncued.keys()))}")
    print()

    # Top clusters by dominance ratio
    print("Top 10 clusters by cued/uncued ratio (cued-dominated):")
    ratios = []
    for c in all_clusters:
        cu = visits_cued.get(c, 0)
        un = visits_uncued.get(c, 0)
        total = cu + un
        if total >= 3:  # min freq filter
            ratio = cu / (un + 1e-10)
            ratios.append((c, cu, un, ratio))
    ratios.sort(key=lambda x: -x[3])
    for c, cu, un, ratio in ratios[:10]:
        print(f"  {c}: cued={cu}, uncued={un}, ratio={ratio:.1f}")
    print()

    print("Top 10 clusters by uncued/cued ratio (uncued-dominated):")
    ratios.sort(key=lambda x: x[3])
    for c, cu, un, ratio in ratios[:10]:
        inv_ratio = un / (cu + 1e-10)
        print(f"  {c}: cued={cu}, uncued={un}, ratio={inv_ratio:.1f}")
    print()

    # Edge distribution
    edges_cued = get_edge_counts(cued)
    edges_uncued = get_edge_counts(uncued)
    all_edges = set(edges_cued.keys()) | set(edges_uncued.keys())
    print(f"Unique edges: {len(all_edges)} total")
    print(f"  Cued-only:   {len(set(edges_cued.keys()) - set(edges_uncued.keys()))}")
    print(f"  Uncued-only: {len(set(edges_uncued.keys()) - set(edges_cued.keys()))}")
    print(f"  Shared:      {len(set(edges_cued.keys()) & set(edges_uncued.keys()))}")
    print()

    # KL divergence of cluster usage
    dist_cued = counts_to_dist(visits_cued)
    dist_uncued = counts_to_dist(visits_uncued)
    kl_cu = kl_divergence(dist_cued, dist_uncued)
    kl_uc = kl_divergence(dist_uncued, dist_cued)
    print(f"KL divergence of cluster usage:")
    print(f"  KL(cued || uncued) = {kl_cu:.4f} bits")
    print(f"  KL(uncued || cued) = {kl_uc:.4f} bits")
    print(f"  Symmetric KL      = {(kl_cu + kl_uc) / 2:.4f} bits")
    print()

    # KL divergence of edge usage
    dist_edges_cued = counts_to_dist(edges_cued)
    dist_edges_uncued = counts_to_dist(edges_uncued)
    kl_e_cu = kl_divergence(dist_edges_cued, dist_edges_uncued)
    kl_e_uc = kl_divergence(dist_edges_uncued, dist_edges_cued)
    print(f"KL divergence of edge usage:")
    print(f"  KL(cued || uncued) = {kl_e_cu:.4f} bits")
    print(f"  KL(uncued || cued) = {kl_e_uc:.4f} bits")
    print(f"  Symmetric KL      = {(kl_e_cu + kl_e_uc) / 2:.4f} bits")

    # Answer distribution per condition
    print()
    print("Answer distribution:")
    for label, group in [("Cued", cued), ("Uncued", uncued)]:
        answers = Counter()
        for r in group.values():
            ans = r.get("processed_response_content", "") or r.get("answer", "")
            answers[ans] += 1
        total = sum(answers.values())
        print(f"  {label}:")
        for ans, cnt in answers.most_common():
            print(f"    {ans}: {cnt} ({cnt / total:.0%})")


if __name__ == "__main__":
    main()

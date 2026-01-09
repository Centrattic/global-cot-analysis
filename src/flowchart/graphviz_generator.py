#!/usr/bin/env python3
"""
Graphviz generator that creates graph layout embeddings for existing flowcharts.
"""

from pathlib import Path
from typing import Dict, Any, List
import requests
import time
import os

from src.utils.json_utils import load_json, write_json
from src.utils.file_utils import FileUtils
from src.flowchart.condense_flowchart import condense_flowchart_file, condense_flowchart_fully_file


class GraphvizGenerator:
    """Generates graphviz embeddings for existing flowcharts."""

    def _check_graph_layout_service(self) -> bool:
        """Check if the graph layout service is running."""
        try:
            response = requests.get("http://127.0.0.1:8010/", timeout=5)
            if response.status_code == 200:
                print("Graph layout service is running")
                return True
            else:
                print(
                    f"Graph layout service returned status code: {response.status_code}"
                )
                return False
        except requests.exceptions.RequestException as e:
            print(f"Graph layout service is not running: {e}")
            return False

    def _get_graph_layout(self,
                          flowchart: Dict[str, Any],
                          dataset_id: str = None) -> Dict[str, Any]:
        """Get graph layout from the layout service."""
        try:
            cluster_key_to_int = {}
            int_to_cluster_key = {}
            next_int_id = 0

            for node_obj in flowchart["nodes"]:
                cluster_key = list(node_obj.keys())[0]
                node_data = node_obj[cluster_key]
                cluster_key_to_int[cluster_key] = next_int_id
                int_to_cluster_key[next_int_id] = cluster_key
                next_int_id += 1

            cluster_key_to_int["START"] = next_int_id
            int_to_cluster_key[next_int_id] = "START"
            next_int_id += 1

            response_nodes = set()
            for seed, rollout_info in flowchart["responses"].items():
                edges = rollout_info.get("edges", [])
                for edge in edges:
                    if edge["node_a"].startswith("response-"):
                        response_nodes.add(edge["node_a"])
                    if edge["node_b"].startswith("response-"):
                        response_nodes.add(edge["node_b"])

            for response_node in sorted(response_nodes):
                cluster_key_to_int[response_node] = next_int_id
                int_to_cluster_key[next_int_id] = response_node
                next_int_id += 1

            nodes_payload = []
            special_nodes = {"START"} | response_nodes

            for node_obj in flowchart["nodes"]:
                cluster_key = list(node_obj.keys())[0]
                if cluster_key not in special_nodes:
                    node_data = node_obj[cluster_key]
                    int_id = cluster_key_to_int[cluster_key]
                    nodes_payload.append({
                        "id": str(int_id),
                        "freq": node_data["freq"]
                    })

            nodes_payload.append({
                "id": str(cluster_key_to_int["START"]),
                "freq": 0
            })

            for response_node in sorted(response_nodes):
                nodes_payload.append({
                    "id":
                    str(cluster_key_to_int[response_node]),
                    "freq":
                    0
                })

            edges_payload = []
            edge_keys = set()
            valid_node_ids = {node["id"] for node in nodes_payload}

            def add_edge(a_key: str, b_key: str):
                a_int = str(cluster_key_to_int.get(a_key))
                b_int = str(cluster_key_to_int.get(b_key))

                if a_int not in valid_node_ids or b_int not in valid_node_ids:
                    return
                key = f"{min(a_int, b_int)}|{max(a_int, b_int)}"
                if key not in edge_keys:
                    edge_keys.add(key)
                    edges_payload.append({"source": a_int, "target": b_int})

            for seed, rollout_info in flowchart["responses"].items():
                edges = rollout_info.get("edges", [])
                for edge in edges:
                    add_edge(edge["node_a"], edge["node_b"])
                if edges:
                    add_edge("START", edges[0]["node_a"])

            payload = {
                "dataset_id": dataset_id or "default",
                "nodes": nodes_payload,
                "edges": edges_payload,
                "options": {
                    "engine": "sfdp",
                    "width": 1200,
                    "height": 800,
                    "padding": 20
                }
            }

            print(
                f"Requesting graph layout for {len(nodes_payload)} nodes and {len(edges_payload)} edges"
            )

            response = requests.post(
                "http://127.0.0.1:8010/graph/layout",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=500)

            if response.status_code == 200:
                layout_data = response.json()
                raw_positions = layout_data.get("positions", {})
                print(
                    f"Received graph layout with {len(raw_positions)} node positions"
                )

                positions = {}
                for int_id_str, pos in raw_positions.items():
                    int_id = int(int_id_str)
                    cluster_key = int_to_cluster_key.get(int_id)
                    if cluster_key:
                        positions[cluster_key] = pos

                print(
                    f"Mapped {len(positions)} positions back to cluster keys")
                return positions
            else:
                print(
                    f"Graph layout service error: {response.status_code} - {response.text}"
                )
                return {}

        except requests.exceptions.RequestException as e:
            print(f"Error requesting graph layout: {e}")
            return {}
        except Exception as e:
            print(f"Error processing graph layout: {e}")
            return {}

    def _maybe_generate_layout(self,
                               flowchart_path: str,
                               recompute: bool = False) -> bool:
        """Generate and persist graph_layout for a single flowchart file if missing.

        Returns True if generation was performed, False if skipped.
        """
        if not FileUtils.file_exists(flowchart_path):
            print(f"Flowchart file not found at {flowchart_path}")
            return False

        print(f"Loading flowchart from {flowchart_path}")
        flowchart = load_json(flowchart_path)
        if not flowchart:
            print("Failed to load flowchart data")
            return False

        if not recompute and isinstance(flowchart.get("graph_layout"),
                                        dict) and flowchart["graph_layout"]:
            print(
                f"graph_layout already present for {flowchart_path}; skipping generation"
            )
            return False

        # If recompute is True, clear both the embedded layout and the cache file
        if recompute:
            print(
                f"Recompute flag is True, clearing existing layout and cache")
            flowchart.pop("graph_layout", None)
            cache_path = FileUtils.get_graph_cache_file_path(flowchart_path)
            if os.path.exists(cache_path):
                print(f"Clearing cache file: {cache_path}")
                os.remove(cache_path)

        print("Getting graph layout from layout service...")
        # Use path-derived dataset id to avoid collisions while staying stable per run
        dataset_id = Path(flowchart_path).stem
        print(f"Using dataset_id: {dataset_id}")

        graph_layout = self._get_graph_layout(flowchart, dataset_id=dataset_id)
        print(f"Generated graph layout with {len(graph_layout)} positions")
        flowchart["graph_layout"] = graph_layout
        FileUtils.ensure_dir(Path(flowchart_path).parent)
        write_json(flowchart_path, flowchart)
        print(f"Updated flowchart with new graph layout at {flowchart_path}")
        return True

    def _sync_property_checkers(self, original_path: str,
                                condensed_path: str) -> None:
        """Copy property checker metadata from the original flowchart to the condensed one.

        - Copies the top-level `property_checkers` list
        - For every rollout in `responses`, copies the fields listed in `property_checkers`
        """
        original = load_json(original_path)
        condensed = load_json(condensed_path)

        if "property_checkers" in original:
            condensed["property_checkers"] = list(
                original["property_checkers"])  # keep order

            # Sync per-response tags
            if "responses" in original and "responses" in condensed:
                o_responses = original["responses"]
                c_responses = condensed["responses"]
                for rid, o_resp in o_responses.items():
                    if rid in c_responses:
                        c_resp = c_responses[rid]
                        for pname in original["property_checkers"]:
                            if pname in o_resp:
                                c_resp[pname] = o_resp[pname]

            write_json(condensed_path, condensed)

    def generate_graphviz_from_config(self,
                                      config: Dict[str, Any],
                                      recompute: bool = False):
        """Generate graphviz embeddings for both original and condensed flowcharts.

        - If either already has graph_layout and recompute=False, skip recomputing that one.
        - Ensure the condensed file exists; if not, create it.
        """
        if not self._check_graph_layout_service():
            raise RuntimeError(
                "Graph layout service is not running. Start it with: cd graph_layout_service && python -m uvicorn app:app --host 127.0.0.1 --port 8010 --reload"
            )

        prompt_index = config["prompt"]
        flowchart_path = FileUtils.get_flowchart_file_path(
            prompt_index, config._name_, config.f._name_)

        # Original
        self._maybe_generate_layout(flowchart_path, recompute=recompute)

        # Condensed
        condensed_path = str(
            Path(flowchart_path).with_name(
                Path(flowchart_path).stem + "_condensed.json"))
        if not FileUtils.file_exists(condensed_path):
            try:
                print(
                    f"Condensed flowchart does not exist, creating: {condensed_path}"
                )
                out_path = condense_flowchart_file(flowchart_path)
                print(f"Condensed flowchart written to {out_path}")
                condensed_path = out_path
            except Exception as e:
                print(f"Failed to create condensed flowchart: {e}")
                return

        # Ensure property checkers are synced from original to condensed
        self._sync_property_checkers(flowchart_path, condensed_path)
        self._maybe_generate_layout(condensed_path, recompute=recompute)

        # Fully Condensed (collapse all cycles)
        fully_condensed_path = str(
            Path(flowchart_path).with_name(
                Path(flowchart_path).stem + "_fully_condensed.json"))
        if not FileUtils.file_exists(fully_condensed_path):
            try:
                print(
                    f"Fully condensed flowchart does not exist, creating: {fully_condensed_path}"
                )
                out_path_fc = condense_flowchart_fully_file(flowchart_path)
                print(f"Fully condensed flowchart written to {out_path_fc}")
                fully_condensed_path = out_path_fc
            except Exception as e:
                print(f"Failed to create fully condensed flowchart: {e}")
                return

        # Sync property checkers and layout for fully condensed
        self._sync_property_checkers(flowchart_path, fully_condensed_path)
        self._maybe_generate_layout(fully_condensed_path)
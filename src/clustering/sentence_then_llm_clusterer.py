"""
Sentence-then-LLM clustering for generating flowcharts.
"""

import json
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import igraph as ig
import leidenalg as la
import numpy as np
import requests
from dotenv import load_dotenv
from joblib import (
    Parallel,
    delayed,
)
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

load_dotenv()

from src.utils.json_utils import load_json

from .base import BaseClusterer


@dataclass
class Cluster:
    """Represents a cluster of sentences."""

    sentences: List[str]
    id: str
    merged_from: List[str] = None

    def __str__(self):
        return " | ".join(self.sentences)

    def merge_with(self, other: "Cluster"):
        """Merge another cluster into this one."""

        self.sentences.extend(other.sentences)


class SentenceThenLLMClusterer(BaseClusterer):
    """Clusters responses using sentence embeddings first, then LLM refinement."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sentence_model_name = config.get(
            "sentence_embedding_model", "sentence-transformers/paraphrase-mpnet-base-v2"
        )
        self.llm_model = config.get(
            "llm_model",
            "openai/gpt-4o-mini",
        )

        self.sentence_similarity_threshold = config.get(
            "sentence_similarity_threshold",
            0.7,
        )

        self.llm_cluster_threshold = config.get(
            "llm_cluster_threshold",
            0.5,
        )
        self.max_workers = config.get(
            "max_workers",
            100,
        )
        self.request_delay = config.get(
            "request_delay",
            0.1,
        )
        self.use_llm = True
        self.use_agglomerative = True
        self.sentence_model = None
        self.api_key = None
        self.base_url = None
        self.headers = None
        self._lock = None
        self.gamma = config.get(
            "gamma",
            0.8,
        )

        self._load_models()

    def _load_models(self):
        """Load the sentence transformer model and optionally OpenAI client."""

        print(f"Loading sentence transformer model: {self.sentence_model_name}")

        self.sentence_model = SentenceTransformer(
            self.sentence_model_name,
            device="cpu",
        )
        print("Sentence model loaded successfully")

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable."
            )

        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._lock = threading.Lock()

    def cluster_responses(
        self,
        responses: Dict[str, Any],
        prompt_index: str,
        model: str,
    ) -> Dict[str, int]:
        """Cluster responses using sentence embeddings, optionally with LLM refinement."""

        method_name = "sentence-then-LLM"
        print(f"Clustering {len(responses)} responses using {method_name} method")

        print("Step 1: Initial clustering with sentence embeddings")
        initial_clusters = self._sentence_clustering(responses)

        print("Step 2: Preparing clusters for analysis")
        clusters = self._create_cluster_objects(
            responses,
            initial_clusters,
        )

        print("Step 3: Finding candidate pairs for LLM comparison")
        candidate_pairs = self._find_candidate_pairs(clusters)

        print("Step 4: LLM refinement of clusters")
        merged_clusters = self._llm_merge_clusters(
            clusters,
            candidate_pairs,
        )

        print("Step 5: Creating rollout edges")
        rollout_edges = self._create_rollout_edges(
            responses,
            merged_clusters,
            prompt_index,
        )

        self.rollout_edges = rollout_edges
        self.merged_clusters = merged_clusters

        # Fake cluster assignments for compatibility with base class (all assigned to 0)
        cluster_assignments = {seed: 0 for seed in responses.keys()}
        return cluster_assignments

    def _get_rollouts_for_cluster(self, cluster_idx: int) -> List[str]:
        """Get list of rollout IDs that pass through this cluster."""

        cluster_key = f"cluster-{cluster_idx}"
        rollouts = []
        for seed, edges in self.rollout_edges.items():
            for edge in edges:
                if edge.get("node_a") == cluster_key or edge.get("node_b") == cluster_key:
                    rollouts.append(seed)
                    break
        return rollouts

    def _get_content_key(self) -> str:
        """Get the content key to use based on config flag."""

        use_sentences = self.config.get("sentences_instead_of_chunks", False)

        return "sentences" if use_sentences else "chunked_cot_content"

    def _calculate_cluster_mean_similarity(
        self,
        cluster: Cluster,
        rollout_ids: List[str],
        responses: Dict[str, Any],
    ) -> float:
        """Calculate mean cosine similarity for sentences in cluster."""

        if not cluster.sentences or len(cluster.sentences) < 2:
            return 1.0

        embeddings = []
        content_key = self._get_content_key()
        for sentence in cluster.sentences:
            for rollout_id in rollout_ids:
                if rollout_id in responses:
                    response_data = responses[rollout_id]
                    content = response_data.get(content_key, [])
                    sentence_embeddings = response_data.get("sentence_embeddings", [])

                    if sentence in content and len(sentence_embeddings) > 0:
                        idx = content.index(sentence)
                        if idx < len(sentence_embeddings):
                            embeddings.append(sentence_embeddings[idx])
                            break

        if len(embeddings) < 2:
            return 1.0

        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(
            embeddings_array,
            axis=1,
            keepdims=True,
        )
        norms[norms == 0] = 1
        normalized = embeddings_array / norms

        similarities = np.dot(
            normalized,
            normalized.T,
        )
        upper_triangle = np.triu_indices_from(
            similarities,
            k=1,
        )
        if len(upper_triangle[0]) == 0:
            return 1.0

        return float(np.mean(similarities[upper_triangle]))

    def _calculate_cluster_entropy(
        self,
        rollout_ids: List[str],
        responses: Dict[str, Any],
    ) -> float:
        """Calculate Shannon entropy from answer distribution."""
        import math
        from collections import Counter

        answers = []
        seen_rollouts = set()
        for rollout_id in rollout_ids:
            if rollout_id not in seen_rollouts and rollout_id in responses:
                answer = responses[rollout_id].get("processed_response_content", "")
                if answer:
                    answers.append(answer)
                seen_rollouts.add(rollout_id)

        if not answers:
            return 0.0

        answer_counts = Counter(answers)
        total = len(answers)
        entropy = 0.0
        for count in answer_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def create_flowchart(
        self,
        responses: Dict[str, Any],
        cluster_assignments: Dict[str, int],
        prompt_index: str,
        models: List[str],
        config_name: str,
        property_checker_names: List[str] = None,
        node_property_checker_names: List[str] = None,
    ) -> Dict[str, Any]:
        """Create flowchart from clustered responses using rollout edges."""

        from src.labeling.cluster_labeler import load_prompt_text

        prompt_text = load_prompt_text("prompts/prompts.json", prompt_index)

        from pathlib import Path

        from src.utils.json_utils import load_json

        algorithms_path = Path("prompts/algorithms.json")
        algorithms = {}
        if algorithms_path.exists():
            algorithms_data = load_json(str(algorithms_path))
            algorithms = algorithms_data.get(prompt_index, {})

        flowchart = {
            "prompt_index": prompt_index,
            "prompt": prompt_text,
            "algorithms": algorithms,
            "models": models,
            "config_name": config_name,
            "clustering_method": "sentence_then_llm",
            "nodes": [],
            "responses": {},
            "graph_layout": {},
        }

        # Create nodes from merged clusters
        for i, cluster in enumerate(self.merged_clusters):
            # Get rollouts that pass through this cluster
            rollout_ids = self._get_rollouts_for_cluster(i)
            unique_rollouts = list(set(rollout_ids))

            # Aggregate sentences by text
            sentence_counts = {}
            for sentence in cluster.sentences:
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1

            # Create sentence breakdown (without rollout_ids)
            sentences = []
            unique_sentences = list(sentence_counts.keys())

            # node property checkers if specified
            node_checker_values = {}
            if node_property_checker_names:
                from src.property_checkers import PropertyCheckerMultiAlgorithm

                # Registry of node property checkers that support get_value_for_node
                node_checker_registry = {
                    "multi_algorithm": PropertyCheckerMultiAlgorithm,
                }

                # Initialize node property checkers
                node_checkers = {}
                for checker_name in node_property_checker_names:
                    if checker_name in node_checker_registry:
                        node_checkers[checker_name] = node_checker_registry[checker_name]()

                # get_value on sentences
                for checker_name, checker in node_checkers.items():
                    if checker_name == "multi_algorithm":
                        values = checker.get_value_for_node(
                            unique_sentences,
                            prompt_index,
                        )
                    else:
                        values = checker.get_value_for_node(unique_sentences)
                    for idx, sentence in enumerate(unique_sentences):
                        if sentence not in node_checker_values:
                            node_checker_values[sentence] = {}
                        node_checker_values[sentence][checker_name] = (
                            values[idx] if idx < len(values) else []
                        )

            for sentence, count in sentence_counts.items():
                sentence_data = {"text": sentence, "count": count}

                if sentence in node_checker_values:
                    for key, value in node_checker_values[sentence].items():
                        sentence_data[key] = value
                sentences.append(sentence_data)

            # sort by count (descending)
            sentences.sort(key=lambda x: -x["count"])

            # add node in new format: {cluster_key: node_data}
            cluster_key = f"cluster-{i}"
            node_data = {
                "freq": len(cluster.sentences),
                "representative_sentence": sentences[0]["text"] if sentences else "",
                "mean_similarity": self._calculate_cluster_mean_similarity(
                    cluster,
                    unique_rollouts,
                    responses,
                ),
                "num_rollouts": len(unique_rollouts),
                "entropy": self._calculate_cluster_entropy(
                    unique_rollouts,
                    responses,
                ),
                "sentences": sentences,
            }
            node = {cluster_key: node_data}
            flowchart["nodes"].append(node)

        # create rollouts with edges
        for seed, response_data in responses.items():
            rollout_data = {
                "index": seed,
                "seed": response_data.get("seed", None),
                "answer": response_data.get("processed_response_content", ""),
                "edges": self.rollout_edges.get(seed, []),
                "correctness": response_data.get("correctness", False),
            }

            # add property checker values
            if property_checker_names:
                for checker_name in property_checker_names:
                    rollout_data[checker_name] = response_data.get(checker_name, None)

            flowchart["responses"][seed] = rollout_data

        # create response nodes (after rollouts are created)
        self._create_response_nodes(flowchart, responses, prompt_index)

        return flowchart

    def _create_response_nodes(
        self, flowchart: Dict[str, Any], responses: Dict[str, Any], prompt_index: str
    ):
        """Create response nodes for the flowchart."""

        from src.utils.prompt_utils import get_prompt_filter

        prompt_filter = get_prompt_filter(prompt_index)
        if not prompt_filter:
            print(f"No prompt filter found for {prompt_index}, skipping response nodes")
            return

        response_values = {}
        print(f"Processing {len(flowchart['responses'])} responses for response nodes...")

        # process each rollout to create response nodes
        for seed, response_info in flowchart["responses"].items():
            response_data = responses.get(seed)
            if not response_data:
                continue

            try:
                final_answer = prompt_filter.extract_final_answer(response_data)

                # Skip empty responses
                if not final_answer or not final_answer.strip():
                    continue
                response_node_id = f"response-{final_answer}"

                # track frequency of this response value
                if response_node_id not in response_values:
                    response_values[response_node_id] = {
                        "freq": 0,
                        "representative_sentence": f"Final answer: {final_answer}",
                        "mean_similarity": 1.0,
                        "sentences": [{"text": f"Final answer: {final_answer}", "count": 0}],
                    }
                response_values[response_node_id]["freq"] += 1
                response_values[response_node_id]["sentences"][0]["count"] += 1

            except Exception as e:
                print(f"Error processing response for seed {seed}: {e}")

        for response_node_id, node_data in response_values.items():
            response_node = {response_node_id: node_data}
            flowchart["nodes"].append(response_node)

        print(f"Created {len(response_values)} response nodes: {list(response_values.keys())}")

    def _sentence_clustering(self, responses: Dict[str, Any]) -> Dict[str, int]:
        """Perform initial clustering using sentence embeddings."""

        # extract sentences/chunks from responses
        sentences = []
        seed_to_sentences = {}
        content_key = self._get_content_key()

        for seed, response_data in responses.items():
            content = response_data.get(content_key, None)
            if not content:
                content = response_data.get("cot_content", "")
            if content:
                if isinstance(content, list):
                    response_sentences = content
                else:
                    response_sentences = [content]

                sentences.extend(response_sentences)
                seed_to_sentences[seed] = response_sentences

        if not sentences:
            raise ValueError("No sentences found in responses - cannot perform clustering")

        print(f"Extracted {len(sentences)} sentences for embedding")

        print("Generating sentence embeddings...")
        embeddings = self.sentence_model.encode(sentences, normalize_embeddings=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")

        print(
            f"Clustering using adjacency graph with threshold {self.sentence_similarity_threshold}..."
        )
        sentence_clusters = self._cluster_by_cosine_similarity(
            embeddings,
            self.sentence_similarity_threshold,
        )

        # Store sentence clusters and sentence-to-cluster mapping for edge creation
        # The edges will be created by tracing each rollout's path through the clusters

        print(
            f"Created {len(set(sentence_clusters))} sentence clusters from {len(sentences)} sentences"
        )

        # store the sentence clusters and mapping for later use in edge creation
        self.sentence_clusters = sentence_clusters
        self.seed_to_sentences = seed_to_sentences

        return {}

    def _create_cluster_objects(
        self, responses: Dict[str, Any], initial_clusters: Dict[str, int]
    ) -> List[Cluster]:
        """Create Cluster objects from sentence clustering results."""
        # Each sentence cluster becomes a Cluster object

        # Group sentences by their cluster ID
        cluster_id_to_sentences = defaultdict(list)
        sentence_idx = 0

        for seed, response_sentences in self.seed_to_sentences.items():
            for sentence in response_sentences:
                cluster_id = self.sentence_clusters[sentence_idx]
                cluster_id_to_sentences[cluster_id].append(sentence)
                sentence_idx += 1

        # Create Cluster objects
        clusters = []
        for cluster_id, sentences in cluster_id_to_sentences.items():
            if sentences:  # only create clusters with sentences
                cluster = Cluster(sentences=sentences, id=str(cluster_id))
                clusters.append(cluster)

        print(
            f"Created {len(clusters)} clusters from {len(cluster_id_to_sentences)} sentence clusters"
        )
        return clusters

    def _find_candidate_pairs(self, clusters: List[Cluster]) -> List[Tuple[str, str]]:
        """Find candidate pairs for LLM comparison based on similarity threshold."""
        candidate_pairs = []

        print(f"Computing pairwise similarities between {len(clusters)} clusters...")
        print(f"LLM cluster threshold: {self.llm_cluster_threshold}")

        print("Pre-computing cluster embeddings...")
        cluster_embeddings = {}
        for i, cluster in enumerate(tqdm(clusters, desc="Computing cluster embeddings")):
            if cluster.sentences:
                # Get embeddings for all sentences in cluster
                embeddings = self.sentence_model.encode(
                    cluster.sentences,
                    normalize_embeddings=True,
                )
                # Compute averaged normalized vector
                cluster_embeddings[cluster.id] = np.mean(
                    embeddings,
                    axis=0,
                )

        # calculate pairwise similarities using pre-computed embeddings
        total_pairs = len(clusters) * (len(clusters) - 1) // 2
        for i in tqdm(range(len(clusters)), desc="Computing cluster similarities"):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]

                # calculate similarity using pre-computed averaged embeddings
                similarity = self._calculate_cluster_similarity_fast(
                    cluster1,
                    cluster2,
                    cluster_embeddings,
                )

                # add to candidates if similarity is above threshold
                if similarity >= self.llm_cluster_threshold:
                    candidate_pairs.append(
                        (
                            cluster1.id,
                            cluster2.id,
                        )
                    )

        print(f"Found {len(candidate_pairs)} candidate pairs for LLM comparison")
        return candidate_pairs

    def _calculate_cluster_similarity_fast(
        self, cluster1: Cluster, cluster2: Cluster, cluster_embeddings: Dict[str, np.ndarray]
    ) -> float:
        """Calculate similarity between two clusters using pre-computed averaged embeddings."""
        if not cluster1.sentences or not cluster2.sentences:
            return 0.0

        # get pre-computed averaged embeddings
        emb1 = cluster_embeddings.get(cluster1.id)
        emb2 = cluster_embeddings.get(cluster2.id)

        if emb1 is None or emb2 is None:
            return 0.0

        # calculate cosine similarity between averaged normalized vectors for efficiency
        similarity = float(
            np.dot(
                emb1,
                emb2,
            )
        )
        return similarity

    def _llm_merge_clusters(
        self,
        clusters: List[Cluster],
        candidate_pairs: List[Tuple[str, str]],
    ) -> List[Cluster]:
        """Merge clusters using LLM analysis with parallel processing."""
        if not candidate_pairs:
            print("No candidate pairs for LLM merging")
            return clusters

        # Create mapping from cluster ID to cluster object
        cluster_id_to_cluster = {cluster.id: cluster for cluster in clusters}

        # Process candidate pairs in parallel
        print(
            f"Processing {len(candidate_pairs)} candidate pairs with {self.max_workers} parallel workers"
        )

        merge_decisions = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(
                    self._process_cluster_pair, cluster_id1, cluster_id2, cluster_id_to_cluster
                ): (cluster_id1, cluster_id2)
                for cluster_id1, cluster_id2 in candidate_pairs
            }

            # Process completed tasks with progress bar
            for future in tqdm(
                as_completed(future_to_pair),
                total=len(candidate_pairs),
                desc="Processing LLM clustering",
            ):
                cluster_id1, cluster_id2, should_merge = future.result()
                if should_merge:
                    merge_decisions.append(
                        (
                            cluster_id1,
                            cluster_id2,
                        )
                    )

        print(f"LLM decided to merge {len(merge_decisions)} pairs")

        if isinstance(self.gamma, str) and self.gamma == "max-cliques":
            merged_clusters = self._merge_fully_connected_components(
                clusters,
                merge_decisions,
            )
        else:
            merged_clusters = self._merge_leiden_components(
                clusters,
                merge_decisions,
            )

        return merged_clusters

    def _process_cluster_pair(
        self,
        cluster_id1: str,
        cluster_id2: str,
        cluster_id_to_cluster: Dict[str, Cluster],
    ) -> Tuple[str, str, bool]:
        """
        Process a single cluster pair comparison in a thread.

        Args:
            cluster_id1: First cluster ID
            cluster_id2: Second cluster ID
            cluster_id_to_cluster: Mapping from cluster ID to cluster object

        Returns:
            Tuple of (cluster_id1, cluster_id2, should_merge)
        """
        cluster1 = cluster_id_to_cluster.get(cluster_id1)
        cluster2 = cluster_id_to_cluster.get(cluster_id2)

        if cluster1 is None or cluster2 is None:
            print(f"Cluster {cluster_id1} or {cluster_id2} not found, skipping")
            return cluster_id1, cluster_id2, False

        print(f"Comparing clusters {cluster1.id} and {cluster2.id}")

        should_merge = self._call_llm_for_merge_decision(cluster1, cluster2)

        if should_merge:
            print(f"Clusters {cluster1.id} and {cluster2.id} should be merged")

        return cluster_id1, cluster_id2, should_merge

    def _analyze_cluster_content(
        self,
        cluster: Cluster,
    ) -> str:
        """Analyze the content of a cluster to provide better context for LLM."""
        if not cluster.sentences:
            return "Empty cluster"

        # Create a structured analysis
        analysis = f"Cluster ID: {cluster.id}\n"
        analysis += f"Number of sentences: {len(cluster.sentences)}\n"
        analysis += "Sentences:\n"

        for i, sentence in enumerate(cluster.sentences[:10]):  # Limit to first 10 sentences
            analysis += f"  {i + 1}. {sentence}\n"

        if len(cluster.sentences) > 10:
            analysis += f"  ... and {len(cluster.sentences) - 10} more sentences\n"

        return analysis

    def _call_llm_for_merge_decision(
        self,
        cluster1: Cluster,
        cluster2: Cluster,
    ) -> bool:
        """Call LLM to decide if two clusters should be merged."""
        # Create a more detailed analysis of the clusters
        cluster1_analysis = self._analyze_cluster_content(cluster1)
        cluster2_analysis = self._analyze_cluster_content(cluster2)

        prompt = f"""You are an expert at analyzing text clusters. Your task is to determine if two clusters should be merged.

IMPORTANT: Only merge if ALL sentences in both clusters represent the SAME IDEA.

MERGE the clusters if they represent the SAME CONCEPT, even if:
- They use different wording (e.g., "calculate" vs "find" vs "compute")
- They have different levels of detail (e.g., "add numbers" vs "add the two numbers together")

DO NOT MERGE if they represent DIFFERENT STEPS, even if they seem similar.

Examples of what SHOULD be merged:
- "add the number of boxes and the number of balls" and "sum the values of balls and boxes" (same concept, different words)
- "convert to binary" and "transform to base 2" (same concept, different terminology)

Examples of what should NOT be merged:
- "add numbers" and "multiply numbers" (different operations)
- "find maximum" and "find minimum" (different concepts)
- "sort ascending" and "sort of interesting" (different steps)

Answer with exactly one word: YES or NO.

Cluster 1:
{cluster1_analysis}

Cluster 2:
{cluster2_analysis}"""

        try:
            # Add thread-safe delay
            with self._lock:
                time.sleep(self.request_delay)

            payload = {
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3,
                "provider": {
                    "only": ["openai"],
                },
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                data=json.dumps(payload),
                timeout=120,
            )  # 2 minute timeout

            response.raise_for_status()
            response_data = response.json()

            result = response_data["choices"][0]["message"]["content"].strip().upper()
            return "YES" in result

        except Exception as e:
            print(f"Error in LLM call: {e}")
            return False

    def _merge_fully_connected_components(
        self, clusters: List[Cluster], merge_decisions: List[Tuple[str, str]]
    ) -> List[Cluster]:
        """Merge clusters only if they form fully connected groups (cliques).

        We build an undirected graph from YES-decisions and enumerate maximal cliques.
        Then we greedily select largest cliques first without overlap and merge each
        selected clique into a single cluster. Remaining clusters stay unmerged.
        """
        if not merge_decisions:
            return clusters

        # build adjacency map (undirected)
        adj: Dict[str, set] = {}
        nodes: set = set()
        for a, b in merge_decisions:
            nodes.add(a)
            nodes.add(b)
            if a not in adj:
                adj[a] = set()
            if b not in adj:
                adj[b] = set()
            adj[a].add(b)
            adj[b].add(a)

        # helper: Bron–Kerbosch with pivot to enumerate maximal cliques
        node_list = sorted(nodes)
        nbhd = {v: set(adj.get(v, set())) for v in node_list}

        cliques: List[set] = []

        def bron_kerbosch_pivot(
            R: set,
            P: set,
            X: set,
        ):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(set(R))
                return
            U = P | X
            if U:
                u = max(U, key=lambda v: len(nbhd.get(v, set())))
                it = P - nbhd.get(u, set())
            else:
                it = set(P)
            for v in list(it):
                bron_kerbosch_pivot(R | {v}, P & nbhd.get(v, set()), X & nbhd.get(v, set()))
                P.remove(v)
                X.add(v)

        bron_kerbosch_pivot(set(), set(node_list), set())

        # greedy non-overlapping selection of largest cliques first
        cliques.sort(key=lambda s: -len(s))
        used: set = set()
        selected: List[List[str]] = []
        for clq in cliques:
            if any(v in used for v in clq):
                continue
            selected.append(sorted(list(clq)))
            for v in clq:
                used.add(v)

        # build output clusters: merge selected cliques; leave others as singletons
        merged_clusters: List[Cluster] = []
        used_ids: set = set()

        def find_cluster_by_id(cid: str) -> Cluster | None:
            for c in clusters:
                if c.id == cid:
                    return c
            return None

        # merge each selected clique
        for group in selected:
            main_cluster: Cluster | None = None
            merged_from_ids: List[str] = []
            for cid in group:
                cl = find_cluster_by_id(cid)
                if cl is None:
                    continue
                if main_cluster is None:
                    main_cluster = Cluster(
                        sentences=cl.sentences.copy(),
                        id=cl.id,
                    )
                    merged_from_ids.append(cl.id)
                else:
                    main_cluster.merge_with(cl)
                    merged_from_ids.append(cl.id)
                used_ids.add(cl.id)
            if main_cluster is not None:
                main_cluster.merged_from = merged_from_ids
                merged_clusters.append(main_cluster)

        # add clusters not merged in any clique
        for cl in clusters:
            if cl.id in used_ids:
                continue
            new_cluster = Cluster(sentences=cl.sentences.copy(), id=cl.id)
            new_cluster.merged_from = [cl.id]
            merged_clusters.append(new_cluster)

        return merged_clusters

    def _merge_leiden_components(
        self, clusters: List[Cluster], merge_decisions: List[Tuple[str, str]]
    ) -> List[Cluster]:
        """Merge clusters using Leiden (CPM objective) with self.gamma.

        Vertices are all clusters; undirected edges are LLM YES decisions.
        Communities of size >= 2 are merged; singletons remain.
        """
        if not clusters:
            return []
        # Map cluster IDs to indices
        id_to_idx = {c.id: i for i, c in enumerate(clusters)}
        edges: List[Tuple[int, int]] = []
        for a, b in merge_decisions:
            if a in id_to_idx and b in id_to_idx and a != b:
                ia, ib = id_to_idx[a], id_to_idx[b]
                if ia != ib:
                    edges.append((ia, ib))
        g = ig.Graph(
            len(clusters),
            edges=edges,
            directed=False,
        )
        part = la.find_partition(
            g,
            la.CPMVertexPartition,
            resolution_parameter=self.gamma,
        )
        labels = part.membership
        groups: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(lab, []).append(idx)

        merged: List[Cluster] = []
        used: set = set()
        for lab, idxs in groups.items():
            if len(idxs) >= 2:
                main: Cluster | None = None
                merged_from: List[str] = []
                for i in idxs:
                    cl = clusters[i]
                    if main is None:
                        main = Cluster(
                            sentences=cl.sentences.copy(),
                            id=cl.id,
                        )
                        merged_from.append(cl.id)
                    else:
                        main.merge_with(cl)
                        merged_from.append(cl.id)
                    used.add(cl.id)
                if main is not None:
                    main.merged_from = merged_from
                    merged.append(main)
        for cl in clusters:
            if cl.id in used:
                continue
            new_cl = Cluster(sentences=cl.sentences.copy(), id=cl.id)
            new_cl.merged_from = [cl.id]
            merged.append(new_cl)
        return merged

    def _create_rollout_edges(
        self,
        responses: Dict[str, Any],
        merged_clusters: List[Cluster],
        prompt_index: str,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Create edges for each rollout by tracing its path through the clusters."""
        # mapping from original cluster IDs to new cluster IDs
        old_to_new_mapping = {}
        for i, cluster in enumerate(merged_clusters):
            if hasattr(cluster, "merged_from") and cluster.merged_from:
                for original_cluster_id in cluster.merged_from:
                    old_to_new_mapping[original_cluster_id] = i
            else:
                old_to_new_mapping[cluster.id] = i

        # chunk-to-cluster mapping
        chunk_to_cluster = {}
        sentence_idx = 0

        for seed, response_sentences in self.seed_to_sentences.items():
            for sentence in response_sentences:
                original_cluster_id = self.sentence_clusters[sentence_idx]
                new_cluster_id = old_to_new_mapping.get(
                    str(original_cluster_id), original_cluster_id
                )
                chunk_to_cluster[sentence] = new_cluster_id
                sentence_idx += 1

        # create edges for each rollout
        rollout_edges = {}
        content_key = self._get_content_key()
        for seed, response_data in responses.items():
            edges = []
            content = response_data.get(content_key, None)
            if not content:
                content = response_data.get("cot_content", "")

            if content:
                if isinstance(content, list):
                    chunks = content
                else:
                    chunks = [content]

                # create edges between consecutive chunks
                for i in range(len(chunks) - 1):
                    current_chunk = chunks[i]
                    next_chunk = chunks[i + 1]

                    if current_chunk in chunk_to_cluster and next_chunk in chunk_to_cluster:
                        current_cluster = chunk_to_cluster[current_chunk]
                        next_cluster = chunk_to_cluster[next_chunk]

                        edges.append(
                            {
                                "node_a": f"cluster-{current_cluster}",
                                "node_b": f"cluster-{next_cluster}",
                                "step_text_a": str(current_chunk),
                                "step_text_b": str(next_chunk),
                            }
                        )

                # add final edge from last chunk to response node
                if chunks and seed in responses:
                    last_chunk = chunks[-1]

                    if last_chunk in chunk_to_cluster:
                        last_cluster = chunk_to_cluster[last_chunk]

                        # extract final answer using the same method as _add_response_nodes
                        from src.utils.prompt_utils import get_prompt_filter

                        prompt_filter = get_prompt_filter(prompt_index)
                        if prompt_filter:
                            final_answer = prompt_filter.extract_final_answer(response_data)
                            if final_answer:
                                response_cluster_id = f"response-{final_answer}"

                                edges.append(
                                    {
                                        "node_a": f"cluster-{last_cluster}",
                                        "node_b": response_cluster_id,
                                        "step_text_a": str(last_chunk),
                                        "step_text_b": str(final_answer),
                                    }
                                )

            rollout_edges[seed] = edges

        return rollout_edges

    def _cluster_by_cosine_similarity(
        self,
        embeddings: np.ndarray,
        threshold: float,
    ) -> List[int]:
        """Cluster chunks using either Agglomerative (complete-link) or clique-based approach."""
        num_items = embeddings.shape[0]
        print(f"Clustering {num_items} items with threshold {threshold}...")
        print("Using AgglomerativeClustering (complete-link, precomputed distances)")
        return self._cluster_by_agglomerative(
            embeddings,
            threshold,
        )

    def _cluster_by_agglomerative(
        self,
        embeddings: np.ndarray,
        threshold: float,
    ) -> List[int]:
        # embeddings assumed normalized already at callsite
        S = embeddings @ embeddings.T
        # distance matrix in float16 to reduce memory
        D = np.clip(1.0 - S, 0.0, 2.0).astype(np.float16)
        # ensure exact zeros on diagonal
        np.fill_diagonal(D, np.float16(0.0))
        h = np.float32(1.0 - threshold)
        model = AgglomerativeClustering(
            linkage="complete",
            metric="precomputed",
            distance_threshold=float(h),
            n_clusters=None,
            compute_full_tree="auto",
        )
        labels = model.fit_predict(D)
        return list(map(int, labels))

# Faith Combined Visualization — Issues & Improvements

Issues discovered during interactive exploration of the `faith_combined` flowchart (1,517 nodes, 10,575 edges, 97 rollouts of Qwen3-8B on a carbohydrate/exercise MCQ).

---

## Issue 1: Most reasoning nodes don't connect to answer nodes

**Severity:** High
**Impact:** The graph looks like most reasoning paths are dead ends, when in reality every rollout produces a final answer.

### What's happening

Only ~26 of 97 rollouts have edges connecting their last reasoning cluster to a response node (`response-A`, `response-C`, etc.). The other 71 are disconnected.

### Root cause

The edge from the last reasoning node to the response node is created in `_create_rollout_edges()` at `src/clustering/sentence_then_llm_clusterer.py:1013-1036`. Three conditions must all pass:

```python
# Line 1014-1017
if chunks and seed in responses:
    last_chunk = chunks[-1]
    if last_chunk in chunk_to_cluster:  # <-- THIS FAILS for ~73% of rollouts
```

The `chunk_to_cluster` mapping (lines 967-978) is built from `self.seed_to_sentences` during the clustering pass. But `last_chunk` (line 1015) comes from a **fresh read** of `response_data[content_key]` (line 985). If there's any mismatch — whitespace normalization, different chunking results, string identity vs equality — the lookup fails silently.

When it fails, no edge is created and the rollout appears disconnected from any answer.

### How to verify

Add logging at line 1017 to count how many rollouts fail the `last_chunk in chunk_to_cluster` check:

```python
if last_chunk in chunk_to_cluster:
    ...
else:
    print(f"WARNING: seed {seed} last_chunk not in chunk_to_cluster mapping")
    print(f"  last_chunk: {repr(last_chunk[:80])}")
```

### Proposed fix

Instead of exact string lookup, fall back to finding the closest matching chunk in the mapping (e.g. by stripping whitespace or using embedding similarity). Or ensure the same data source is used for both the mapping construction and the edge creation.

---

## Issue 2: Clustering uses sentence fragments, not full sentences

**Severity:** Medium
**Impact:** Clusters contain fragments like `"then during exercise"` or `"So D says \"greater reliance on muscle glycogen\""` instead of complete reasoning steps. This makes clusters harder to interpret and inflates the node count.

### What's happening

The chunking pipeline (`src/chunking.py`, function `chunk()`) is a 7-stage pipeline designed for math-heavy CoTs:

1. Sentence split (on `.`, `?`, `!`, newlines)
2. Merge math-heavy consecutive chunks
3. **Clause-level split** (on commas, colons, semicolons) — **main culprit**
4. Parenthetical extraction
5. Post-processing merge rules
6. Math coalescing
7. Whitespace normalization

Stage 3 splits on commas/colons whenever both sides have >= 2 alphabetic words. For natural language MCQ reasoning (not math), this over-splits.

Example:
```
Input:  "The body uses glucose during rest, and then during exercise it uses glycogen."
Stage 1: ["The body uses glucose during rest, and then during exercise it uses glycogen."]
Stage 3: ["The body uses glucose during rest,", "and then during exercise it uses glycogen."]
```

### Existing config solution

There's already a flag in `configs/f/combined.yaml` (line 7):

```yaml
sentences_instead_of_chunks: false
```

Setting this to `true` makes the clustering use `split_into_sentences()` output (stored as `"sentences"` in each rollout JSON) instead of `chunk()` output (stored as `"chunked_cot_content"`). Both are always pre-computed during rollout generation (`src/generation/generate_responses.py:252,264`).

The config flag is read by `_get_content_key()` in `src/clustering/sentence_then_llm_clusterer.py:177-182` and `src/clustering/base.py:227-234`.

### How to test

```bash
# 1. Edit configs/f/combined.yaml: sentences_instead_of_chunks: true
# 2. Delete cached flowchart
rm flowcharts/faith_combined/config-faith_combined-combined_qwen3_8b_flowchart.json
# 3. Regenerate
uv run python -m src.main --config-name=faith_combined command=flowcharts
# 4. Recompute layout (delete cache first)
rm -rf graph_layout_service/cache/
uv run python -m src.main --config-name=faith_combined command=graphviz
```

### Trade-off

- **Chunks** (current): finer-grained, better for math-heavy CoTs, more nodes, more fragment noise
- **Sentences** (proposed): coarser, cleaner clusters for natural language reasoning, fewer nodes, may merge distinct steps

---

## Issue 3: Graph layout is unordered (no START-to-answer flow)

**Severity:** Medium
**Impact:** The graph renders as an organic blob (force-directed layout). Hard to see the flow from START through reasoning to answers. Ideally START is on the left and response nodes (`response-A`, `response-C`) are on the right.

### What's happening

The layout uses `sfdp` (scalable force-directed placement) which doesn't respect edge direction. It produces a 2D embedding where proximity = connectivity, but there's no left-to-right flow.

### Proposed fix: 3 changes

**Change 1** — `src/flowchart/graphviz_generator.py` line 111:

```python
# Before
"options": {"engine": "sfdp", "width": 1200, "height": 800, "padding": 20},

# After
"options": {"engine": "dot", "width": 1200, "height": 800, "padding": 20},
```

**Change 2** — `graph_layout_service/app.py` line 103:

```python
# Before
graph = pgv.AGraph(strict=False, directed=False)

# After
graph = pgv.AGraph(strict=False, directed=True)
```

**Change 3** — `graph_layout_service/app.py` line 126:

```python
# Before
graph.graph_attr.update(overlap="true")

# After
graph.graph_attr.update(overlap="true", rankdir="LR")
```

### Caveat

`dot` is designed for DAGs and may struggle with 1,517 nodes and 10,575 edges (cycles, high density). If it's too slow or produces a bad layout, alternatives:

- `dot` with the size threshold cranked up (e.g. min size >= 10, reducing to ~400 nodes)
- Keep `sfdp` but pin START at x=0 and response nodes at x=max in the layout service
- Use `dot` only for the filtered/distilled view

### After changing, regenerate layout

```bash
rm -rf graph_layout_service/cache/
# Restart layout service (kill existing, re-launch)
uv run python -m src.main --config-name=faith_combined command=graphviz
```

---

## Summary

| # | Issue | Severity | Fix complexity | Config-only? |
|---|-------|----------|---------------|--------------|
| 1 | Rollouts disconnected from answer nodes | High | Medium (debug mapping, fix lookup) | No |
| 2 | Sentence fragment clustering | Medium | Trivial (`sentences_instead_of_chunks: true`) | Yes |
| 3 | No left-to-right layout ordering | Medium | Small (3 line changes + cache clear) | No |

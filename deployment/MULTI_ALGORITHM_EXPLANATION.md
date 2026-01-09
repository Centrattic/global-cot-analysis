# Multi-Algorithm Property Checker: Understanding and Implementation

## Overview

The `multi_algorithm` property checker tracks which algorithms are used at different positions within a rollout trajectory. This allows visualization of how different algorithms (e.g., "0", "1", "custom-algorithm") are distributed across clusters in the flowchart.

## Data Structure

The `multi_algorithm` property is stored per rollout and follows this format:

```typescript
// Format: [algorithm_1, cut_1, algorithm_2, cut_2, ...]
// Where algorithm_i is a string (e.g., "0", "1", "custom")
// And cut_i is a number representing the position (1-indexed) where the algorithm changes
// Example: ["0", 3, "1", 7] means:
//   - Algorithm "0" from position 1 to 3 (inclusive)
//   - Algorithm "1" from position 4 to 7 (inclusive)
//   - Algorithm "1" continues to the end if no more cuts
```

## Key Components

### 1. Parsing Multi-Algorithm Data

The parsing happens in multiple places:

#### a) Extracting Unique Algorithms (Lines 1015-1031)
```typescript
getPropertyCheckerUniqueValues(propertyName: string, rollouts: string[])
```
- Extracts all unique algorithm IDs across rollouts
- Parses string or array format
- Returns sorted list of unique algorithm IDs

#### b) Mapping Algorithms to Clusters (Lines 1482-1516)
For each rollout, builds a mapping: `clusterId -> algorithmId`
```typescript
nodeAlgoByRollout: Map<rolloutId, Map<clusterId, algorithmId>>
```

**Process:**
1. Get raw `multi_algorithm` value (may be string, parse to array)
2. Extract algorithm names and cut positions (even indices = algorithms, odd = cuts)
3. Map positions (1-indexed, excluding START) to algorithms:
   - Position 1 to `cuts[0]-1` → `algos[0]`
   - Position `cuts[0]` to `cuts[1]-1` → `algos[1]`
   - etc.
4. Map each cluster in the path to its corresponding algorithm (first occurrence)

**Key code (lines 1487-1515):**
```typescript
const pathForAlgo = dedupedSeq.filter(cid => cid !== 'START')
const L = pathForAlgo.length
const posToAlgo: (string | null)[] = new Array(L).fill(null)

// Parse [algo1, cut1, algo2, cut2, ...] format
for (let i = 0; i < algos.length; i++) {
    const algo = algos[i]
    const endExclusive = (i < cuts.length) ? cuts[i] : (L + 1)
    const end = Math.max(start, Math.min(L, endExclusive - 1))
    if (start <= end) {
        for (let p = start; p <= end; p++) posToAlgo[p - 1] = algo
    }
    start = endExclusive
}

// Map cluster IDs to algorithms (first occurrence only)
pathForAlgo.forEach((cid, idx) => {
    const algo = posToAlgo[idx]
    if (algo && !map.has(cid)) map.set(cid, algo)
})
```

### 2. Tracking Rollouts Through Clusters

The system tracks which rollouts pass through each cluster using `incomingByNode`:

```typescript
incomingByNode: Map<clusterId, Map<rolloutId, count>>
```

**How it's populated:**
- When an edge is drawn from cluster U → V, `addIncoming(V, rolloutId, 1)` is called
- This tracks that the rollout passed through cluster V
- The count represents how many times (typically 1, but could be multiple if cycles aren't collapsed)

**Code (lines 1320-1324, 1659):**
```typescript
const addIncoming = (nodeId: string, rolloutKey: string, weight: number) => {
    if (!incomingByNode.has(nodeId)) incomingByNode.set(nodeId, new Map())
    const m = incomingByNode.get(nodeId)!
    m.set(rolloutKey, (m.get(rolloutKey) || 0) + weight)
}

// Called when drawing edge U → V:
addIncoming(v, String(rolloutId), 1)
```

### 3. Calculating Algorithm Ratios Per Cluster

The algorithm ratios are calculated when rendering pie charts for each cluster node (lines 1999-2020):

**Algorithm:**
1. For each cluster, get `incomingByNode.get(clusterId)` → Map of `rolloutId -> count`
2. For each rollout that passed through the cluster:
   - Get the rollout's count (weight)
   - Look up which algorithm was active at this cluster: `nodeAlgoByRollout.get(rolloutId).get(clusterId)`
   - Aggregate counts by algorithm: `algoTotals[algorithm] += count`
3. Create pie slices with algorithm as key, aggregated count as value

**Code (lines 1999-2020):**
```typescript
if (isMultiAlgorithmActive) {
    const algoTotals = new Map<string, number>()
    newValidRollouts.forEach(rid => {
        const id = String(rid)
        const w = counts.get(id) || 0  // Count of times this rollout passed through cluster
        if (w <= 0) return
        
        const map = nodeAlgoByRollout.get(id)  // Get cluster→algorithm mapping for this rollout
        const algo = map ? map.get(clusterId) : null  // Get algorithm for this specific cluster
        if (!algo) return  // Skip if no algorithm assigned (e.g., after collapsing)
        
        // Aggregate by algorithm
        if (propertyFilter.filterMode !== 'both' && typeof propertyFilter.filterMode === 'number') {
            // Filter mode: only count selected algorithm class
            const classes = getPropertyCheckerUniqueValues('multi_algorithm', newValidRollouts)
            const target = classes[propertyFilter.filterMode as number]
            if (key === target) algoTotals.set(key, (algoTotals.get(key) || 0) + w)
        } else {
            // Both mode: count all algorithms
            algoTotals.set(key, (algoTotals.get(key) || 0) + w)
        }
    })
    // Create pie slices: one slice per algorithm, size = total count
    dataSlices = Array.from(algoTotals.entries()).map(([algo, value]) => ({
        key: algo,
        value,
        color: getAlgorithmColor(algo)
    }))
}
```

### 4. Example Calculation

**Scenario:**
- Cluster "cluster-5" is visited by:
  - Rollout "1": uses algorithm "0", passes through once (count=1)
  - Rollout "2": uses algorithm "0", passes through once (count=1)
  - Rollout "3": uses algorithm "1", passes through once (count=1)
  - Rollout "4": uses algorithm "1", passes through twice (count=2, e.g., due to cycle)

**Calculation:**
```typescript
algoTotals = new Map()
// Rollout "1": algo="0", count=1 → algoTotals["0"] = 1
// Rollout "2": algo="0", count=1 → algoTotals["0"] = 2
// Rollout "3": algo="1", count=1 → algoTotals["1"] = 1
// Rollout "4": algo="1", count=2 → algoTotals["1"] = 3

// Final ratios:
// Algorithm "0": 2 passes (40%)
// Algorithm "1": 4 passes (60%)
```

**Pie Chart:**
- Slice 1: Algorithm "0", size=2, color=blue
- Slice 2: Algorithm "1", size=4, color=red

## Special Cases

### 1. Delta Mode (Lines 1184-1219)
In delta mode, `multi_algorithm` is processed per-position (not per-rollout):
- Builds `nodeSeenByClass` and `edgeSeenByClass` per algorithm
- Shows only exclusive nodes/edges (appear in only one algorithm)

### 2. START Node Edges (Lines 1554-1566)
- In "Both" mode: START edges are colored with the first algorithm's color
- In class-specific mode: Only drawn if first algorithm matches selected class

### 3. Edge Coloring (Lines 1581-1632)
- If edge endpoints use same algorithm: solid color
- If edge crosses algorithm boundary: gradient from algo1 to algo2

### 4. Filtering (Lines 2012-2018)
- If specific algorithm class is selected, only that class is counted in pie charts
- Otherwise, all algorithms are shown

## Key Functions Reference

1. **`getPropertyCheckerUniqueValues()`** (lines 1014-1041)
   - Extracts all unique algorithm IDs across rollouts

2. **`nodeAlgoByRollout`** (lines 1315, 1482-1516)
   - Map: `rolloutId → (clusterId → algorithmId)`
   - Built once per visualization update

3. **`incomingByNode`** (lines 1317, 1320-1324, 1659, 1796)
   - Map: `clusterId → (rolloutId → count)`
   - Populated as edges are drawn

4. **`addIncoming()`** (lines 1320-1324)
   - Increments count for a rollout passing through a cluster

5. **Pie Chart Calculation** (lines 1999-2020)
   - Aggregates `incomingByNode` counts by algorithm using `nodeAlgoByRollout`
   - Creates D3 pie slices with algorithm as key

## Summary

To calculate algorithm ratios for a cluster:

1. **Get all rollouts passing through cluster**: `incomingByNode.get(clusterId)`
2. **For each rollout**: 
   - Get count: `counts.get(rolloutId)`
   - Get algorithm: `nodeAlgoByRollout.get(rolloutId).get(clusterId)`
   - Accumulate: `algoTotals[algorithm] += count`
3. **Calculate ratios**: Each algorithm's total / sum of all totals
4. **Render pie chart**: One slice per algorithm, sized by ratio

The pie chart visually represents the ratio of algorithms passing through each cluster, with colors distinguishing different algorithms.


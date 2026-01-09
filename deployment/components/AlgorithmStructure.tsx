// @ts-nocheck
'use client'

import { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { FlowchartData } from '@/types/flowchart'
import './graph.css'

interface AlgorithmStructureProps {
    data: FlowchartData
    promptIndex?: string
}

interface AlgorithmNode {
    id: string
    description: string
}

interface AlgorithmEdge {
    source: string
    target: string
    weight: number
}

export default function AlgorithmStructure({ data, promptIndex }: AlgorithmStructureProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selectedAlgorithm, setSelectedAlgorithm] = useState<AlgorithmNode | null>(null)
    const [minEdgeWeight, setMinEdgeWeight] = useState<number>(0)
    const [maxEdgeWeight, setMaxEdgeWeight] = useState<number>(0)
    const [algorithms, setAlgorithms] = useState<Map<string, AlgorithmNode>>(new Map())
    const [edges, setEdges] = useState<AlgorithmEdge[]>([])
    const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform | null>(null)

    // Load algorithms from flowchart data, with fallback to API
    useEffect(() => {
        if (!data) return

        const algData = (data as any).algorithms
        const algMap = new Map<string, AlgorithmNode>()

        // First try to load from flowchart data
        if (algData && typeof algData === 'object' && Object.keys(algData).length > 0) {
            // Check if there's a top-level descriptions object
            if (algData.descriptions && typeof algData.descriptions === 'object') {
                Object.entries(algData.descriptions).forEach(([id, desc]) => {
                    if (typeof desc === 'string') {
                        algMap.set(id, { id, description: desc })
                    }
                })
            } else {
                // Legacy format: iterate over algorithm entries
                Object.entries(algData).forEach(([id, algObj]) => {
                    // Handle both string format and object format
                    if (typeof algObj === 'string') {
                        algMap.set(id, { id, description: algObj })
                    } else if (algObj && typeof algObj === 'object') {
                        // Try new format: algObj.descriptions.description
                        if (algObj.descriptions && typeof algObj.descriptions === 'object' && algObj.descriptions.description) {
                            algMap.set(id, { id, description: algObj.descriptions.description })
                        }
                        // Fallback: try direct description property
                        else if (algObj.description && typeof algObj.description === 'string') {
                            algMap.set(id, { id, description: algObj.description })
                        }
                    }
                })
            }

            if (algMap.size > 0) {
                setAlgorithms(algMap)
                return
            }
        }

        // Fallback: fetch from algorithms.json API if algorithms key is missing or empty
        if (promptIndex) {
            fetch(`/api/algorithms/${encodeURIComponent(promptIndex)}`)
                .then(res => res.json())
                .then((apiAlgData: any) => {
                    const apiAlgMap = new Map<string, AlgorithmNode>()
                    if (apiAlgData && typeof apiAlgData === 'object') {
                        Object.entries(apiAlgData).forEach(([id, desc]) => {
                            if (typeof desc === 'string') {
                                apiAlgMap.set(id, { id, description: desc })
                            }
                        })
                    }
                    if (apiAlgMap.size > 0) {
                        setAlgorithms(apiAlgMap)
                    }
                })
                .catch(err => {
                    console.error('Failed to fetch algorithms from API:', err)
                })
        }
    }, [data, promptIndex])

    // Extract algorithm sequences from rollouts and compute edges
    const algorithmSequences = useMemo(() => {
        if (!data) return []

        const responsesData: any = (data as any).responses || (data as any).rollouts || {}

        // Detect data format
        const isOldFormat = Array.isArray(responsesData) && responsesData.length > 0 && responsesData[0].index !== undefined

        const rolloutIds: string[] = Array.isArray(responsesData)
            ? (isOldFormat
                ? responsesData.map((r: any) => String(r.index))
                : responsesData.map((x: any) => Object.keys(x || {})[0]).filter(Boolean))
            : Object.keys(responsesData)

        const sequences: string[][] = []

        rolloutIds.forEach(rid => {
            let rolloutData: any = null

            if (Array.isArray(responsesData)) {
                if (isOldFormat) {
                    rolloutData = responsesData.find((r: any) => r.index && String(r.index) === rid)
                } else {
                    const rolloutObj = responsesData.find((x: any) => x[rid])
                    rolloutData = rolloutObj ? rolloutObj[rid] : null
                }
            } else {
                rolloutData = responsesData[rid]
            }

            if (!rolloutData) return

            // Try to find multi_algorithm property - could be multi_algorithm, debug_multi_algorithm, or any other variant
            let multiAlgo: any = null

            // Check common property names (in order of preference)
            if (rolloutData.multi_algorithm !== undefined) {
                multiAlgo = rolloutData.multi_algorithm
            } else if (rolloutData.debug_multi_algorithm !== undefined) {
                multiAlgo = rolloutData.debug_multi_algorithm
            } else {
                // Check for any property ending in _multi_algorithm
                for (const key in rolloutData) {
                    if (key.endsWith('_multi_algorithm')) {
                        multiAlgo = rolloutData[key]
                        break
                    }
                }
            }

            if (!multiAlgo) return

            if (typeof multiAlgo === 'string') {
                try {
                    multiAlgo = JSON.parse(multiAlgo)
                } catch {
                    return
                }
            }

            const boundaries: number[] = []
            const algorithms: string[] = [initialAlgo]

            // Extract boundaries and subsequent algorithms
            for (let i = 1; i < multiAlgo.length; i += 2) {
                const boundary = multiAlgo[i]
                const algo = multiAlgo[i + 1]

                console.log('[DEBUG AlgorithmStructure] Processing i=', i, 'boundary:', boundary, 'algo:', algo, 'boundary type:', typeof boundary, 'algo type:', typeof algo)

                if (typeof boundary === 'number' && typeof algo === 'string') {
                    boundaries.push(boundary)
                    algorithms.push(algo)
                    if (algo === "0") {
                        console.log('[DEBUG AlgorithmStructure] 🎯 ALGORITHM 0 DETECTED in switch at boundary', boundary, 'for rollout', rid)
                    }
                } else {
                    console.log('[DEBUG AlgorithmStructure] Skipping boundary/algo pair - boundary is number?', typeof boundary === 'number', 'algo is string?', typeof algo === 'string')
                }
            }

            console.log('[DEBUG AlgorithmStructure] Extracted for rollout', rid, '- algorithms:', algorithms, 'boundaries:', boundaries)
            if (algorithms.includes("0")) {
                console.log('[DEBUG AlgorithmStructure] ✅ ALGORITHM 0 IS IN FINAL ALGORITHMS ARRAY for rollout', rid, 'full sequence:', algorithms)
            }

            if (algorithms.length === 0) return

            // Filter switches based on sentence distance (skip for "cubes" prompt)
            let filteredSequence: string[] = []
            console.log('[DEBUG AlgorithmStructure] promptIndex:', promptIndex, 'algorithms:', algorithms, 'boundaries:', boundaries, 'rolloutId:', rid)
            if (promptIndex === "cubes") {
                // For cubes, include all switches without filtering
                console.log('[DEBUG AlgorithmStructure] CUBES PROMPT DETECTED - skipping switch filtering')
                console.log('[DEBUG AlgorithmStructure] Original sequence (cubes):', algorithms)
                filteredSequence = algorithms
                console.log('[DEBUG AlgorithmStructure] Filtered sequence (cubes, no filtering):', filteredSequence)
            } else {
                // For other prompts, filter switches that are too close (< 3 sentences apart)
                if (algorithms.length === 1) {
                    filteredSequence = algorithms
                } else {
                    // Always include the first algorithm
                    filteredSequence.push(algorithms[0])

                    // Track the last included boundary (start at 0 for the initial algorithm)
                    let lastIncludedBoundary = 0

                    // Check each subsequent switch
                    for (let i = 0; i < boundaries.length; i++) {
                        const boundary = boundaries[i]
                        const nextAlgo = algorithms[i + 1]

                        // Calculate distance from last included boundary
                        const distance = boundary - lastIncludedBoundary

                        // Only include switch if distance >= 3 sentences
                        if (distance >= 3) {
                            filteredSequence.push(nextAlgo)
                            lastIncludedBoundary = boundary
                        }
                    }
                }
                console.log('[DEBUG AlgorithmStructure] Original sequence (non-cubes):', algorithms)
                console.log('[DEBUG AlgorithmStructure] Filtered sequence (non-cubes):', filteredSequence)
            }

            if (filteredSequence.length > 0) {
                console.log('[DEBUG AlgorithmStructure] Adding sequence to sequences:', filteredSequence, 'rolloutId:', rid)
                sequences.push(filteredSequence)
            } else {
                console.log('[DEBUG AlgorithmStructure] Filtered sequence is empty, not adding. Original was:', algorithms, 'rolloutId:', rid)
            }
        })

        console.log('[DEBUG AlgorithmStructure] Total rollouts processed:', rolloutIds.length)
        console.log('[DEBUG AlgorithmStructure] Sequences extracted:', sequences.length)
        console.log('[DEBUG AlgorithmStructure] Sample sequences (first 5):', sequences.slice(0, 5))
        const totalTransitions = sequences.reduce((sum, seq) => sum + Math.max(0, seq.length - 1), 0)
        console.log('[DEBUG AlgorithmStructure] Total transitions across all sequences:', totalTransitions)

        return sequences
    }, [data, promptIndex])

    // Compute edges from sequences
    useEffect(() => {
        const edgeMap = new Map<string, number>()

        algorithmSequences.forEach(sequence => {
            for (let i = 0; i < sequence.length - 1; i++) {
                const source = sequence[i]
                const target = sequence[i + 1]
                if (source && target) {
                    const key = `${source}->${target}`
                    edgeMap.set(key, (edgeMap.get(key) || 0) + 1)
                }
            }
        })

        const edgeList: AlgorithmEdge[] = Array.from(edgeMap.entries()).map(([key, weight]) => {
            const [source, target] = key.split('->')
            return { source, target, weight }
        })

        console.log('[DEBUG AlgorithmStructure] Edge weights computed:')
        console.log('[DEBUG AlgorithmStructure] Total edges:', edgeList.length)
        const sortedEdges = edgeList.sort((a, b) => b.weight - a.weight)
        console.log('[DEBUG AlgorithmStructure] All edges (sorted by weight):', sortedEdges)
        console.log('[DEBUG AlgorithmStructure] Top 10 edges:', sortedEdges.slice(0, 10))
        console.log('[DEBUG AlgorithmStructure] Max weight:', edgeList.length > 0 ? Math.max(...edgeList.map(e => e.weight)) : 0)

        // Log algorithm IDs being used
        const allAlgoIds = new Set<string>()
        edgeList.forEach(e => {
            allAlgoIds.add(e.source)
            allAlgoIds.add(e.target)
        })
        console.log('[DEBUG AlgorithmStructure] Algorithm IDs found in edges:', Array.from(allAlgoIds))

        setEdges(edgeList)

        if (edgeList.length > 0) {
            const weights = edgeList.map(e => e.weight)
            const maxWeight = Math.max(...weights)
            setMaxEdgeWeight(maxWeight)
            setMinEdgeWeight(0)
        }
    }, [algorithmSequences])

    // Filter edges by weight and ensure both source and target exist in algorithms
    const filteredEdges = useMemo(() => {
        const filtered = edges.filter(e =>
            e.weight >= minEdgeWeight &&
            algorithms.has(e.source) &&
            algorithms.has(e.target)
        )

        console.log('[DEBUG AlgorithmStructure] Edge filtering:')
        console.log('[DEBUG AlgorithmStructure] Total edges before filter:', edges.length)
        console.log('[DEBUG AlgorithmStructure] Min edge weight filter:', minEdgeWeight)
        console.log('[DEBUG AlgorithmStructure] Algorithms set size:', algorithms.size)
        console.log('[DEBUG AlgorithmStructure] Algorithm IDs in set:', Array.from(algorithms.values()).map(a => a.id))
        console.log('[DEBUG AlgorithmStructure] Edges after filter:', filtered.length)
        console.log('[DEBUG AlgorithmStructure] Filtered edges details:', filtered)

        // Check which edges were filtered out and why
        edges.forEach(e => {
            if (!filtered.includes(e)) {
                const reasons = []
                if (e.weight < minEdgeWeight) reasons.push(`weight ${e.weight} < ${minEdgeWeight}`)
                if (!algorithms.has(e.source)) reasons.push(`source "${e.source}" not in algorithms`)
                if (!algorithms.has(e.target)) reasons.push(`target "${e.target}" not in algorithms`)
                console.log('[DEBUG AlgorithmStructure] Edge filtered out:', `${e.source}->${e.target} (weight: ${e.weight})`, 'reasons:', reasons)
            }
        })

        return filtered
    }, [edges, minEdgeWeight, algorithms])

    // Draw visualization
    useEffect(() => {
        if (!svgRef.current || algorithms.size === 0) return

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        const width = 1200
        const height = 900
        svg.attr('width', width).attr('height', height)

        const g = svg.append('g')

        // Create circular layout for algorithm nodes
        const radius = Math.min(width, height) * 0.3
        const centerX = width / 2
        const centerY = height / 2

        const algorithmList = Array.from(algorithms.values())
        const angleStep = (2 * Math.PI) / algorithmList.length

        const nodePositions = new Map<string, { x: number; y: number }>()
        algorithmList.forEach((alg, index) => {
            const angle = index * angleStep - Math.PI / 2
            const x = centerX + radius * Math.cos(angle)
            const y = centerY + radius * Math.sin(angle)

            // Validate position is not NaN
            if (isNaN(x) || isNaN(y)) {
                console.warn('[DEBUG AlgorithmStructure] Invalid position calculated for algorithm:', alg.id, 'x:', x, 'y:', y)
                return
            }

            nodePositions.set(alg.id, { x, y })
        })

        // Verify all edges reference nodes with positions
        filteredEdges.forEach(edge => {
            if (!nodePositions.has(edge.source)) {
                console.warn('[DEBUG AlgorithmStructure] Missing position for source node:', edge.source)
            }
            if (!nodePositions.has(edge.target)) {
                console.warn('[DEBUG AlgorithmStructure] Missing position for target node:', edge.target)
            }
        })

        // Setup zoom
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 5])
            .on('zoom', (event) => {
                g.attr('transform', event.transform)
                setCurrentTransform(event.transform)
            })

        svg.call(zoom)
        if (currentTransform) {
            svg.call(zoom.transform, currentTransform)
        }

        // No longer need marker definitions - arrows are drawn at midpoints

        // Check for bidirectional edges
        const bidirectionalEdges = new Set<string>()
        filteredEdges.forEach(e1 => {
            filteredEdges.forEach(e2 => {
                if (e1.source === e2.target && e1.target === e2.source) {
                    const key1 = `${e1.source}->${e1.target}`
                    const key2 = `${e2.source}->${e2.target}`
                    bidirectionalEdges.add(key1)
                    bidirectionalEdges.add(key2)
                }
            })
        })

        // Draw edges
        const edgeGroup = g.append('g').attr('class', 'edges')
        filteredEdges.forEach(edge => {
            const sourcePos = nodePositions.get(edge.source)
            const targetPos = nodePositions.get(edge.target)
            if (!sourcePos || !targetPos) return

            // Validate positions are valid numbers
            if (isNaN(sourcePos.x) || isNaN(sourcePos.y) || isNaN(targetPos.x) || isNaN(targetPos.y)) {
                console.warn('[DEBUG AlgorithmStructure] Invalid node positions for edge:', edge.source, '->', edge.target, sourcePos, targetPos)
                return
            }

            const edgeKey = `${edge.source}->${edge.target}`
            const isBidirectional = bidirectionalEdges.has(edgeKey)
            const isSelfLoop = edge.source === edge.target

            let pathData: string

            if (isSelfLoop) {
                // Render self-loop as an elliptical arc above the node
                const nodeRadius = 25
                const loopRadiusX = 35
                const loopRadiusY = 30

                // Start from the top-left of the node, curve up and around clockwise, end at top-right
                // Using elliptical arc (A command in SVG)
                const startX = sourcePos.x - nodeRadius * 0.7
                const startY = sourcePos.y - nodeRadius * 0.7
                const endX = sourcePos.x + nodeRadius * 0.7
                const endY = sourcePos.y - nodeRadius * 0.7

                // Create an elliptical arc that loops above the node
                // A rx ry x-axis-rotation large-arc-flag sweep-flag x y
                // We'll use a cubic bezier for smoother control
                const controlX1 = sourcePos.x - loopRadiusX
                const controlY1 = sourcePos.y - nodeRadius - loopRadiusY
                const controlX2 = sourcePos.x + loopRadiusX
                const controlY2 = sourcePos.y - nodeRadius - loopRadiusY

                // Cubic bezier curve for smooth loop
                pathData = `M ${startX} ${startY} C ${controlX1} ${controlY1}, ${controlX2} ${controlY2}, ${endX} ${endY}`
            } else if (isBidirectional) {
                // Create curved path for bidirectional edges
                // Use a consistent reference direction (from lower ID to higher ID)
                const sourceId = edge.source
                const targetId = edge.target
                const isLexicographicallyFirst = sourceId < targetId

                // Always calculate perpendicular based on direction from lower to higher ID
                // This ensures consistent perpendicular direction for both edges
                let refSourcePos: { x: number; y: number }
                let refTargetPos: { x: number; y: number }

                if (isLexicographicallyFirst) {
                    refSourcePos = sourcePos
                    refTargetPos = targetPos
                } else {
                    refSourcePos = targetPos
                    refTargetPos = sourcePos
                }

                const dx = refTargetPos.x - refSourcePos.x
                const dy = refTargetPos.y - refSourcePos.y
                const dist = Math.sqrt(dx * dx + dy * dy)

                // Guard against zero distance or NaN
                if (dist < 0.001 || isNaN(dist)) {
                    // Fallback to straight line if nodes are too close
                    pathData = `M ${sourcePos.x} ${sourcePos.y} L ${targetPos.x} ${targetPos.y}`
                } else {
                    // Calculate perpendicular direction (consistent for both edges)
                    const perpX = -dy / dist
                    const perpY = dx / dist
                    const curveOffset = 30

                    // For the edge going from lower to higher ID, curve in positive direction
                    // For the edge going from higher to lower ID, curve in negative direction
                    const offset = isLexicographicallyFirst ? curveOffset : -curveOffset

                    // Control point is offset perpendicular to the line connecting nodes
                    const midX = (sourcePos.x + targetPos.x) / 2
                    const midY = (sourcePos.y + targetPos.y) / 2
                    const controlX = midX + perpX * offset
                    const controlY = midY + perpY * offset

                    // Guard against NaN in control point
                    if (isNaN(controlX) || isNaN(controlY)) {
                        pathData = `M ${sourcePos.x} ${sourcePos.y} L ${targetPos.x} ${targetPos.y}`
                    } else {
                        // Quadratic bezier curve
                        pathData = `M ${sourcePos.x} ${sourcePos.y} Q ${controlX} ${controlY} ${targetPos.x} ${targetPos.y}`
                    }
                }
            } else {
                // Straight line for unidirectional edges
                pathData = `M ${sourcePos.x} ${sourcePos.y} L ${targetPos.x} ${targetPos.y}`
            }

            // Calculate stroke width - thinner for self-loops
            const strokeWidth = isSelfLoop
                ? Math.max(1, Math.min(2.5, edge.weight * 0.15))
                : Math.max(1, Math.min(3.5, edge.weight * 0.25))

            const path = edgeGroup.append('path')
                .attr('d', pathData)
                .attr('fill', 'none')
                .attr('stroke', '#94a3b8')
                .attr('stroke-width', strokeWidth)
                .attr('opacity', 0.7)
                .attr('vector-effect', 'non-scaling-stroke')
                .attr('class', `edge edge-${edge.source}-${edge.target}`)
                .style('cursor', 'pointer')

            // Calculate midpoint and tangent for arrow placement
            const pathElement = path.node() as SVGPathElement
            const pathLength = pathElement.getTotalLength()
            const midLength = pathLength / 2
            const midpoint = pathElement.getPointAtLength(midLength)

            // Calculate tangent direction by sampling nearby points along the path
            const epsilon = Math.min(1, pathLength / 10)
            const beforePoint = pathElement.getPointAtLength(Math.max(0, midLength - epsilon))
            const afterPoint = pathElement.getPointAtLength(Math.min(pathLength, midLength + epsilon))

            // Direction vector along the path
            const dx = afterPoint.x - beforePoint.x
            const dy = afterPoint.y - beforePoint.y
            const angleRad = Math.atan2(dy, dx)

            // Draw arrow at midpoint pointing in the direction of the path
            const arrowSize = 10
            const arrowWidth = 12

            // Arrow tip at midpoint
            const tipX = midpoint.x
            const tipY = midpoint.y

            // Arrow base (behind the tip)
            const baseX = tipX - arrowSize * Math.cos(angleRad)
            const baseY = tipY - arrowSize * Math.sin(angleRad)

            // Perpendicular direction for arrow base width
            const perpX = -Math.sin(angleRad)
            const perpY = Math.cos(angleRad)

            // Left and right points of arrow base
            const leftX = baseX + perpX * arrowWidth / 2
            const leftY = baseY + perpY * arrowWidth / 2
            const rightX = baseX - perpX * arrowWidth / 2
            const rightY = baseY - perpY * arrowWidth / 2

            edgeGroup.append('polygon')
                .attr('points', `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`)
                .attr('fill', '#94a3b8')
                .attr('stroke', '#94a3b8')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.7)

            // Calculate position for edge label (on curve if bidirectional or self-loop)
            // Position labels closer to edge but still slightly off-center toward target
            let labelX: number, labelY: number
            if (isSelfLoop) {
                // Position label above the loop
                const nodeRadius = 25
                const loopRadius = 30
                labelX = sourcePos.x
                labelY = sourcePos.y - nodeRadius - loopRadius - 12
            } else if (isBidirectional) {
                // Use same reference calculation as path for consistency
                const sourceId = edge.source
                const targetId = edge.target
                const isLexicographicallyFirst = sourceId < targetId

                let refSourcePos: { x: number; y: number }
                let refTargetPos: { x: number; y: number }

                if (isLexicographicallyFirst) {
                    refSourcePos = sourcePos
                    refTargetPos = targetPos
                } else {
                    refSourcePos = targetPos
                    refTargetPos = sourcePos
                }

                const dx = refTargetPos.x - refSourcePos.x
                const dy = refTargetPos.y - refSourcePos.y
                const dist = Math.sqrt(dx * dx + dy * dy)

                // Guard against zero distance or NaN
                if (dist < 0.001 || isNaN(dist)) {
                    // Offset slightly toward target
                    labelX = (sourcePos.x + targetPos.x) / 2 + dx * 0.12
                    labelY = (sourcePos.y + targetPos.y) / 2 + dy * 0.12
                } else {
                    const perpX = -dy / dist
                    const perpY = dx / dist
                    // Reduced perpendicular offset and add slight offset toward target
                    const perpOffset = isLexicographicallyFirst ? 32 : -32
                    const forwardOffset = 0.12 // Offset toward target (12% of distance)
                    labelX = (sourcePos.x + targetPos.x) / 2 + perpX * perpOffset + dx * forwardOffset
                    labelY = (sourcePos.y + targetPos.y) / 2 + perpY * perpOffset + dy * forwardOffset

                    // Final guard against NaN
                    if (isNaN(labelX) || isNaN(labelY)) {
                        labelX = (sourcePos.x + targetPos.x) / 2 + dx * 0.12
                        labelY = (sourcePos.y + targetPos.y) / 2 + dy * 0.12
                    }
                }
            } else {
                // For unidirectional edges, offset perpendicular and slightly toward target
                const dx = targetPos.x - sourcePos.x
                const dy = targetPos.y - sourcePos.y
                const dist = Math.sqrt(dx * dx + dy * dy)

                if (dist < 0.001 || isNaN(dist)) {
                    labelX = (sourcePos.x + targetPos.x) / 2
                    labelY = (sourcePos.y + targetPos.y) / 2
                } else {
                    // Perpendicular direction (rotated 90 degrees)
                    const perpX = -dy / dist
                    const perpY = dx / dist
                    // Reduced perpendicular offset and forward (toward target)
                    const perpOffset = 28
                    const forwardOffset = 0.12
                    labelX = (sourcePos.x + targetPos.x) / 2 + perpX * perpOffset + dx * forwardOffset
                    labelY = (sourcePos.y + targetPos.y) / 2 + perpY * perpOffset + dy * forwardOffset
                }
            }

            // Add edge label with weight
            edgeGroup.append('text')
                .attr('x', labelX)
                .attr('y', labelY)
                .attr('text-anchor', 'middle')
                .attr('font-size', '10px')
                .attr('fill', '#64748b')
                .attr('pointer-events', 'none')
                .text(edge.weight)
                .attr('class', 'edge-label')
        })

        // Draw nodes
        const nodeGroup = g.append('g').attr('class', 'nodes')
        algorithmList.forEach(alg => {
            const pos = nodePositions.get(alg.id)
            if (!pos) return

            const nodeGroupItem = nodeGroup.append('g')
                .attr('class', `node node-${alg.id}`)
                .attr('transform', `translate(${pos.x}, ${pos.y})`)
                .style('cursor', 'pointer')

            nodeGroupItem.append('circle')
                .attr('r', 25)
                .attr('fill', '#3b82f6')
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 3)
                .attr('opacity', 0.9)

            nodeGroupItem.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '0.35em')
                .attr('font-size', '14px')
                .attr('font-weight', '600')
                .attr('fill', '#ffffff')
                .text(alg.id)

            nodeGroupItem.on('click', () => {
                setSelectedAlgorithm(alg)
            })

            nodeGroupItem.on('mouseenter', function () {
                d3.select(this).select('circle')
                    .attr('r', 30)
                    .attr('fill', '#2563eb')
            })

            nodeGroupItem.on('mouseleave', function () {
                d3.select(this).select('circle')
                    .attr('r', 25)
                    .attr('fill', '#3b82f6')
            })
        })

        // Apply initial transform
        if (currentTransform) {
            g.attr('transform', currentTransform.toString())
        }

    }, [algorithms, filteredEdges])

    if (algorithms.size === 0) {
        return (
            <div className="emptyState">
                <div className="emptyStateText">
                    Loading algorithms...
                </div>
            </div>
        )
    }

    return (
        <div className="container">
            {/* Controls */}
            <div className="controls">
                <div className="controlsTitle">
                    Edge Weight Filter
                </div>
                <div className="sliderContainer">
                    <input
                        type="range"
                        min={0}
                        max={maxEdgeWeight}
                        value={minEdgeWeight}
                        onChange={(e) => setMinEdgeWeight(parseInt(e.target.value))}
                        className="slider"
                    />
                </div>
                <div className="sliderInfo">
                    Min weight: {minEdgeWeight} | Max: {maxEdgeWeight} | {filteredEdges.length} edges shown
                </div>
                <div className="checkboxContainer" style={{ marginTop: 16 }}>
                    <div className="controlsTitle">
                        Algorithms ({algorithms.size})
                    </div>
                    {Array.from(algorithms.values()).map(alg => (
                        <div key={alg.id} className="checkboxLabel" style={{ marginTop: 8 }}>
                            <div style={{
                                width: 12,
                                height: 12,
                                borderRadius: '50%',
                                backgroundColor: '#3b82f6',
                                marginRight: 8
                            }} />
                            <span>{alg.id}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Main visualization */}
            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

            {/* Algorithm Description Modal */}
            {selectedAlgorithm && (
                <div className="nodeModal">
                    <div className="modalHeader">
                        <h3 className="modalTitle">
                            Algorithm {selectedAlgorithm.id}
                        </h3>
                        <button
                            onClick={() => setSelectedAlgorithm(null)}
                            className="modalCloseButton"
                        >
                            ×
                        </button>
                    </div>
                    <div className="modalContent">
                        <div className="modalText">
                            {selectedAlgorithm.description}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}


// @ts-nocheck
'use client'

import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { FlowchartData, Node } from '@/types/flowchart'

interface GraphizVisualizationProps {
    data: FlowchartData
    selectedRollouts: string[]
    datasetId?: string
}

export default function GraphizVisualization({ data, selectedRollouts, datasetId }: GraphizVisualizationProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selectedNode, setSelectedNode] = useState<Node | null>(null)
    const [minClusterSize, setMinClusterSize] = useState<number>(1)
    const [hoveredRollout, setHoveredRollout] = useState<string | null>(null)
    const [validRollouts, setValidRollouts] = useState<string[]>([])
    const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform | null>(null)
    const positionsRef = useRef<Map<string, { x: number; y: number }>>(new Map())
    const [positionsReady, setPositionsReady] = useState<number>(0)
    const [layoutError, setLayoutError] = useState<string | null>(null)
    const globalEdgeCountRef = useRef<Map<string, number>>(new Map())
    const rolloutEdgeCountRef = useRef<Map<string, Map<string, number>>>(new Map())
    const [panelRollout, setPanelRollout] = useState<string | null>(null)
    const [highlightClusterId, setHighlightClusterId] = useState<string | null>(null)
    const rolloutPanelRef = useRef<HTMLDivElement | null>(null)
    const [rolloutColorMap, setRolloutColorMap] = useState<Map<string, string>>(new Map())
    const [showCorrect, setShowCorrect] = useState<boolean>(true)
    const [showIncorrect, setShowIncorrect] = useState<boolean>(true)
    const [directedEdges, setDirectedEdges] = useState<boolean>(false)

    const LAYOUT_ENDPOINT = '/api/graph/layout'

    const getCanvasSize = () => {
        const isLarge = (datasetId && datasetId.includes('2000')) || (data?.nodes?.length || 0) > 800
        return isLarge ? { width: 2200, height: 1600, padding: 40 } : { width: 1200, height: 800, padding: 20 }
    }

    const isResponseNodeId = (id: string) => typeof id === 'string' && id.startsWith('response-')

    const isBalanced19v20 = !!(datasetId && datasetId.includes('19_vs_20_balanced'))
    const getRolloutCorrectness = (rid: string): boolean | undefined => {
        if (Array.isArray(data.rollouts)) {
            const r = data.rollouts.find((x: any) => x.index.toString() === rid)
            return r?.correctness
        } else {
            const rv = (data.rollouts as any)[rid]
            return rv ? rv.correctness : undefined
        }
    }

    // Calculate cluster sizes and filter
    const getClusterSizes = () => {
        const sizes = data.nodes.map(node => node.freq)
        return {
            min: Math.min(...sizes),
            max: Math.max(...sizes),
            sizes
        }
    }

    const { min: minSize, max: maxSize } = getClusterSizes()

    // Reset minClusterSize when data changes to ensure we show all clusters by default
    useEffect(() => {
        setMinClusterSize(minSize)
    }, [data])

    // Filter nodes by size, but never filter out response nodes
    const filteredNodes = data.nodes.filter(node => node.freq >= minClusterSize || isResponseNodeId(node.cluster_id))
    const getCompressedSequenceForRollout = (rolloutId: string) => {
        let rolloutEdges: { node_a: string; node_b: string }[] = []
        if (Array.isArray(data.rollouts)) {
            const r = data.rollouts.find((x: any) => x.index.toString() === rolloutId)
            rolloutEdges = r?.edges || []
        } else {
            const rv = (data.rollouts as any)[rolloutId]
            if (rv && typeof rv === 'object' && 'edges' in rv) {
                rolloutEdges = (rv as any).edges || []
            } else if (Array.isArray(rv)) {
                rolloutEdges = rv as any
            }
        }
        const seq: string[] = []
        rolloutEdges.forEach((e, i) => {
            if (i === 0) seq.push(e.node_a)
            seq.push(e.node_b)
        })
        const displayedIds = new Set(filteredNodes.map(n => n.cluster_id))
        const filteredSeq = seq.filter(id => displayedIds.has(id))
        // Dedupe for panel display to avoid duplicate rows when toggling filter
        const seen = new Set<string>()
        const unique: string[] = []
        filteredSeq.forEach(id => {
            if (!seen.has(id)) {
                seen.add(id)
                unique.push(id)
            }
        })
        return unique
    }


    // Color palettes
    const rolloutColors = [
        '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
        '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
    ]
    const greenShades = [
        '#14532d', '#166534', '#15803d', '#059669', '#10b981',
        '#16a34a', '#22c55e', '#34d399', '#4ade80', '#86efac'
    ]
    const redShades = [
        '#b91c1c', '#dc2626', '#ef4444', '#f87171', '#fb7185',
        '#fca5a5', '#991b1b', '#7f1d1d', '#e11d48', '#f43f5e'
    ]

    // Better scaling function for cluster radius (logarithmic with minimum size)
    const getClusterRadius = (freq: number) => {
        const minRadius = 4
        const maxRadius = 30
        const scaled = Math.log(freq + 1) * 4
        return Math.max(minRadius, Math.min(maxRadius, scaled))
    }

    // Fetch layout once per dataset (full-graph embedding), cache in positionsRef
    useEffect(() => {
        if (!svgRef.current) return
        positionsRef.current.clear()
        globalEdgeCountRef.current.clear()
        rolloutEdgeCountRef.current.clear()

        const { width, height, padding } = getCanvasSize()

        const nodesPayload = data.nodes.map(n => ({ id: String(n.cluster_id), freq: n.freq }))
        // include START node in embedding
        nodesPayload.push({ id: 'START', freq: 0 })
        const edgesPayload: { source: string; target: string }[] = []
        const edgeKeys = new Set<string>()
        const validNodeIds = new Set<string>(nodesPayload.map(n => n.id))
        const addEdge = (aRaw: string, bRaw: string) => {
            const a = String(aRaw)
            const b = String(bRaw)
            if (!validNodeIds.has(a) || !validNodeIds.has(b)) return
            const key = a < b ? `${a}|${b}` : `${b}|${a}`
            if (!edgeKeys.has(key)) {
                edgeKeys.add(key)
                edgesPayload.push({ source: a, target: b })
            }
        }
        const incrDirGlobal = (a: string, b: string) => {
            const k = `${a}->${b}`
            const m = globalEdgeCountRef.current
            m.set(k, (m.get(k) || 0) + 1)
        }
        const incrDirRollout = (rolloutId: string, a: string, b: string) => {
            const k = `${a}->${b}`
            const m = rolloutEdgeCountRef.current
            if (!m.has(rolloutId)) m.set(rolloutId, new Map())
            const inner = m.get(rolloutId)!
            inner.set(k, (inner.get(k) || 0) + 1)
        }
        if (Array.isArray(data.rollouts)) {
            data.rollouts.forEach((r: any) => {
                const rEdges = r?.edges || []
                rEdges.forEach((e: any) => {
                    addEdge(e.node_a, e.node_b)
                    incrDirGlobal(e.node_a, e.node_b)
                    incrDirRollout(String(r.index), e.node_a, e.node_b)
                })
                // connect START -> first node of this rollout by order
                if (rEdges.length > 0) {
                    const firstA = rEdges[0].node_a
                    addEdge('START', firstA)
                    incrDirGlobal('START', firstA)
                    incrDirRollout(String(r.index), 'START', firstA)
                }
            })
        } else {
            Object.values(data.rollouts).forEach((rv: any, idx) => {
                if (rv && typeof rv === 'object' && 'edges' in rv) {
                    const rEdges = (rv as any).edges || []
                    rEdges.forEach((e: any) => {
                        addEdge(e.node_a, e.node_b)
                        incrDirGlobal(e.node_a, e.node_b)
                        incrDirRollout(String(idx), e.node_a, e.node_b)
                    })
                    if (rEdges.length > 0) {
                        const firstA = rEdges[0].node_a
                        addEdge('START', firstA)
                        incrDirGlobal('START', firstA)
                        incrDirRollout(String(idx), 'START', firstA)
                    }
                } else if (Array.isArray(rv)) {
                    const rEdges = rv as any[]
                    rEdges.forEach((e: any) => {
                        addEdge(e.node_a, e.node_b)
                        incrDirGlobal(e.node_a, e.node_b)
                        incrDirRollout(String(idx), e.node_a, e.node_b)
                    })
                    if (rEdges.length > 0) {
                        const firstA = rEdges[0].node_a
                        addEdge('START', firstA)
                        incrDirGlobal('START', firstA)
                        incrDirRollout(String(idx), 'START', firstA)
                    }
                }
            })
        }

        const body = JSON.stringify({
            dataset_id: datasetId || (data as any).dataset_id || 'default',
            nodes: nodesPayload,
            edges: edgesPayload,
            options: { engine: 'sfdp', width, height, padding }
        })

        setLayoutError(null)
        fetch(LAYOUT_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body
        }).then(res => res.json()).then(layout => {
            const positions = layout.positions as Record<string, { x: number; y: number }>
            const map = positionsRef.current
            Object.entries(positions).forEach(([id, pos]) => {
                map.set(id, pos)
            })
            setPositionsReady(v => v + 1)
        }).catch(err => {
            setLayoutError('Layout request failed')
            setPositionsReady(v => v + 1)
        })
    }, [data])

    // Draw/update visualization using cached positions; do not refetch on hover
    useEffect(() => {
        if (!svgRef.current) return

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        // Initialize collections
        const edges: { node_a: string; node_b: string }[] = []
        const rolloutPaths = new Map<string, string[]>()
        const nodePositions = positionsRef.current

        // Build paths for each rollout
        const newValidRollouts: string[] = []
        selectedRollouts.forEach((rolloutId) => {
            const passesCorrectness = (rid: string) => {
                if (!isBalanced19v20) return true
                if (showCorrect && showIncorrect) return true
                const corr = getRolloutCorrectness(rid)
                if (showCorrect && !showIncorrect) return corr === true
                if (!showCorrect && showIncorrect) return corr === false
                return false
            }
            if (!passesCorrectness(rolloutId)) return
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (Array.isArray(data.rollouts)) {
                const rollout = data.rollouts.find((r: any) => r.index.toString() === rolloutId)
                rolloutEdges = rollout?.edges || []
            } else {
                const rolloutData = data.rollouts[rolloutId]
                if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                    rolloutEdges = (rolloutData as any).edges || []
                } else if (Array.isArray(rolloutData)) {
                    rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
                }
            }

            if (rolloutEdges.length > 0) {
                newValidRollouts.push(rolloutId)
                edges.push(...rolloutEdges)

                const path = new Set<string>()
                rolloutEdges.forEach(edge => {
                    path.add(edge.node_a)
                    path.add(edge.node_b)
                })
                rolloutPaths.set(rolloutId, Array.from(path))
            }
        })

        // Calculate dimensions
        const { width, height } = getCanvasSize()

        svg.attr('width', width).attr('height', height)

        const g = svg.append('g')

        const applyNodeScale = (k: number) => {
            const kSafe = k > 0 ? k : 1
            const nodesSel = g.selectAll<SVGGElement, any>('.clusters .cluster')
            nodesSel.attr('transform', (d: any) => {
                const p = nodePositions.get(d.cluster_id)
                const tx = p ? p.x : 0
                const ty = p ? p.y : 0
                return `translate(${tx}, ${ty}) scale(${1 / kSafe})`
            })
            const activeSel = g.selectAll<SVGGElement, any>('.clusters-active .cluster')
            activeSel.attr('transform', (d: any) => {
                const p = nodePositions.get(d.cluster_id)
                const tx = p ? p.x : 0
                const ty = p ? p.y : 0
                return `translate(${tx}, ${ty}) scale(${1 / kSafe})`
            })
        }

        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                g.attr('transform', event.transform)
                applyNodeScale(event.transform.k)
                setCurrentTransform(event.transform)
            })

        svg.call(zoom)
        if (currentTransform) {
            g.attr('transform', currentTransform.toString())
            svg.call(zoom.transform, currentTransform)
        }

        if (layoutError) {
            svg.append('text')
                .attr('x', 400)
                .attr('y', 200)
                .attr('text-anchor', 'middle')
                .attr('font-size', '18px')
                .attr('fill', '#ef4444')
                .text(layoutError)
            return
        }

        if (edges.length === 0 || nodePositions.size === 0) {
            svg.append('text')
                .attr('x', 400)
                .attr('y', 200)
                .attr('text-anchor', 'middle')
                .attr('font-size', '18px')
                .attr('fill', '#6b7280')
                .text(nodePositions.size === 0 ? 'Computing layout…' : 'No edges found for selected rollouts')
            return
        }

        setValidRollouts(newValidRollouts)

        const nodeMap = new Map<string, Node>()
        data.nodes.forEach(node => {
            nodeMap.set(node.cluster_id, node)
        })

        const defs = svg.append('defs')
        const defaultColorMap = new Map<string, string>()
        newValidRollouts.forEach((rid, idx) => defaultColorMap.set(rid, rolloutColors[idx % rolloutColors.length]))
        const correctIds = newValidRollouts.filter(rid => getRolloutCorrectness(rid) === true)
        const incorrectIds = newValidRollouts.filter(rid => getRolloutCorrectness(rid) === false)
        const colorForRollout = (rid: string, fallbackIndex?: number) => {
            if (isBalanced19v20) {
                const corr = getRolloutCorrectness(rid)
                if (corr === true) {
                    const idx = correctIds.indexOf(rid)
                    return greenShades[(idx >= 0 ? idx : 0) % greenShades.length]
                }
                if (corr === false) {
                    const idx = incorrectIds.indexOf(rid)
                    return redShades[(idx >= 0 ? idx : 0) % redShades.length]
                }
            }
            if (typeof fallbackIndex === 'number') return rolloutColors[fallbackIndex % rolloutColors.length]
            return defaultColorMap.get(rid) || rolloutColors[0]
        }

        // Build and persist color map for legend and pies to stay consistent
        const newColorMap = new Map<string, string>()
        newValidRollouts.forEach((rid, idx) => {
            newColorMap.set(rid, colorForRollout(rid, idx))
        })
        setRolloutColorMap(newColorMap)

        newValidRollouts.forEach((rolloutId, index) => {
            const gradient = defs.append('radialGradient')
                .attr('id', `gradient-${rolloutId}`)
                .attr('cx', '30%')
                .attr('cy', '30%')

            gradient.append('stop')
                .attr('offset', '0%')
                .attr('stop-color', colorForRollout(rolloutId, index))
                .attr('stop-opacity', 0.8)

            gradient.append('stop')
                .attr('offset', '100%')
                .attr('stop-color', colorForRollout(rolloutId, index))
                .attr('stop-opacity', 0.3)

            // Arrow marker for directed mode (color per rollout)
            const marker = defs.append('marker')
                .attr('id', `arrow-${rolloutId}`)
                .attr('viewBox', '0 0 10 10')
                .attr('refX', 5)
                .attr('refY', 5)
                .attr('markerWidth', 3)
                .attr('markerHeight', 3)
                .attr('orient', 'auto')
            marker.append('path')
                .attr('d', 'M 0 0 L 10 5 L 0 10 z')
                .attr('fill', colorForRollout(rolloutId, index))
        })

        const trajectoryGroup = g.append('g').attr('class', 'trajectories')
        const displayedPathNodeSet = new Set<string>()
        const incomingByNode = new Map<string, Map<string, number>>()
        const addIncoming = (nodeId: string, rolloutKey: string, weight: number) => {
            if (!incomingByNode.has(nodeId)) incomingByNode.set(nodeId, new Map())
            const m = incomingByNode.get(nodeId)!
            m.set(rolloutKey, (m.get(rolloutKey) || 0) + weight)
        }
        newValidRollouts.forEach((rolloutId, index) => {
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (Array.isArray(data.rollouts)) {
                const rollout = data.rollouts.find((r: any) => r.index.toString() === rolloutId)
                rolloutEdges = rollout?.edges || []
            } else {
                const rolloutData = data.rollouts[rolloutId]
                if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                    rolloutEdges = (rolloutData as any).edges || []
                } else if (Array.isArray(rolloutData)) {
                    rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
                }
            }

            // Build ordered path from full edge list, then compress by filter
            const nextMap = new Map<string, string[]>()
            const edgeOrderIndex = new Map<string, number>()
            rolloutEdges.forEach((e, i) => {
                if (!nextMap.has(e.node_a)) nextMap.set(e.node_a, [])
                nextMap.get(e.node_a)!.push(e.node_b)
                edgeOrderIndex.set(`${e.node_a}->${e.node_b}`, i)
                addIncoming(e.node_b, String(rolloutId), 1)
            })
            const incomingFull = new Set<string>()
            rolloutEdges.forEach(e => incomingFull.add(e.node_b))
            const allSourcesFull = new Set(rolloutEdges.map(e => e.node_a))
            let startNodeId = Array.from(allSourcesFull).find(a => !incomingFull.has(a)) || (rolloutEdges[0] ? rolloutEdges[0].node_a : undefined)

            // Build sequence and draw paths
            const fullSequence: string[] = []
            rolloutEdges.forEach((e, i) => {
                if (i === 0) fullSequence.push(e.node_a)
                fullSequence.push(e.node_b)
            })

            const displayedIds = new Set(filteredNodes.map(n => n.cluster_id))
            const compressedSeq = fullSequence.filter(id => displayedIds.has(id))
            compressedSeq.forEach(id => displayedPathNodeSet.add(id))

            if (!directedEdges) {
                if (compressedSeq.length >= 2) {
                    const pathPoints: { x: number; y: number }[] = []
                    compressedSeq.forEach(id => {
                        const p = nodePositions.get(id)
                        if (p) pathPoints.push(p)
                    })
                    if (pathPoints.length >= 2) {
                        const line = d3.line<{ x: number; y: number }>()
                            .x(d => d.x)
                            .y(d => d.y)
                        const weight = 1
                        trajectoryGroup.append('path')
                            .attr('d', line(pathPoints))
                            .attr('fill', 'none')
                            .attr('stroke', colorForRollout(rolloutId, index))
                            .attr('stroke-width', Math.min(6, 3 + Math.log2(1 + weight) * 1.5))
                            .attr('vector-effect', 'non-scaling-stroke')
                            .attr('opacity', 0.7)
                            .attr('class', `trajectory trajectory-${rolloutId}`)
                            .style('cursor', 'pointer')
                            .on('mouseenter', function () {
                                d3.select(this).attr('opacity', 0.9)
                                setHoveredRollout(rolloutId)
                            })
                            .on('mouseleave', function () {
                                d3.select(this).attr('opacity', 0.7)
                                setHoveredRollout(null)
                            })
                    }
                }
            } else {
                // Directed: draw per-edge segments with a midpoint arrow marker
                rolloutEdges.forEach(e => {
                    if (!displayedIds.has(e.node_a) || !displayedIds.has(e.node_b)) return
                    const a = nodePositions.get(e.node_a)
                    const b = nodePositions.get(e.node_b)
                    if (!a || !b) return
                    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }
                    const d = `M ${a.x} ${a.y} L ${mid.x} ${mid.y} L ${b.x} ${b.y}`
                    const weight = 1
                    trajectoryGroup.append('path')
                        .attr('d', d)
                        .attr('fill', 'none')
                        .attr('stroke', colorForRollout(rolloutId, index))
                        .attr('stroke-width', Math.min(6, 3 + Math.log2(1 + weight) * 1.5))
                        .attr('vector-effect', 'non-scaling-stroke')
                        .attr('opacity', 0.7)
                        .attr('class', `trajectory trajectory-${rolloutId}`)
                        .attr('marker-mid', `url(#arrow-${rolloutId})`)
                })
            }

            // Also draw START -> first displayed node for this rollout (after filtering)
            const startPos = nodePositions.get('START')
            if (startPos) {
                const firstDisplayedNodeId = (function () {
                    const seq: string[] = []
                    rolloutEdges.forEach((e, i) => {
                        if (i === 0) seq.push(e.node_a)
                        seq.push(e.node_b)
                    })
                    const displayedIds = new Set(filteredNodes.map(n => n.cluster_id))
                    const compressed = seq.filter(id => displayedIds.has(id))
                    return compressed[0]
                })()

                if (firstDisplayedNodeId) {
                    const firstPos = nodePositions.get(firstDisplayedNodeId)
                    if (firstPos) {
                        const countsForRollout = rolloutEdgeCountRef.current.get(String(rolloutId)) || new Map()
                        const weight = countsForRollout.get('START->' + firstDisplayedNodeId) || 1
                        if (!directedEdges) {
                            const lineStart = d3.line<{ x: number; y: number }>()
                                .x(d => d.x)
                                .y(d => d.y)
                            trajectoryGroup.append('path')
                                .attr('d', lineStart([startPos, firstPos]))
                                .attr('fill', 'none')
                                .attr('stroke', colorForRollout(rolloutId, index))
                                .attr('stroke-width', Math.min(6, 3 + Math.log2(1 + weight) * 1.5))
                                .attr('vector-effect', 'non-scaling-stroke')
                                .attr('opacity', 0.7)
                                .attr('class', `trajectory trajectory-start-${rolloutId}`)
                        } else {
                            const mid = { x: (startPos.x + firstPos.x) / 2, y: (startPos.y + firstPos.y) / 2 }
                            const dseg = `M ${startPos.x} ${startPos.y} L ${mid.x} ${mid.y} L ${firstPos.x} ${firstPos.y}`
                            trajectoryGroup.append('path')
                                .attr('d', dseg)
                                .attr('fill', 'none')
                                .attr('stroke', colorForRollout(rolloutId, index))
                                .attr('stroke-width', Math.min(8, 4 + Math.log2(1 + weight) * 1.5))
                                .attr('vector-effect', 'non-scaling-stroke')
                                .attr('opacity', 0.7)
                                .attr('class', `trajectory trajectory-start-${rolloutId}`)
                                .attr('marker-mid', `url(#arrow-${rolloutId})`)
                        }
                        addIncoming(firstDisplayedNodeId, String(rolloutId), weight)
                    }
                }
            }

            // note: polyline for entire chain removed; segments are drawn individually above
        })

        const nodeGroup = g.append('g').attr('class', 'clusters')
        const startNode: any = { cluster_id: 'START', freq: 0, representative_sentence: '', sentences: [] }
        const displayedNodes = [...filteredNodes, startNode]
        const nodes = nodeGroup.selectAll('.cluster')
            .data(displayedNodes)
            .enter()
            .append('g')
            .attr('class', 'cluster')
            .attr('transform', d => {
                const p = nodePositions.get(d.cluster_id)
                const k = currentTransform ? currentTransform.k : 1
                return `translate(${p ? p.x : 0}, ${p ? p.y : 0}) scale(${1 / (k || 1)})`
            })
            .style('cursor', 'pointer')
            .on('click', (event, d) => setSelectedNode(d))
            .on('mouseenter', function (event, d) {
                setHighlightClusterId(d.cluster_id)
            })
            .on('mouseleave', function (event, d) {
                setHighlightClusterId(null)
            })

        nodes.append('circle')
            .attr('r', d => (d.cluster_id === 'START' || isResponseNodeId(d.cluster_id)) ? 20 : getClusterRadius(d.freq))
            .attr('fill', d => {
                if (d.cluster_id === 'START' || isResponseNodeId(d.cluster_id)) return '#000000'
                const counts = incomingByNode.get(d.cluster_id) || new Map()
                let total = 0
                newValidRollouts.forEach(rid => { total += (counts.get(String(rid)) || 0) })
                return total === 0 ? '#d1d5db' : '#ffffff'
            })
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2)
            .attr('vector-effect', 'non-scaling-stroke')
            .attr('opacity', 0.95)

        // Pie slices for nodes with incoming counts
        nodes.each(function (d: any) {
            if (d.cluster_id === 'START' || isResponseNodeId(d.cluster_id)) return
            const counts = incomingByNode.get(d.cluster_id) || new Map()
            let total = 0
            newValidRollouts.forEach(rid => { total += (counts.get(String(rid)) || 0) })
            if (total === 0) return
            const group = d3.select(this)
            const r = getClusterRadius(d.freq)
            const dataSlices = newValidRollouts.map((rid, idx) => ({ rid: String(rid), value: counts.get(String(rid)) || 0, color: rolloutColorMap.get(String(rid)) || colorForRollout(String(rid), idx) }))
            const pieGen = d3.pie<any>().value((x: any) => x.value).sort(null)
            const arcs = pieGen(dataSlices as any)
            const arcGen = d3.arc<any>().innerRadius(0).outerRadius(r)
            arcs.forEach((a: any) => {
                if (a.data.value <= 0) return
                group.append('path')
                    .attr('d', arcGen(a) as any)
                    .attr('fill', a.data.color)
                    .attr('opacity', 0.95)
            })
        })

        nodes.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('font-size', d => {
                if (d.cluster_id === 'START') return 10
                if (isResponseNodeId(d.cluster_id)) return 12
                return d.freq >= 5 ? Math.min(12, Math.max(8, getClusterRadius(d.freq) * 0.4)) : 0
            })
            .attr('font-weight', '600')
            .attr('fill', '#ffffff')
            .attr('text-shadow', '1px 1px 2px rgba(0,0,0,0.7)')
            .text(d => {
                if (d.cluster_id === 'START') return 'START'
                if (isResponseNodeId(d.cluster_id)) return d.representative_sentence || ''
                return d.freq >= 5 ? d.freq.toString() : ''
            })

        // Layering: gray nodes (bottom) < trajectories (middle) < active nodes (top)
        // 1) Move trajectories above the initial clusters layer
        trajectoryGroup.raise()
        // 2) Reparent active nodes into a new top layer so they stay above trajectories
        const activeLayer = g.append('g').attr('class', 'clusters-active')
        nodes.filter(function (d: any) {
            if (d.cluster_id === 'START') return true
            const counts = incomingByNode.get(d.cluster_id) || new Map()
            let total = 0
            newValidRollouts.forEach(rid => { total += (counts.get(String(rid)) || 0) })
            return total > 0
        }).each(function () {
            const parent = activeLayer.node()
            if (parent) parent.appendChild(this as any)
        })

        // ensure hover highlighting applies to entire group, not just circle
        nodes.on('mouseenter.highlight', function (event, d: any) {
            setHighlightClusterId(d.cluster_id)
            d3.select(this).select('circle')
                .attr('stroke-width', 4)
                .attr('opacity', 1)
                .attr('r', (d.cluster_id === 'START' || isResponseNodeId(d.cluster_id)) ? 24 : (d.cluster_id === 'START' ? 24 : getClusterRadius(d.freq) + 3))
            // try immediate scroll for rollout panel
            if (rolloutPanelRef.current) {
                const row = rolloutPanelRef.current.querySelector(`#rollout-row-${d.cluster_id}`) as HTMLElement | null
                if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
            }
        })
        nodes.on('mouseleave.highlight', function () {
            setHighlightClusterId(null)
            d3.select(this).select('circle')
                .attr('stroke-width', 2)
                .attr('opacity', 0.95)
                .attr('r', function (d: any) { return (d.cluster_id === 'START' || isResponseNodeId(d.cluster_id)) ? 20 : getClusterRadius(d.freq) })
        })

    }, [data, selectedRollouts, minClusterSize, positionsReady, showCorrect, showIncorrect, directedEdges])

    // Bidirectional hover sync: update node styles and auto-scroll rollout panel
    useEffect(() => {
        if (!svgRef.current) return
        const svg = d3.select(svgRef.current)
        const k = currentTransform ? currentTransform.k : 1
        const groups = svg.selectAll('.clusters .cluster, .clusters-active .cluster') as any
        groups.each(function (d: any) {
            const group = d3.select(this)
            const p = positionsRef.current.get(d.cluster_id)
            const hover = !!(highlightClusterId && d && d.cluster_id === highlightClusterId)
            const hoverScale = hover ? 1.2 : 1
            group.attr('transform', `translate(${p ? p.x : 0}, ${p ? p.y : 0}) scale(${(1 / (k || 1)) * hoverScale})`)
            const circle = group.select('circle')
            if (!circle.empty()) {
                if (hover) {
                    circle.attr('stroke-width', 4).attr('opacity', 1)
                } else {
                    circle.attr('stroke-width', 2).attr('opacity', 0.95)
                }
            }
        })
        if (highlightClusterId && rolloutPanelRef.current) {
            const container = rolloutPanelRef.current
            const doScroll = () => {
                const row = container.querySelector(`#rollout-row-${highlightClusterId}`) as HTMLElement | null
                if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
            }
            if (typeof requestAnimationFrame !== 'undefined') {
                requestAnimationFrame(doScroll)
            } else {
                setTimeout(doScroll, 0)
            }
        }
    }, [highlightClusterId, panelRollout, minClusterSize, currentTransform])

    if (selectedRollouts.length === 0) {
        return (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <div style={{ color: '#6b7280', fontSize: '1.125rem' }}>
                    Select rollouts to see the cluster trajectory visualization
                </div>
            </div>
        )
    }

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            {/* Controls */}
            <div style={{
                position: 'absolute',
                top: '20px',
                left: '20px',
                zIndex: 1000,
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(229, 231, 235, 0.8)',
                borderRadius: '12px',
                padding: '16px',
                boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
            }}>
                <div style={{ marginBottom: '12px', fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                    Cluster Size Filter
                </div>
                <div style={{ marginBottom: '8px' }}>
                    {(() => {
                        const SLIDER_MAX = 1000
                        const denom = Math.max(1, maxSize - minSize)
                        const fromClusterSize = (size: number) => {
                            const norm = Math.max(0, Math.min(1, (size - minSize) / denom))
                            return Math.round(Math.sqrt(norm) * SLIDER_MAX)
                        }
                        const toClusterSize = (val: number) => {
                            const t = Math.max(0, Math.min(1, val / SLIDER_MAX))
                            const mapped = minSize + denom * (t * t)
                            return Math.round(mapped)
                        }
                        const sliderValue = fromClusterSize(minClusterSize)
                        return (
                            <input
                                type="range"
                                min={0}
                                max={SLIDER_MAX}
                                value={sliderValue}
                                onChange={(e) => setMinClusterSize(toClusterSize(parseInt(e.target.value)))}
                                style={{ width: '200px' }}
                            />
                        )
                    })()}
                </div>
                <div style={{ fontSize: '12px', color: '#6b7280' }}>
                    Min size: {minClusterSize} | Showing {filteredNodes.length} clusters
                </div>
                <div style={{ marginTop: '12px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#374151', cursor: 'pointer' }}>
                        <input type="checkbox" checked={directedEdges} onChange={(e) => setDirectedEdges(e.target.checked)} />
                        Directed edges
                    </label>
                </div>
            </div>

            {/* Rollout Legend (right) */}
            {validRollouts.length > 0 && (
                <div style={{
                    position: 'absolute',
                    top: '20px',
                    right: '20px',
                    zIndex: 1000,
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(229, 231, 235, 0.8)',
                    borderRadius: '12px',
                    padding: '16px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
                }}>
                    <div style={{ marginBottom: '12px', fontSize: '14px', fontWeight: '600', color: '#374151' }}>
                        Rollout Trajectories
                    </div>
                    {isBalanced19v20 && (
                        <div style={{ display: 'flex', gap: '12px', marginBottom: '10px' }}>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#374151' }}>
                                <input type="checkbox" checked={showCorrect} onChange={(e) => setShowCorrect(e.target.checked)} />
                                Correct
                            </label>
                            <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: '#374151' }}>
                                <input type="checkbox" checked={showIncorrect} onChange={(e) => setShowIncorrect(e.target.checked)} />
                                Incorrect
                            </label>
                        </div>
                    )}
                    {validRollouts.map((rolloutId, index) => (
                        <div key={rolloutId} style={{
                            display: 'flex',
                            alignItems: 'center',
                            marginBottom: '6px',
                            cursor: 'pointer',
                            opacity: hoveredRollout === rolloutId ? 1 : 0.9
                        }}
                            onMouseEnter={() => setHoveredRollout(rolloutId)}
                            onMouseLeave={() => setHoveredRollout(null)}
                            onClick={() => setPanelRollout(rolloutId)}
                        >
                            <div style={{
                                width: '12px',
                                height: '12px',
                                borderRadius: '50%',
                                backgroundColor: rolloutColorMap.get(rolloutId) || rolloutColors[index % rolloutColors.length],
                                marginRight: '8px'
                            }} />
                            <span style={{ fontSize: '12px', color: '#374151' }}>
                                Rollout {rolloutId}
                            </span>
                        </div>
                    ))}
                </div>
            )}

            {/* Rollout Details Panel (left) */}
            {panelRollout && (
                <div style={{
                    position: 'absolute',
                    top: '160px',
                    left: '20px',
                    zIndex: 1000,
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(229, 231, 235, 0.8)',
                    borderRadius: '12px',
                    padding: '12px',
                    maxHeight: '70%',
                    width: '320px',
                    overflowY: 'auto',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
                }} ref={rolloutPanelRef}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <div style={{ fontSize: '14px', fontWeight: '600', color: '#374151' }}>Rollout {panelRollout}</div>
                        <button onClick={() => setPanelRollout(null)} style={{ border: 'none', background: 'none', cursor: 'pointer', color: '#6b7280' }}>×</button>
                    </div>
                    {(() => {
                        const seq = getCompressedSequenceForRollout(panelRollout)
                        return (
                            <div>
                                {seq.map((cid) => {
                                    const node = data.nodes.find(n => n.cluster_id === cid)
                                    if (!node) return null
                                    const isHighlighted = highlightClusterId === cid
                                    return (
                                        <div key={cid}
                                            id={`rollout-row-${cid}`}
                                            onMouseEnter={() => setHighlightClusterId(cid)}
                                            onMouseLeave={() => setHighlightClusterId(null)}
                                            style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                padding: '10px',
                                                marginBottom: '6px',
                                                borderRadius: '8px',
                                                border: '1px solid #e5e7eb',
                                                backgroundColor: isHighlighted ? 'rgba(59,130,246,0.08)' : 'white'
                                            }}>
                                            <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: '#e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: '10px', fontSize: '12px', color: '#111827' }}>{node.freq}</div>
                                            <div style={{ flex: 1 }}>
                                                <div style={{ fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: '4px' }}>Cluster {cid}</div>
                                                <div style={{ fontSize: '12px', color: '#4b5563', whiteSpace: 'normal', wordBreak: 'break-word' }}>{node.representative_sentence}</div>
                                            </div>
                                        </div>
                                    )
                                })}
                                {seq.length === 0 && (
                                    <div style={{ fontSize: '12px', color: '#6b7280' }}>No clusters at current size</div>
                                )}
                            </div>
                        )
                    })()}
                </div>
            )}

            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

            {/* Node Details Modal */}
            {selectedNode && (
                <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    width: '400px',
                    maxHeight: '500px',
                    backgroundColor: 'rgba(255, 255, 255, 0.98)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid rgba(229, 231, 235, 0.8)',
                    borderRadius: '16px',
                    padding: '24px',
                    boxShadow: '0 20px 60px rgba(0,0,0,0.2)',
                    zIndex: 2000
                }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                        <h3 style={{ margin: 0, fontSize: '18px', color: '#111827' }}>
                            Cluster {selectedNode.cluster_id}
                        </h3>
                        <button
                            onClick={() => setSelectedNode(null)}
                            style={{
                                background: 'none',
                                border: 'none',
                                fontSize: '24px',
                                cursor: 'pointer',
                                color: '#6b7280',
                                padding: '4px'
                            }}
                        >
                            ×
                        </button>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                        <div style={{ fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '4px' }}>
                            Size: {selectedNode.freq} sentences
                        </div>
                        <div style={{ fontSize: '14px', fontWeight: '600', color: '#374151', marginBottom: '8px' }}>
                            Representative:
                        </div>
                        <p style={{ margin: '0 0 16px 0', fontSize: '14px', color: '#4b5563', lineHeight: '1.5' }}>
                            {selectedNode.representative_sentence}
                        </p>
                    </div>

                    <div style={{ marginBottom: '12px' }}>
                        <strong style={{ fontSize: '14px', color: '#374151' }}>
                            Sample Sentences ({selectedNode.sentences.length} total):
                        </strong>
                    </div>

                    <div style={{
                        maxHeight: '250px',
                        overflowY: 'auto',
                        overflowX: 'hidden',
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        padding: '12px',
                        backgroundColor: '#f9fafb'
                    }}>
                        {selectedNode.sentences.slice(0, 5).map((sentence, index) => (
                            <div key={index} style={{
                                marginBottom: '8px',
                                padding: '8px',
                                backgroundColor: 'white',
                                borderRadius: '6px',
                                fontSize: '12px',
                                border: '1px solid #e5e7eb',
                                wordWrap: 'break-word'
                            }}>
                                <div style={{ fontWeight: '600', marginBottom: '4px', color: '#374151' }}>
                                    Count: {sentence.count}
                                </div>
                                <div style={{ color: '#4b5563', lineHeight: '1.4' }}>
                                    {sentence.text}
                                </div>
                            </div>
                        ))}
                        {selectedNode.sentences.length > 5 && (
                            <div style={{ fontSize: '12px', color: '#6b7280', textAlign: 'center', marginTop: '8px' }}>
                                ... and {selectedNode.sentences.length - 5} more sentences
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
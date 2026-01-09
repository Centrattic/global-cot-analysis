// @ts-nocheck
'use client'

import { useEffect, useRef, useState, useMemo } from 'react'
import * as d3 from 'd3'
import { FlowchartData, Node } from '@/types/flowchart'
import GraphizLegend from './GraphizLegend'
import './graph.css'

interface ClusterAlgorithmStructureProps {
    data: FlowchartData
    promptIndex?: string
    selectedRollouts?: string[]
}

interface AlgorithmNode {
    id: string
    description: string
}

interface ClusterData {
    clusterId: string
    algorithmFractions: Map<string, number> // algorithm -> fraction (0-1)
    position: { x: number; y: number } | null
    jitteredPosition: { x: number; y: number } | null
}

interface ClusterEdge {
    source: string
    target: string
    weight: number
}

export default function ClusterAlgorithmStructure({ data, promptIndex, selectedRollouts = [] }: ClusterAlgorithmStructureProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selectedCluster, setSelectedCluster] = useState<ClusterData | null>(null)
    const [minEdgeWeight, setMinEdgeWeight] = useState<number>(0)
    const [maxEdgeWeight, setMaxEdgeWeight] = useState<number>(0)
    const [algorithms, setAlgorithms] = useState<Map<string, AlgorithmNode>>(new Map())
    const [clusters, setClusters] = useState<Map<string, ClusterData>>(new Map())
    const [edges, setEdges] = useState<ClusterEdge[]>([])
    const [currentTransform, setCurrentTransform] = useState<d3.ZoomTransform | null>(null)
    const [selectedRolloutForView, setSelectedRolloutForView] = useState<string | null>(null)
    const [rolloutEdgeSet, setRolloutEdgeSet] = useState<Set<string>>(new Set())
    const [showAllClusters, setShowAllClusters] = useState<boolean>(true)
    const [selectedRolloutResponse, setSelectedRolloutResponse] = useState<string | null>(null)
    const [hoveredClusterForSentence, setHoveredClusterForSentence] = useState<string | null>(null)
    const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null)
    const [rolloutJson, setRolloutJson] = useState<any | null>(null)
    const [loadingRollout, setLoadingRollout] = useState<boolean>(false)
    const [skipQuestionRestatements, setSkipQuestionRestatements] = useState<boolean>(false)
    const [collapseAnswerCycles, setCollapseAnswerCycles] = useState<boolean>(false)
    const [collapseAllCyclesExceptQuestion, setCollapseAllCyclesExceptQuestion] = useState<boolean>(false)

    // Fetch rollout JSON when selectedRolloutResponse changes (like GraphizRolloutPanel)
    useEffect(() => {
        if (!selectedRolloutResponse || !promptIndex) {
            setRolloutJson(null)
            return
        }
        const modelName: string | undefined = (data && (data as any).models && (data as any).models.length > 0)
            ? (data as any).models[0]
            : ((data && (data as any).model) ? (data as any).model : undefined)

        if (!modelName) {
            setRolloutJson(null)
            return
        }

        setLoadingRollout(true)
        fetch(`/api/rollout/${encodeURIComponent(promptIndex)}/${encodeURIComponent(modelName)}/${encodeURIComponent(selectedRolloutResponse)}`)
            .then(r => r.ok ? r.json() : null)
            .then(j => setRolloutJson(j))
            .catch(() => setRolloutJson(null))
            .finally(() => setLoadingRollout(false))
    }, [selectedRolloutResponse, promptIndex, data])

    // Detect data format
    const isOldFormat = data.nodes.length > 0 && data.nodes[0].cluster_id !== undefined

    // Helper function to get cluster ID and node data from both formats
    const getNodeInfo = (node: any) => {
        if (isOldFormat) {
            return { clusterId: node.cluster_id, nodeData: node }
        } else {
            const clusterKey = Object.keys(node)[0]
            return { clusterId: clusterKey, nodeData: node[clusterKey] }
        }
    }

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

    // Process clusters to compute algorithm fractions
    const processedClusters = useMemo(() => {
        if (!data || algorithms.size === 0) return new Map<string, ClusterData>()

        const clusterMap = new Map<string, ClusterData>()

        data.nodes.forEach(node => {
            const { clusterId, nodeData } = getNodeInfo(node)

            // Skip START and response nodes
            if (clusterId === 'START' || clusterId.startsWith('response-')) return

            const sentences = nodeData.sentences || []
            const algorithmCounts = new Map<string, number>()
            let totalCount = 0

            sentences.forEach((sentence: any) => {
                const count = sentence.count || 1
                let multiAlgo: any = sentence.debug_multi_algorithm || sentence.multi_algorithm

                if (!multiAlgo) {
                    // Skip sentences with no algorithms
                    return
                }

                // Handle string format (JSON string)
                if (typeof multiAlgo === 'string') {
                    try {
                        multiAlgo = JSON.parse(multiAlgo)
                    } catch {
                        return
                    }
                }

                if (!Array.isArray(multiAlgo) || multiAlgo.length === 0) {
                    return
                }

                // Extract algorithms from the array
                // Could be format ["A", 7, "D", 9] (alternating) or ["A", "B"] (simple list)
                const algosInSentence = new Set<string>()

                // Check if it's alternating format (numbers at odd indices)
                const hasNumbers = multiAlgo.some((item, idx) => idx % 2 === 1 && typeof item === 'number')

                if (hasNumbers) {
                    // Alternating format: strings at even indices
                    for (let i = 0; i < multiAlgo.length; i += 2) {
                        const algo = multiAlgo[i]
                        if (typeof algo === 'string' && algorithms.has(algo)) {
                            algosInSentence.add(algo)
                        }
                    }
                } else {
                    // Simple list format: all strings
                    multiAlgo.forEach((algo: any) => {
                        if (typeof algo === 'string' && algorithms.has(algo)) {
                            algosInSentence.add(algo)
                        }
                    })
                }

                // Count this sentence for each algorithm it contains
                if (algosInSentence.size > 0) {
                    totalCount += count
                    algosInSentence.forEach(algo => {
                        algorithmCounts.set(algo, (algorithmCounts.get(algo) || 0) + count)
                    })
                }
            })

            // Compute fractions
            const fractions = new Map<string, number>()
            if (totalCount > 0) {
                algorithmCounts.forEach((count, algo) => {
                    fractions.set(algo, count / totalCount)
                })
            }

            // Include all clusters, even if they have no algorithm fractions
            // (clusters without algorithm data will have empty fractions map)
            clusterMap.set(clusterId, {
                clusterId,
                algorithmFractions: fractions,
                position: null,
                jitteredPosition: null
            })
        })

        return clusterMap
    }, [data, algorithms, isOldFormat])

    useEffect(() => {
        setClusters(processedClusters)
    }, [processedClusters])

    // Position clusters on the circle based on algorithm fractions
    useEffect(() => {
        if (algorithms.size === 0 || processedClusters.size === 0) return

        const algorithmList = Array.from(algorithms.values())
        if (algorithmList.length === 0) return

        // Create circular layout for algorithm nodes
        const radius = 300
        const centerX = 0
        const centerY = 0

        const angleStep = (2 * Math.PI) / algorithmList.length
        const algorithmPositions = new Map<string, { x: number; y: number }>()

        algorithmList.forEach((alg, index) => {
            const angle = index * angleStep - Math.PI / 2
            const x = centerX + radius * Math.cos(angle)
            const y = centerY + radius * Math.sin(angle)
            algorithmPositions.set(alg.id, { x, y })
        })

        // Position clusters based on algorithm fractions
        const updatedClusters = new Map(processedClusters)

        updatedClusters.forEach((cluster, clusterId) => {
            const fractions = cluster.algorithmFractions

            // Calculate weighted position based on fractions
            let weightedX = 0
            let weightedY = 0
            let totalFraction = 0

            fractions.forEach((fraction, algoId) => {
                const pos = algorithmPositions.get(algoId)
                if (pos) {
                    weightedX += pos.x * fraction
                    weightedY += pos.y * fraction
                    totalFraction += fraction
                }
            })

            if (totalFraction > 0) {
                // Normalize by total fraction to get average position
                weightedX /= totalFraction
                weightedY /= totalFraction
                cluster.position = { x: weightedX, y: weightedY }
            } else {
                // Clusters without algorithm fractions: position them randomly near center
                // This handles clusters that don't have algorithm data
                const angle = Math.random() * 2 * Math.PI
                const distance = 50 + Math.random() * 50
                cluster.position = {
                    x: distance * Math.cos(angle),
                    y: distance * Math.sin(angle)
                }
            }
            updatedClusters.set(clusterId, cluster)
        })

        // Apply jitter to prevent overlap
        const jitterRadius = 40
        const jitteredClusters = new Map(updatedClusters)
        const positions = new Set<string>()

        jitteredClusters.forEach((cluster, clusterId) => {
            if (!cluster.position) return

            let attempts = 0
            let finalX = cluster.position.x
            let finalY = cluster.position.y

            // Try to find a non-overlapping position
            while (attempts < 50) {
                const posKey = `${Math.round(finalX)},${Math.round(finalY)}`
                if (!positions.has(posKey)) {
                    positions.add(posKey)
                    break
                }
                // Add jitter
                const angle = Math.random() * 2 * Math.PI
                const distance = Math.random() * jitterRadius
                finalX = cluster.position.x + distance * Math.cos(angle)
                finalY = cluster.position.y + distance * Math.sin(angle)
                attempts++
            }

            cluster.jitteredPosition = { x: finalX, y: finalY }
            jitteredClusters.set(clusterId, cluster)
        })

        setClusters(jitteredClusters)
    }, [algorithms, processedClusters])

    // Extract cluster edges from rollouts (only selected rollouts)
    const clusterEdges = useMemo(() => {
        if (!data || clusters.size === 0) return []

        // If no rollouts selected, return empty edges
        if (!selectedRollouts || selectedRollouts.length === 0) return []

        const responsesData: any = (data as any).responses || (data as any).rollouts || {}
        const isOldFormat = Array.isArray(responsesData) && responsesData.length > 0 && responsesData[0]?.index !== undefined

        const edgeMap = new Map<string, number>()

        // Build node label map for collapse/skip logic
        const nodeMap = new Map<string, any>()
            ; (data.nodes || []).forEach((n: any) => {
                const { clusterId, nodeData } = getNodeInfo(n)
                nodeMap.set(clusterId, nodeData)
            })

        const isAnswerLabel = (cid: string) => {
            const nd = nodeMap.get(cid)
            const lbl = nd ? nd.label : undefined
            return lbl === 'answer' || (typeof lbl === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(lbl))
        }
        const isQuestion = (cid: string) => {
            const nd = nodeMap.get(cid)
            return nd && nd.label === 'question'
        }

        const processSequence = (seq: string[]): string[] => {
            let compressed = [...seq]
            // Skip question restatements (bridge runs of question nodes)
            if (skipQuestionRestatements) {
                const bridged: string[] = []
                for (let i = 0; i < compressed.length; i++) {
                    const cur = compressed[i]
                    if (!isQuestion(cur)) { bridged.push(cur); continue }
                    let j = i
                    while (j + 1 < compressed.length && isQuestion(compressed[j + 1])) j++
                    const prev = bridged.length > 0 ? bridged[bridged.length - 1] : null
                    const next = (j + 1 < compressed.length) ? compressed[j + 1] : null
                    if (prev && next && prev !== next) {
                        if (bridged.length === 0 || bridged[bridged.length - 1] !== prev) bridged.push(prev)
                        if (bridged[bridged.length - 1] !== next) bridged.push(next)
                    }
                    i = j
                }
                if (bridged.length > 0) compressed = bridged
            }
            // Collapse cycles
            const shouldCollapseAll = collapseAllCyclesExceptQuestion
            if (collapseAnswerCycles || shouldCollapseAll) {
                const isEligible = (cid: string) => {
                    const nd = nodeMap.get(cid)
                    const lbl = nd ? nd.label : undefined
                    const q = lbl === 'question'
                    const a = isAnswerLabel(cid)
                    return shouldCollapseAll ? !q : a
                }
                const out: string[] = []
                let i = 0
                while (i < compressed.length) {
                    const cur = compressed[i]
                    out.push(cur)
                    if (!isEligible(cur)) { i++; continue }
                    let lastSame = -1
                    for (let j = compressed.length - 1; j > i; j--) { if (compressed[j] === cur) { lastSame = j; break } }
                    if (lastSame > i) {
                        let firstOther = -1
                        for (let k = i + 1; k < lastSame; k++) { const c = compressed[k]; if (isEligible(c) && c !== cur) { firstOther = k; break } }
                        if (firstOther !== -1) { for (let t = i + 1; t <= firstOther; t++) out.push(compressed[t]) }
                        const after = (lastSame + 1 < compressed.length) ? compressed[lastSame + 1] : null
                        if (after && /^response-/.test(after)) { if (out[out.length - 1] !== after) out.push(after); i = lastSame + 2 } else { i = lastSame + 1 }
                        continue
                    }
                    i++
                }
                const dedup: string[] = []
                out.forEach(id => { if (dedup.length === 0 || dedup[dedup.length - 1] !== id) dedup.push(id) })
                compressed = dedup
            }
            // Final dedup of consecutive duplicates
            const finalSeq: string[] = []
            compressed.forEach(id => { if (finalSeq.length === 0 || finalSeq[finalSeq.length - 1] !== id) finalSeq.push(id) })
            return finalSeq
        }

        // Only process selected rollouts
        selectedRollouts.forEach(rid => {
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

            // Get edges from rollout
            const edges = rolloutData.edges || []
            if (!Array.isArray(edges) || edges.length === 0) return

            // Normalize node ID helper
            const normalizeNodeId = (nodeId: string) => {
                if (/^\d+$/.test(nodeId)) {
                    return `cluster-${nodeId}`
                }
                return nodeId
            }

            // Build sequence of clusters from edges
            const clusterSequence: string[] = []
            edges.forEach((edge: any, idx: number) => {
                const nodeA = normalizeNodeId(edge.node_a)
                const nodeB = normalizeNodeId(edge.node_b)

                // Add first node if this is the first edge
                if (idx === 0 && nodeA !== 'START' && !nodeA.startsWith('response-') && clusters.has(nodeA)) {
                    clusterSequence.push(nodeA)
                }

                // Add second node
                if (nodeB !== 'START' && !nodeB.startsWith('response-') && clusters.has(nodeB)) {
                    // Only add if different from last added
                    if (clusterSequence.length === 0 || clusterSequence[clusterSequence.length - 1] !== nodeB) {
                        clusterSequence.push(nodeB)
                    }
                }
            })

            // Apply collapse/skip processing
            const processedSeq = processSequence(clusterSequence)

            // Create edges between consecutive clusters (processed)
            for (let i = 0; i < processedSeq.length - 1; i++) {
                const source = processedSeq[i]
                const target = processedSeq[i + 1]
                if (source && target) {
                    const key = `${source}->${target}`
                    edgeMap.set(key, (edgeMap.get(key) || 0) + 1)
                }
            }
        })

        return Array.from(edgeMap.entries()).map(([key, weight]) => {
            const [source, target] = key.split('->')
            return { source, target, weight }
        })
    }, [data, clusters, selectedRollouts, skipQuestionRestatements, collapseAnswerCycles, collapseAllCyclesExceptQuestion])

    // Filter clusters based on toggle setting
    // Rollouts only affect which edges are shown
    const visibleClusters = useMemo(() => {
        if (showAllClusters) {
            // Show all clusters
            return clusters
        } else {
            // Only show clusters with algorithm fractions
            const filtered = new Map<string, ClusterData>()
            clusters.forEach((cluster, clusterId) => {
                if (cluster.algorithmFractions.size > 0) {
                    filtered.set(clusterId, cluster)
                }
            })
            return filtered
        }
    }, [clusters, showAllClusters])

    // Compute rollout path if a rollout is selected for view
    const rolloutPath = useMemo(() => {
        if (!selectedRolloutForView || !data) return []

        const responsesData: any = (data as any).responses || (data as any).rollouts || {}
        const isOldFormat = Array.isArray(responsesData) && responsesData.length > 0 && responsesData[0]?.index !== undefined

        let rolloutData: any = null
        if (Array.isArray(responsesData)) {
            if (isOldFormat) {
                rolloutData = responsesData.find((r: any) => r.index && String(r.index) === selectedRolloutForView)
            } else {
                const rolloutObj = responsesData.find((x: any) => x[selectedRolloutForView])
                rolloutData = rolloutObj ? rolloutObj[selectedRolloutForView] : null
            }
        } else {
            rolloutData = responsesData[selectedRolloutForView]
        }

        if (!rolloutData) return []

        const normalizeNodeId = (nodeId: string) => {
            if (/^\d+$/.test(nodeId)) {
                return `cluster-${nodeId}`
            }
            return nodeId
        }

        const edges = rolloutData.edges || []
        const clusterSequence: string[] = []

        edges.forEach((edge: any, idx: number) => {
            const nodeA = normalizeNodeId(edge.node_a)
            const nodeB = normalizeNodeId(edge.node_b)

            if (idx === 0 && nodeA !== 'START' && !nodeA.startsWith('response-') && visibleClusters.has(nodeA)) {
                clusterSequence.push(nodeA)
            }
            if (nodeB !== 'START' && !nodeB.startsWith('response-') && visibleClusters.has(nodeB)) {
                if (clusterSequence.length === 0 || clusterSequence[clusterSequence.length - 1] !== nodeB) {
                    clusterSequence.push(nodeB)
                }
            }
        })

        // Build node label map
        const nodeMap = new Map<string, any>()
            ; (data.nodes || []).forEach((n: any) => {
                const { clusterId, nodeData } = getNodeInfo(n)
                nodeMap.set(clusterId, nodeData)
            })
        const isAnswerLabel = (cid: string) => {
            const nd = nodeMap.get(cid)
            const lbl = nd ? nd.label : undefined
            return lbl === 'answer' || (typeof lbl === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(lbl))
        }
        const isQuestion = (cid: string) => { const nd = nodeMap.get(cid); return nd && nd.label === 'question' }
        const processSequence = (seq: string[]): string[] => {
            let compressed = [...seq]
            if (skipQuestionRestatements) {
                const bridged: string[] = []
                for (let i = 0; i < compressed.length; i++) { const cur = compressed[i]; if (!isQuestion(cur)) { bridged.push(cur); continue } let j = i; while (j + 1 < compressed.length && isQuestion(compressed[j + 1])) j++; const prev = bridged.length > 0 ? bridged[bridged.length - 1] : null; const next = (j + 1 < compressed.length) ? compressed[j + 1] : null; if (prev && next && prev !== next) { if (bridged.length === 0 || bridged[bridged.length - 1] !== prev) bridged.push(prev); if (bridged[bridged.length - 1] !== next) bridged.push(next) } i = j }
                if (bridged.length > 0) compressed = bridged
            }
            const shouldCollapseAll = collapseAllCyclesExceptQuestion
            if (collapseAnswerCycles || shouldCollapseAll) {
                const isEligible = (cid: string) => { const nd = nodeMap.get(cid); const lbl = nd ? nd.label : undefined; const q = lbl === 'question'; const a = isAnswerLabel(cid); return shouldCollapseAll ? !q : a }
                const out: string[] = []
                let i = 0
                while (i < compressed.length) { const cur = compressed[i]; out.push(cur); if (!isEligible(cur)) { i++; continue } let lastSame = -1; for (let j = compressed.length - 1; j > i; j--) { if (compressed[j] === cur) { lastSame = j; break } } if (lastSame > i) { let firstOther = -1; for (let k = i + 1; k < lastSame; k++) { const c = compressed[k]; if (isEligible(c) && c !== cur) { firstOther = k; break } } if (firstOther !== -1) { for (let t = i + 1; t <= firstOther; t++) out.push(compressed[t]) } const after = (lastSame + 1 < compressed.length) ? compressed[lastSame + 1] : null; if (after && /^response-/.test(after)) { if (out[out.length - 1] !== after) out.push(after); i = lastSame + 2 } else { i = lastSame + 1 } continue } i++ }
                const dedup: string[] = []
                out.forEach(id => { if (dedup.length === 0 || dedup[dedup.length - 1] !== id) dedup.push(id) })
                compressed = dedup
            }
            const finalSeq: string[] = []
            compressed.forEach(id => { if (finalSeq.length === 0 || finalSeq[finalSeq.length - 1] !== id) finalSeq.push(id) })
            return finalSeq
        }

        return processSequence(clusterSequence)
    }, [selectedRolloutForView, data, visibleClusters, skipQuestionRestatements, collapseAnswerCycles, collapseAllCyclesExceptQuestion])

    // Compute rollout edge set
    useEffect(() => {
        const edgeSet = new Set<string>()
        for (let i = 0; i < rolloutPath.length - 1; i++) {
            const key = `${rolloutPath[i]}->${rolloutPath[i + 1]}`
            edgeSet.add(key)
        }
        setRolloutEdgeSet(edgeSet)
    }, [rolloutPath])

    // Update edges state
    useEffect(() => {
        setEdges(clusterEdges)

        if (clusterEdges.length > 0) {
            const weights = clusterEdges.map(e => e.weight)
            const maxWeight = Math.max(...weights)
            setMaxEdgeWeight(maxWeight)
            setMinEdgeWeight(0)
        }
    }, [clusterEdges])

    // Filter edges by weight
    const filteredEdges = useMemo(() => {
        return edges.filter(e => e.weight >= minEdgeWeight)
    }, [edges, minEdgeWeight])

    // Draw visualization
    useEffect(() => {
        if (!svgRef.current || algorithms.size === 0 || visibleClusters.size === 0) return

        const svg = d3.select(svgRef.current)
        svg.selectAll('*').remove()

        const width = 1200
        const height = 900
        svg.attr('width', width).attr('height', height)

        // Create main group (no initial transform, zoom will handle it)
        const g = svg.append('g')

        // Setup zoom (similar to GraphizVisualization)
        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                g.attr('transform', event.transform)
                setCurrentTransform(event.transform)
            })

        svg.call(zoom)
        if (currentTransform) {
            g.attr('transform', currentTransform.toString())
            svg.call(zoom.transform, currentTransform)
        } else {
            // Initial zoom to center the visualization
            const initialTransform = d3.zoomIdentity
                .translate(width / 2, height / 2)
                .scale(1)
            g.attr('transform', initialTransform.toString())
            svg.call(zoom.transform, initialTransform)
            setCurrentTransform(initialTransform)
        }

        // Create arrow marker definitions (for directed edges at midpoint)
        const defs = svg.append('defs')
        const marker = defs.append('marker')
            .attr('id', 'arrow-cluster-mid')
            .attr('id', 'arrow-cluster-mid')
            .attr('viewBox', '0 0 10 10')
            .attr('refX', 5)
            .attr('refX', 5)
            .attr('refY', 5)
            .attr('markerWidth', 8)
            .attr('markerHeight', 8)
            .attr('orient', 'auto')
            .attr('markerUnits', 'userSpaceOnUse')
        marker.append('path')
            .attr('d', 'M 0 0 L 10 5 L 0 10 z')
            .attr('fill', '#94a3b8')
            .attr('stroke', '#94a3b8')
            .attr('stroke-width', 0.5)

        // Draw cluster edges
        const edgeGroup = g.append('g').attr('class', 'cluster-edges')
        filteredEdges.forEach(edge => {
            const sourceCluster = visibleClusters.get(edge.source)
            const targetCluster = visibleClusters.get(edge.target)

            if (!sourceCluster?.jitteredPosition || !targetCluster?.jitteredPosition) return

            const edgeKey = `${edge.source}->${edge.target}`
            const isInRolloutPath = selectedRolloutForView ? rolloutEdgeSet.has(edgeKey) : false

            // Calculate midpoint for arrow placement
            const midX = (sourceCluster.jitteredPosition.x + targetCluster.jitteredPosition.x) / 2
            const midY = (sourceCluster.jitteredPosition.y + targetCluster.jitteredPosition.y) / 2

            // Calculate angle for arrow direction
            const dx = targetCluster.jitteredPosition.x - sourceCluster.jitteredPosition.x
            const dy = targetCluster.jitteredPosition.y - sourceCluster.jitteredPosition.y
            const angle = Math.atan2(dy, dx) * (180 / Math.PI)

            // Draw the edge line
            const line = edgeGroup.append('line')
                .attr('x1', sourceCluster.jitteredPosition.x)
                .attr('y1', sourceCluster.jitteredPosition.y)
                .attr('x2', targetCluster.jitteredPosition.x)
                .attr('y2', targetCluster.jitteredPosition.y)
                .attr('fill', 'none')
                .attr('stroke', isInRolloutPath ? '#3b82f6' : '#94a3b8')
                .attr('stroke-width', isInRolloutPath
                    ? Math.max(2, Math.min(6, edge.weight * 0.3 + 2))
                    : Math.max(1, Math.min(5, edge.weight * 0.2 + 1)))
                .attr('opacity', selectedRolloutForView ? (isInRolloutPath ? 0.9 : 0.2) : 0.6)
                .attr('vector-effect', 'non-scaling-stroke')
                .attr('marker-mid', 'url(#arrow-cluster-mid)')
                .attr('class', `cluster-edge cluster-edge-${edge.source}-${edge.target}`)

            // Add arrow marker at midpoint
            const arrowGroup = edgeGroup.append('g')
                .attr('transform', `translate(${midX}, ${midY}) rotate(${angle})`)

            arrowGroup.append('path')
                .attr('d', 'M -5 0 L 5 -4 L 5 4 Z')
                .attr('fill', isInRolloutPath ? '#3b82f6' : '#94a3b8')
                .attr('stroke', isInRolloutPath ? '#3b82f6' : '#94a3b8')
                .attr('stroke-width', 0.5)
                .attr('opacity', selectedRolloutForView ? (isInRolloutPath ? 0.9 : 0.2) : 0.6)
        })

        // Create circular layout for algorithm nodes
        const algorithmList = Array.from(algorithms.values())
        const angleStep = (2 * Math.PI) / algorithmList.length
        const radius = 300

        // Draw algorithm nodes
        const algorithmNodeGroup = g.append('g').attr('class', 'algorithm-nodes')
        algorithmList.forEach((alg, index) => {
            const angle = index * angleStep - Math.PI / 2
            const x = radius * Math.cos(angle)
            const y = radius * Math.sin(angle)

            const nodeGroup = algorithmNodeGroup.append('g')
                .attr('class', `algorithm-node algorithm-node-${alg.id}`)
                .attr('transform', `translate(${x}, ${y})`)
                .style('cursor', 'pointer')

            nodeGroup.append('circle')
                .attr('r', 25)
                .attr('fill', '#3b82f6')
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 3)
                .attr('opacity', 0.9)

            // Position label outside the circle, along the radius
            const labelDistance = 35
            const labelX = labelDistance * Math.cos(angle)
            const labelY = labelDistance * Math.sin(angle)

            nodeGroup.append('text')
                .attr('x', labelX)
                .attr('y', labelY)
                .attr('text-anchor', 'middle')
                .attr('dy', '0.35em')
                .attr('font-size', '14px')
                .attr('font-weight', '600')
                .attr('fill', '#1f2937')
                .attr('style', 'pointer-events: none;')
                .text(alg.id)

            nodeGroup.on('click', () => {
                setSelectedAlgorithm(alg.id)
            })

            nodeGroup.on('mouseenter', function () {
                d3.select(this).select('circle')
                    .attr('r', 30)
                    .attr('fill', '#2563eb')
            })

            nodeGroup.on('mouseleave', function () {
                d3.select(this).select('circle')
                    .attr('r', 25)
                    .attr('fill', '#3b82f6')
            })
        })

        // Draw clusters with hover highlighting
        const clusterGroup = g.append('g').attr('class', 'clusters')
        visibleClusters.forEach(cluster => {
            if (!cluster.jitteredPosition) return

            const nodeGroup = clusterGroup.append('g')
                .attr('class', `cluster-node cluster-node-${cluster.clusterId}`)
                .attr('transform', `translate(${cluster.jitteredPosition.x}, ${cluster.jitteredPosition.y})`)
                .style('cursor', 'pointer')

            // Check if this cluster is being hovered (from sentence hover)
            const isHoveredFromSentence = hoveredClusterForSentence === cluster.clusterId

            // Calculate dominant algorithm for color
            let dominantAlgo = ''
            let maxFraction = 0
            cluster.algorithmFractions.forEach((fraction, algoId) => {
                if (fraction > maxFraction) {
                    maxFraction = fraction
                    dominantAlgo = algoId
                }
            })

            const algoColor = dominantAlgo ? '#8b5cf6' : '#94a3b8'

            const baseRadius = isHoveredFromSentence ? 14 : 8
            const baseOpacity = isHoveredFromSentence ? 1 : 0.8
            const highlightColor = isHoveredFromSentence ? '#2563eb' : algoColor

            nodeGroup.append('circle')
                .attr('r', baseRadius)
                .attr('fill', highlightColor)
                .attr('stroke', isHoveredFromSentence ? '#1e40af' : '#ffffff')
                .attr('stroke-width', isHoveredFromSentence ? 2 : 1)
                .attr('opacity', baseOpacity)

            nodeGroup.on('click', () => {
                // Get the full cluster data from processedClusters to include sentences
                const fullClusterData = {
                    ...cluster,
                    nodeData: null as any
                }
                // Find the node data to get sentences
                data.nodes.forEach(node => {
                    const { clusterId, nodeData } = getNodeInfo(node)
                    if (clusterId === cluster.clusterId) {
                        fullClusterData.nodeData = nodeData
                    }
                })
                setSelectedCluster(fullClusterData as any)
            })

            nodeGroup.on('mouseenter', function () {
                if (!isHoveredFromSentence) {
                    d3.select(this).select('circle')
                        .attr('r', 12)
                        .attr('opacity', 1)
                }
            })

            nodeGroup.on('mouseleave', function () {
                if (!isHoveredFromSentence) {
                    d3.select(this).select('circle')
                        .attr('r', baseRadius)
                        .attr('opacity', baseOpacity)
                }
            })
        })

        // Initial transform is already applied via centerGroup above

    }, [algorithms, visibleClusters, filteredEdges, selectedRolloutForView, rolloutEdgeSet, hoveredClusterForSentence, data])

    // Get valid rollouts for legend (must be before early returns)
    const validRollouts = useMemo(() => {
        if (!selectedRollouts || selectedRollouts.length === 0) return []
        return selectedRollouts
    }, [selectedRollouts])

    // Generate rollout colors (must be before early returns)
    const rolloutColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    const rolloutColorMap = useMemo(() => {
        const map = new Map<string, string>()
        validRollouts.forEach((rid, idx) => {
            map.set(rid, rolloutColors[idx % rolloutColors.length])
        })
        return map
    }, [validRollouts])

    // Dummy functions for GraphizLegend (must be before early returns)
    const getRolloutPropertyValue = (_rolloutId: string, _propertyName: string) => undefined

    if (algorithms.size === 0) {
        return (
            <div className="emptyState">
                <div className="emptyStateText">
                    Loading algorithms...
                </div>
            </div>
        )
    }

    if (selectedRollouts.length === 0) {
        return (
            <div className="emptyState">
                <div className="emptyStateText">
                    Select rollouts to see the cluster trajectory visualization
                </div>
            </div>
        )
    }

    if (visibleClusters.size === 0) {
        return (
            <div className="emptyState">
                <div className="emptyStateText">
                    No clusters with algorithm data found in selected rollouts.
                </div>
            </div>
        )
    }

    return (
        <div className="container">
            {/* Rollout Legend */}
            {validRollouts.length > 0 && (
                <GraphizLegend
                    validRollouts={validRollouts}
                    rolloutColorMap={rolloutColorMap}
                    rolloutColors={rolloutColors}
                    hoveredRollout={selectedRolloutForView}
                    onRolloutHover={setSelectedRolloutForView}
                    onRolloutClick={(rid) => {
                        setSelectedRolloutForView(rid)
                        setSelectedRolloutResponse(rid)
                    }}
                    enabledPropertyCheckers={new Set()}
                    getRolloutPropertyValue={getRolloutPropertyValue}
                />
            )}

            {/* Controls */}
            <div className="controls">
                {/* moved toggles to bottom of panel */}
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
                <div className="checkboxContainer" style={{ marginTop: 16 }}>
                    <div className="controlsTitle">
                        Clusters ({visibleClusters.size})
                    </div>
                    <div className="sliderInfo" style={{ fontSize: '11px', color: '#6b7280' }}>
                        Clusters positioned by algorithm fractions
                    </div>
                </div>

                {/* Toggles (moved to bottom) */}
                <div className="checkboxContainer" style={{ marginTop: 16 }}>
                    <label className="checkboxLabel" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                        <input
                            type="checkbox"
                            checked={showAllClusters}
                            onChange={(e) => setShowAllClusters(e.target.checked)}
                            style={{ marginRight: 8, cursor: 'pointer' }}
                        />
                        <span>Show all clusters (including those without algorithm data)</span>
                    </label>
                </div>
                <div className="checkboxContainer" style={{ marginTop: 8 }}>
                    <label className="checkboxLabel" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                        <input
                            type="checkbox"
                            checked={collapseAnswerCycles}
                            onChange={(e) => setCollapseAnswerCycles(e.target.checked)}
                            style={{ marginRight: 8, cursor: 'pointer' }}
                        />
                        <span style={{ color: '#8b5cf6', fontWeight: 600 }}>Collapse answer cycles</span>
                    </label>
                </div>
                <div className="checkboxContainer" style={{ marginTop: 8 }}>
                    <label className="checkboxLabel" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                        <input
                            type="checkbox"
                            checked={collapseAllCyclesExceptQuestion}
                            onChange={(e) => setCollapseAllCyclesExceptQuestion(e.target.checked)}
                            style={{ marginRight: 8, cursor: 'pointer' }}
                        />
                        <span style={{ color: '#3b82f6', fontWeight: 600 }}>Collapse all cycles (except question restatement nodes)</span>
                    </label>
                </div>
                <div className="checkboxContainer" style={{ marginTop: 8, marginBottom: 8 }}>
                    <label className="checkboxLabel" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                        <input
                            type="checkbox"
                            checked={skipQuestionRestatements}
                            onChange={(e) => setSkipQuestionRestatements(e.target.checked)}
                            style={{ marginRight: 8, cursor: 'pointer' }}
                        />
                        <span style={{ color: '#FFD700', fontWeight: 600 }}>Skip question restatements</span>
                    </label>
                </div>
            </div>

            {/* Main visualization */}
            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

            {/* Rollout Response Modal */}
            {selectedRolloutResponse && (() => {
                if (loadingRollout) {
                    return (
                        <div className="nodeModal">
                            <div className="modalHeader">
                                <h3 className="modalTitle">Response {selectedRolloutResponse}</h3>
                                <button
                                    onClick={() => setSelectedRolloutResponse(null)}
                                    className="modalCloseButton"
                                >
                                    ×
                                </button>
                            </div>
                            <div className="modalContent" style={{ padding: '16px' }}>
                                <div>Loading rollout...</div>
                            </div>
                        </div>
                    )
                }

                if (!rolloutJson) {
                    return (
                        <div className="nodeModal">
                            <div className="modalHeader">
                                <h3 className="modalTitle">Response {selectedRolloutResponse}</h3>
                                <button
                                    onClick={() => setSelectedRolloutResponse(null)}
                                    className="modalCloseButton"
                                >
                                    ×
                                </button>
                            </div>
                            <div className="modalContent" style={{ padding: '16px' }}>
                                <div>Failed to load rollout data.</div>
                            </div>
                        </div>
                    )
                }

                // Get chunked content (like GraphizRolloutPanel)
                const chunkList: string[] = []
                if (Array.isArray(rolloutJson.chunked_cot_content)) {
                    chunkList.push(...rolloutJson.chunked_cot_content as string[])
                } else if (rolloutJson.cot_content) {
                    const content = (rolloutJson.cot_content || '').toString()
                    chunkList.push(...content.split(/\n+/).map((s: string) => s.trim()).filter(Boolean))
                }

                // Build rollout path with texts (like getRolloutPathWithTexts)
                const normalizeNodeId = (nodeId: string) => {
                    if (/^\d+$/.test(nodeId)) {
                        return `cluster-${nodeId}`
                    }
                    return nodeId
                }

                // Get rollout data from flowchart data (like GraphizVisualization does)
                // First try flowchart data, then fallback to API data
                const responsesData: any = (data as any).responses || (data as any).rollouts || {}
                const isOldFormat = Array.isArray(responsesData) && responsesData.length > 0 && responsesData[0]?.index !== undefined

                let rolloutDataFromFlowchart: any = null
                if (Array.isArray(responsesData)) {
                    if (isOldFormat) {
                        rolloutDataFromFlowchart = responsesData.find((r: any) => r.index && String(r.index) === selectedRolloutResponse)
                    } else {
                        const rolloutObj = responsesData.find((x: any) => x[selectedRolloutResponse])
                        rolloutDataFromFlowchart = rolloutObj ? rolloutObj[selectedRolloutResponse] : null
                    }
                } else {
                    rolloutDataFromFlowchart = responsesData[selectedRolloutResponse]
                }

                // Use edges from flowchart data if available, otherwise from API
                let edges: any[] = []
                if (rolloutDataFromFlowchart && rolloutDataFromFlowchart.edges) {
                    edges = rolloutDataFromFlowchart.edges
                    console.log('[DEBUG Rollout Modal] Using edges from flowchart data:', edges.length)
                } else if (rolloutJson.edges) {
                    edges = rolloutJson.edges
                    console.log('[DEBUG Rollout Modal] Using edges from API data:', edges.length)
                } else {
                    console.warn('[DEBUG Rollout Modal] No edges found in flowchart or API data!')
                }

                console.log('[DEBUG Rollout Modal] Raw edges:', edges.length, 'first 3:', edges.slice(0, 3))

                const normalizedEdges = edges.map((edge: any) => ({
                    node_a: normalizeNodeId(edge.node_a),
                    node_b: normalizeNodeId(edge.node_b),
                    step_text_a: edge.step_text_a,
                    step_text_b: edge.step_text_b,
                }))
                console.log('[DEBUG Rollout Modal] Normalized edges (first 3):', normalizedEdges.slice(0, 3))

                type PathItem = { id: string, text?: string, idx: number }
                const orig: PathItem[] = []
                normalizedEdges.forEach((e, i) => {
                    if (i === 0) orig.push({ id: e.node_a, text: e.step_text_a, idx: orig.length })
                    orig.push({ id: e.node_b, text: e.step_text_b, idx: orig.length })
                })
                console.log('[DEBUG Rollout Modal] Original path items (first 5):', orig.slice(0, 5).map(x => `${x.id} (idx=${x.idx})`))

                // If the FIRST STEP AFTER START has no text, prefer node_a of the second edge
                if (orig.length > 1 && (orig[1].text == null || String(orig[1].text).trim() === "")) {
                    if (normalizedEdges.length >= 2) {
                        const second = normalizedEdges[1]
                        if (second && second.step_text_a != null && String(second.step_text_a).trim() !== "") {
                            orig[1].text = second.step_text_a
                        }
                    }
                }

                // Build displayed cluster IDs from data.nodes (like GraphizVisualization)
                const displayedIds = new Set<string>()
                data.nodes.forEach((n: any) => {
                    const { clusterId } = getNodeInfo(n)
                    displayedIds.add(clusterId)
                })
                displayedIds.add('START')
                console.log('[DEBUG Rollout Modal] Displayed IDs count:', displayedIds.size, 'sample:', Array.from(displayedIds).slice(0, 10))

                // Filter path to only include displayed clusters
                const kept: PathItem[] = orig.filter(item => displayedIds.has(item.id))
                console.log('[DEBUG Rollout Modal] After filtering by displayedIds: kept count:', kept.length, 'kept (first 5):', kept.slice(0, 5).map(x => `${x.id} (idx=${x.idx})`))

                // After filtering, collapse only duplicates that were not originally adjacent
                const out: PathItem[] = []
                for (let i = 0; i < kept.length; i++) {
                    const cur = kept[i]
                    if (out.length === 0) { out.push(cur); continue }
                    const prev = out[out.length - 1]
                    if (prev.id === cur.id) {
                        // If these occurrences were originally adjacent, keep both
                        const wereAdjacent = (cur.idx - prev.idx) === 1
                        if (!wereAdjacent) {
                            // collapse non-adjacent duplicates made adjacent by filtering
                            continue
                        }
                    }
                    out.push(cur)
                }
                console.log('[DEBUG Rollout Modal] After deduplication: out count:', out.length, 'out (first 5):', out.slice(0, 5).map(x => `${x.id}`))

                // Remove START and response nodes, build initial sequence
                let compressedSeq: string[] = []
                const compressedStepTexts: (string | undefined)[] = []
                out.forEach(item => {
                    // Skip START and response nodes
                    if (item.id === 'START' || item.id.startsWith('response-')) return
                    if (compressedSeq.length === 0 || compressedSeq[compressedSeq.length - 1] !== item.id) {
                        compressedSeq.push(item.id)
                        compressedStepTexts.push(item.text)
                    }
                })

                // Build node map for label lookups
                const nodeMap = new Map<string, any>()
                data.nodes.forEach(node => {
                    const { clusterId, nodeData } = getNodeInfo(node)
                    nodeMap.set(clusterId, nodeData)
                })

                // Helper to check if a cluster is an answer node
                const isAnswerLabel = (cid: string) => {
                    const nd = nodeMap.get(cid)
                    const label = nd ? nd.label : undefined
                    return label === 'answer' || (typeof label === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(label))
                }

                // Helper to check if a cluster is a question node
                const isQuestion = (cid: string) => {
                    const nd = nodeMap.get(cid)
                    return nd && nd.label === 'question'
                }

                // Apply skipQuestionRestatements if enabled
                if (skipQuestionRestatements) {
                    // Remove consecutive runs of question-labeled clusters and bridge A->B
                    const bridged: string[] = []
                    const bridgedTexts: (string | undefined)[] = []
                    for (let i = 0; i < compressedSeq.length; i++) {
                        const cur = compressedSeq[i]
                        const curText = compressedStepTexts[i]
                        if (!isQuestion(cur)) {
                            bridged.push(cur)
                            bridgedTexts.push(curText)
                            continue
                        }
                        // we're at Q run; find end
                        let j = i
                        while (j + 1 < compressedSeq.length && isQuestion(compressedSeq[j + 1])) j++
                        const prev = bridged.length > 0 ? bridged[bridged.length - 1] : null
                        const next = (j + 1 < compressedSeq.length) ? compressedSeq[j + 1] : null
                        if (prev && next && prev !== next) {
                            // Add edge prev->next via inserting next if needed; avoid duplicate consecutive
                            if (bridged.length === 0 || bridged[bridged.length - 1] !== prev) {
                                const prevIdx = compressedSeq.indexOf(prev, i - 1)
                                bridged.push(prev)
                                bridgedTexts.push(compressedStepTexts[prevIdx])
                            }
                            if (bridged[bridged.length - 1] !== next) {
                                bridged.push(next)
                                bridgedTexts.push(compressedStepTexts[j + 1])
                            }
                        }
                        i = j
                    }
                    if (bridged.length > 0) {
                        compressedSeq = bridged
                        compressedStepTexts.splice(0, compressedStepTexts.length, ...bridgedTexts)
                    }
                }

                // Apply collapseAnswerCycles or collapseAllCyclesExceptQuestion if enabled
                const shouldCollapseAll = collapseAllCyclesExceptQuestion
                if (collapseAnswerCycles || shouldCollapseAll) {
                    const isEligible = (cid: string) => {
                        const nd = nodeMap.get(cid)
                        const lbl = nd ? nd.label : undefined
                        const isQuestionNode = lbl === 'question'
                        const isAnswerNode = isAnswerLabel(cid)
                        return shouldCollapseAll ? !isQuestionNode : isAnswerNode
                    }
                    const out: string[] = []
                    const outTexts: (string | undefined)[] = []
                    let i = 0
                    while (i < compressedSeq.length) {
                        const cur = compressedSeq[i]
                        const curText = compressedStepTexts[i]
                        out.push(cur)
                        outTexts.push(curText)
                        if (!isEligible(cur)) { i++; continue }
                        // find next occurrence of same eligible node
                        let lastSame = -1
                        for (let j = compressedSeq.length - 1; j > i; j--) {
                            if (compressedSeq[j] === cur) { lastSame = j; break }
                        }
                        if (lastSame > i) {
                            // find first other eligible between i and lastSame
                            let firstOther = -1
                            for (let k = i + 1; k < lastSame; k++) {
                                const c = compressedSeq[k]
                                if (isEligible(c) && c !== cur) { firstOther = k; break }
                            }
                            if (firstOther !== -1) {
                                for (let t = i + 1; t <= firstOther; t++) {
                                    out.push(compressedSeq[t])
                                    outTexts.push(compressedStepTexts[t])
                                }
                            }
                            const after = (lastSame + 1 < compressedSeq.length) ? compressedSeq[lastSame + 1] : null
                            if (after && /^response-/.test(after)) {
                                if (out[out.length - 1] !== after) {
                                    out.push(after)
                                    outTexts.push(compressedStepTexts[lastSame + 1])
                                }
                                i = lastSame + 2
                            } else {
                                i = lastSame + 1
                            }
                            continue
                        }
                        i++
                    }
                    // dedup consecutive
                    const dedup: string[] = []
                    const dedupTexts: (string | undefined)[] = []
                    out.forEach((id, idx) => {
                        if (dedup.length === 0 || dedup[dedup.length - 1] !== id) {
                            dedup.push(id)
                            dedupTexts.push(outTexts[idx])
                        }
                    })
                    compressedSeq = dedup
                    compressedStepTexts.splice(0, compressedStepTexts.length, ...dedupTexts)
                }

                // Final sequence and texts
                const seq = compressedSeq
                const stepTexts = compressedStepTexts
                console.log('[DEBUG Rollout Modal] Final seq count:', seq.length, 'seq:', seq)

                // Compute repeated clusters and colors (like GraphizRolloutPanel)
                const counts = new Map<string, number>()
                seq.forEach(cid => counts.set(cid, (counts.get(cid) || 0) + 1))
                const repeatedIds = new Set<string>(Array.from(counts.entries()).filter(([_, c]) => c && c > 1).map(([cid]) => cid))
                const repeatedList = Array.from(repeatedIds)
                const hash = (s: string) => {
                    let h = 0
                    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
                    return h >>> 0
                }
                const hueOffset = selectedRolloutResponse ? (hash(selectedRolloutResponse) % 360) : 0
                const colorByClusterId = new Map<string, string>()
                if (repeatedList.length > 0) {
                    const step = 360 / repeatedList.length
                    repeatedList.sort()
                    repeatedList.forEach((cid, idx) => {
                        const hue = Math.round((hueOffset + idx * step) % 360)
                        colorByClusterId.set(cid, `hsl(${hue}, 70%, 85%)`)
                    })
                }

                return (
                    <div className="nodeModal" style={{ maxWidth: '800px', maxHeight: '90vh', overflow: 'auto' }}>
                        <div className="modalHeader">
                            <h3 className="modalTitle">
                                Response {selectedRolloutResponse}
                            </h3>
                            <button
                                onClick={() => setSelectedRolloutResponse(null)}
                                className="modalCloseButton"
                            >
                                ×
                            </button>
                        </div>
                        <div style={{ overflowY: 'auto' }}>
                            {seq.map((cid, idx) => {
                                const node = data.nodes.find((n: any) => {
                                    if (n.cluster_id) {
                                        return n.cluster_id === cid
                                    } else {
                                        const clusterKey = Object.keys(n)[0]
                                        return clusterKey === cid
                                    }
                                })
                                if (!node) return null
                                const isHighlighted = hoveredClusterForSentence === cid
                                const displayText = (stepTexts && stepTexts[idx]) ? String(stepTexts[idx]) : ''
                                const freq = node.cluster_id ? node.freq : (() => {
                                    const clusterKey = Object.keys(node)[0]
                                    return node[clusterKey].freq
                                })()

                                return (
                                    <div
                                        key={`${cid}-${idx}`}
                                        className={`rolloutRow ${isHighlighted ? 'rolloutRowHighlighted' : ''}`}
                                        style={{
                                            backgroundColor: isHighlighted ? undefined : (repeatedIds.has(cid) ? colorByClusterId.get(cid) : undefined)
                                        }}
                                        onMouseEnter={() => setHoveredClusterForSentence(cid)}
                                        onMouseLeave={() => setHoveredClusterForSentence(null)}
                                    >
                                        <div className="clusterCircle">
                                            {freq}
                                        </div>
                                        <div className="clusterInfo">
                                            <div className="clusterId">Cluster {cid}</div>
                                            <div className="clusterSentence">{displayText}</div>
                                        </div>
                                    </div>
                                )
                            })}
                            {seq.length === 0 && (
                                <div className="noClustersMessage">No clusters found</div>
                            )}
                        </div>
                    </div>
                )
            })()}

            {/* Algorithm Description Modal */}
            {selectedAlgorithm && algorithms.has(selectedAlgorithm) && (
                <div className="nodeModal">
                    <div className="modalHeader">
                        <h3 className="modalTitle">
                            Algorithm {selectedAlgorithm}
                        </h3>
                        <button
                            onClick={() => setSelectedAlgorithm(null)}
                            className="modalCloseButton"
                        >
                            ×
                        </button>
                    </div>
                    <div className="modalContent" style={{ padding: '16px' }}>
                        <div style={{ marginBottom: '8px' }}>
                            <strong>Description:</strong>
                        </div>
                        <p className="modalText" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                            {algorithms.get(selectedAlgorithm)?.description || 'No description available.'}
                        </p>
                    </div>
                </div>
            )}

            {/* Cluster Details Modal */}
            {selectedCluster && (
                <div className="nodeModal">
                    <div className="modalHeader">
                        <h3 className="modalTitle">
                            Cluster {selectedCluster.clusterId}
                        </h3>
                        <button
                            onClick={() => setSelectedCluster(null)}
                            className="modalCloseButton"
                        >
                            ×
                        </button>
                    </div>

                    <div className="modalContent">
                        <div className="modalSize">
                            Size: {(selectedCluster as any).nodeData?.freq || (selectedCluster as any).nodeData?.sentences?.length || 0} sentences
                        </div>
                        <div className="modalRepresentative">
                            Representative:
                        </div>
                        <p className="modalText">
                            {(selectedCluster as any).nodeData?.representative_sentence || ''}
                        </p>
                    </div>

                    <div className="modalSentencesTitle" style={{ marginTop: 16, marginBottom: 8 }}>
                        <strong>
                            Algorithm Fractions:
                        </strong>
                    </div>
                    <div className="modalContent" style={{ marginBottom: 16 }}>
                        {Array.from(selectedCluster.algorithmFractions.entries())
                            .sort((a, b) => b[1] - a[1])
                            .map(([algoId, fraction]) => (
                                <div key={algoId} style={{ marginBottom: 8 }}>
                                    <span style={{ fontWeight: 600 }}>{algoId}:</span>{' '}
                                    {(fraction * 100).toFixed(1)}%
                                </div>
                            ))}
                    </div>

                    <div className="modalSentencesTitle">
                        <strong>
                            Sentences ({(selectedCluster as any).nodeData?.sentences?.length || 0} total):
                        </strong>
                    </div>

                    <div className="sentencesContainer">
                        {((selectedCluster as any).nodeData?.sentences || []).map((sentence: any, index: number) => (
                            <div key={index} className="sentenceItem">
                                <div className="sentenceCount">
                                    Count: {sentence.count || 1}
                                </div>
                                <div className="sentenceText">
                                    {sentence.text}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}


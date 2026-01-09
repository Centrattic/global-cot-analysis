// @ts-nocheck
'use client'

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import * as d3 from 'd3'
import { FlowchartData, Node } from '@/types/flowchart'
import GraphizControls from './GraphizControls'
import GraphizLegend from './GraphizLegend'
import GraphizRolloutPanel from './GraphizRolloutPanel'
import './graph.css'

interface GraphizVisualizationProps {
    data: FlowchartData
    selectedRollouts: string[]
    datasetId?: string
    propertyCheckers?: string[]
    clusterSearchTerm?: string
    clusterSelectedSeq?: string[]
    strictClusterSelection?: boolean
}

export default function GraphizVisualization({ data, selectedRollouts, datasetId, propertyCheckers = [], clusterSearchTerm: propClusterSearchTerm = '', clusterSelectedSeq = [], strictClusterSelection = false }: GraphizVisualizationProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selectedNode, setSelectedNode] = useState<Node | null>(null)
    const [minClusterSize, setMinClusterSize] = useState<number>(0)
    const [maxEntropy, setMaxEntropy] = useState<number>(1) // Will be updated with actual max value
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
    const rolloutPanelContainerRef = useRef<HTMLDivElement | null>(null)
    const controlsRef = useRef<HTMLDivElement | null>(null)
    const [rolloutColorMap, setRolloutColorMap] = useState<Map<string, string>>(new Map())
    const [propertyFilter, setPropertyFilter] = useState<{ propertyName: string | null; filterMode: 'both' | number | 'delta' }>({ propertyName: null, filterMode: 'both' })
    const [directedEdges, setDirectedEdges] = useState<boolean>(false)
    const [enabledPropertyCheckers, setEnabledPropertyCheckers] = useState<Set<string>>(new Set())
    const [propertyCheckerValues, setPropertyCheckerValues] = useState<Map<string, any>>(new Map())
    const [balanceMode, setBalanceMode] = useState<'none' | 'equal_count' | 'equal_length'>('none')
    const [focusedClusters, setFocusedClusters] = useState<Set<string>>(new Set())
    const [panelHover, setPanelHover] = useState<{ rolloutId: string; index: number } | null>(null)
    const [focusMode, setFocusMode] = useState<boolean>(true)
    const [showAnswerLabel, setShowAnswerLabel] = useState<boolean>(false)
    const [showQuestionLabel, setShowQuestionLabel] = useState<boolean>(false)
    const [showSelfCheckingLabel, setShowSelfCheckingLabel] = useState<boolean>(false)
    const [skipQuestionRestatements, setSkipQuestionRestatements] = useState<boolean>(false)
    const [collapseAnswerCycles, setCollapseAnswerCycles] = useState<boolean>(false)
    const [collapseAllCyclesExceptQuestion, setCollapseAllCyclesExceptQuestion] = useState<boolean>(false)
    const [showDetails, setShowDetails] = useState<boolean>(false)
    const [balanceStats, setBalanceStats] = useState<any>(null)
    const [panelPosition, setPanelPosition] = useState<{ x: number; y: number }>(() => {
        // Set initial position immediately to prevent jumping
        if (typeof window !== 'undefined') {
            const legendTop = 20
            const legendMaxHeight = 400
            // Position panel to overlap with bottom portion of legend
            const panelStartY = legendTop + legendMaxHeight - 200 // Start 200px up from bottom of legend
            const panelWidth = 320
            const rightMargin = 20

            return {
                x: Math.max(rightMargin, window.innerWidth - panelWidth - rightMargin),
                y: panelStartY
            }
        }
        return { x: 20, y: 40 } // fallback for SSR
    })
    const [isDragging, setIsDragging] = useState<boolean>(false)
    const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
    const [panelHeight, setPanelHeight] = useState<number>(400)
    const [isResizing, setIsResizing] = useState<boolean>(false)
    const [resizeStartY, setResizeStartY] = useState<number>(0)
    const [resizeStartHeight, setResizeStartHeight] = useState<number>(0)
    // Rollout length filter state
    const [maxRolloutLength, setMaxRolloutLength] = useState<number>(0)
    const [selectedMaxRolloutLength, setSelectedMaxRolloutLength] = useState<number>(0)
    const [rolloutLengthGreaterMode, setRolloutLengthGreaterMode] = useState<boolean>(false)

    // Strict selection panel drag state
    const [strictPanelPosition, setStrictPanelPosition] = useState<{ x: number; y: number }>(() => {
        const top = (typeof window !== 'undefined' && controlsRef.current) ? (controlsRef.current.getBoundingClientRect().height + 16) : 56
        const right = 20
        const x = (typeof window !== 'undefined') ? Math.max(right, window.innerWidth - 340 - right) : 20
        return { x, y: top }
    })
    const [isStrictDragging, setIsStrictDragging] = useState<boolean>(false)
    const [strictDragOffset, setStrictDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 })
    const [showStrictPanel, setShowStrictPanel] = useState<boolean>(true)


    const getCanvasSize = () => {
        const isLarge = (datasetId && datasetId.includes('2000')) || (data?.nodes?.length || 0) > 800
        return isLarge ? { width: 2200, height: 1600, padding: 40 } : { width: 1200, height: 900, padding: 20 }
    }

    const getDefaultPanelPosition = () => {
        // Position panel on the right side, overlapping with bottom portion of legend
        // Legend is at right: 20px, top: 20px, max-height: 400px
        // Position panel to overlap with bottom portion of legend
        const legendTop = 20
        const legendMaxHeight = 400
        const panelStartY = legendTop + legendMaxHeight - 300 // Start 200px up from bottom of legend
        const panelWidth = 320
        const rightMargin = 20

        return {
            x: Math.max(rightMargin, (typeof window !== 'undefined' ? window.innerWidth : 1200) - panelWidth - rightMargin),
            y: panelStartY
        }
    }

    const isResponseNodeId = (id: string) => typeof id === 'string' && id.startsWith('response-')

    // Detect data format once
    const isOldFormat = data.nodes.length > 0 && data.nodes[0].cluster_id !== undefined
    const isNewFormat = !isOldFormat

    // Debug: log first few cluster IDs
    console.log('DEBUG: First 5 cluster IDs from data.nodes:', data.nodes.slice(0, 5).map(n => isOldFormat ? n.cluster_id : Object.keys(n)[0]))

    // Helper function to get cluster ID and node data from both formats
    const getNodeInfo = (node: any) => {
        if (isOldFormat) {
            // Old format: direct properties
            return { clusterId: node.cluster_id, nodeData: node }
        } else {
            // New format: objects with cluster keys
            const clusterKey = Object.keys(node)[0]
            return { clusterId: clusterKey, nodeData: node[clusterKey] }
        }
    }

    // Helper to compute the filtered rollout sequence used for hover darkening
    const getCompressedSequenceForRollout = (rolloutId: string): string[] => {
        const rolloutData = getRolloutData(rolloutId)
        let rolloutEdges: { node_a: string; node_b: string }[] = []
        if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
            rolloutEdges = rolloutData.edges || []
        } else if (Array.isArray(rolloutData)) {
            rolloutEdges = rolloutData as any
        }
        const normalizedEdges = rolloutEdges.map((edge: any) => ({
            node_a: normalizeNodeId(edge.node_a),
            node_b: normalizeNodeId(edge.node_b),
        }))
        const fullSequence: string[] = []
        normalizedEdges.forEach((e, i) => {
            if (i === 0) fullSequence.push(e.node_a)
            fullSequence.push(e.node_b)
        })
        const displayedIds = new Set(filteredNodes.map(n => {
            const { clusterId } = getNodeInfo(n)
            return clusterId
        }))
        displayedIds.add('START')
        const filteredSeq = fullSequence.filter(id => displayedIds.has(id))
        const dedupedSeq: string[] = []
        filteredSeq.forEach(id => {
            if (dedupedSeq.length === 0 || dedupedSeq[dedupedSeq.length - 1] !== id) {
                dedupedSeq.push(id)
            }
        })
        return dedupedSeq
    }

    // Helper function to normalize node IDs (handle cluster- prefix mismatch)
    const normalizeNodeId = (nodeId: string) => {
        // If the nodeId is just a number, add cluster- prefix
        if (/^\d+$/.test(nodeId)) {
            return `cluster-${nodeId}`
        }
        return nodeId
    }

    // Helper function to get rollout data from both formats
    const getRolloutData = (rolloutId: string) => {
        // Handle both "rollouts" (old) and "responses" (new) keys
        const responsesData = data.responses || data.rollouts

        if (Array.isArray(responsesData)) {
            if (isOldFormat) {
                // Old format: array with index property
                return responsesData.find((r: any) => r.index && r.index.toString() === rolloutId)
            } else {
                // New format: array of objects with rollout ID keys
                const rolloutObj = responsesData.find((x: any) => x[rolloutId])
                return rolloutObj ? rolloutObj[rolloutId] : null
            }
        } else {
            // Object format: direct keys (this is the current format)
            return responsesData[rolloutId]
        }
    }


    const isBalanced19v20 = !!(datasetId && datasetId.includes('19_vs_20_balanced'))
    const getRolloutCorrectness = (rid: string): boolean | undefined => {
        const rolloutData = getRolloutData(rid)
        return rolloutData ? rolloutData.correctness : undefined
    }

    const getRolloutPropertyValue = (rid: string, propertyName: string): any => {
        const rolloutData = getRolloutData(rid)
        return rolloutData ? rolloutData[propertyName] : undefined
    }

    // Deterministic helper: stable sort by (num_edges desc, rolloutId asc)
    const sortByEdgeLenDescStable = (ids: string[]) => {
        const withLen = ids.map(id => {
            const rd = getRolloutData(id)
            const edges = rd && rd.edges ? rd.edges.length : 0
            return { id, len: edges }
        })
        withLen.sort((a, b) => {
            if (b.len !== a.len) return b.len - a.len
            return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
        })
        return withLen
    }

    // Function to compute cluster size sum for a rollout (sum of num_rollouts or freq for clusters it visits)
    const getRolloutClusterSizeSum = (rolloutId: string): number => {
        const rolloutData = getRolloutData(rolloutId)
        let rolloutEdges: { node_a: string; node_b: string }[] = []
        if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
            rolloutEdges = rolloutData.edges || []
        } else if (Array.isArray(rolloutData)) {
            rolloutEdges = rolloutData as any
        }

        const normalizedEdges = rolloutEdges.map(edge => ({
            node_a: normalizeNodeId(edge.node_a),
            node_b: normalizeNodeId(edge.node_b)
        }))

        const visitedClusters = new Set<string>()
        normalizedEdges.forEach(edge => {
            if (!isResponseNodeId(edge.node_a) && edge.node_a !== 'START') {
                visitedClusters.add(edge.node_a)
            }
            if (!isResponseNodeId(edge.node_b) && edge.node_b !== 'START') {
                visitedClusters.add(edge.node_b)
            }
        })

        let total = 0
        visitedClusters.forEach(clusterId => {
            const node = data.nodes.find((n: any) => {
                const { clusterId: cid } = getNodeInfo(n)
                return cid === clusterId
            })
            if (node) {
                const { nodeData } = getNodeInfo(node)
                const clusterSize = nodeData.num_rollouts !== undefined ? nodeData.num_rollouts : nodeData.freq
                total += clusterSize
            }
        })
        return total
    }

    // Function to get balanced rollouts and details based on balanceMode when a property checker is enabled
    const getBalancedRolloutsAndDetails = (rollouts: string[]): { rollouts: string[]; details: any } => {
        if (enabledPropertyCheckers.size === 0 || balanceMode === 'none') return { rollouts, details: null }
        const enabledChecker = Array.from(enabledPropertyCheckers)[0]

        // Get unique values for the enabled property checker
        const values = new Set<any>()
        rollouts.forEach(rid => {
            const v = getRolloutPropertyValue(rid, enabledChecker)
            if (v !== undefined && v !== null) {
                values.add(v)
            }
        })
        const uniqueValues = Array.from(values).sort()
        if (uniqueValues.length < 2) return { rollouts, details: null }

        // Partition into groups by property value
        const groups = new Map<any, string[]>()
        uniqueValues.forEach(val => groups.set(val, []))
        rollouts.forEach(rid => {
            const v = getRolloutPropertyValue(rid, enabledChecker)
            if (v !== undefined && v !== null && groups.has(v)) {
                groups.get(v)!.push(rid)
            }
        })

        // Convert to arrays sorted by size (smallest first)
        const sortedGroups = Array.from(groups.entries()).sort((a, b) => a[1].length - b[1].length)
        const smallestGroup = sortedGroups[0]
        const largestGroup = sortedGroups[sortedGroups.length - 1]

        if (balanceMode === 'equal_count') {
            // Deterministic equal counts: keep smallest group fully, drop from larger groups
            const minCount = smallestGroup[1].length
            const keptSmallest = [...smallestGroup[1]].sort()

            // For each larger group, drop rollouts with largest cluster size sums
            const keptFromLarger: string[] = []
            sortedGroups.forEach(([val, groupRollouts]) => {
                if (val === smallestGroup[0]) return // Skip smallest group

                // Calculate cluster size sum for each rollout and sort by it (largest first)
                const withClusterSums = groupRollouts.map(rid => ({
                    id: rid,
                    clusterSum: getRolloutClusterSizeSum(rid)
                }))
                withClusterSums.sort((a, b) => {
                    if (b.clusterSum !== a.clusterSum) return b.clusterSum - a.clusterSum
                    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
                })

                // Keep only the first minCount (smallest cluster sums)
                const kept = withClusterSums.slice(-minCount).map(x => x.id).sort()
                keptFromLarger.push(...kept)
            })

            const chosen = [...keptSmallest, ...keptFromLarger]
            const getLen = (id: string) => {
                const rd = getRolloutData(id)
                return rd && rd.edges ? rd.edges.length : 0
            }

            // Build details with generic property values
            const finalCounts: any = {}
            sortedGroups.forEach(([val]) => {
                const kept = val === smallestGroup[0]
                    ? keptSmallest
                    : keptFromLarger.filter(rid => getRolloutPropertyValue(rid, enabledChecker) === val)
                finalCounts[`num_${val}`] = kept.length
                finalCounts[`len_${val}`] = kept.reduce((s, id) => s + getLen(id), 0)
            })

            const initialCounts: any = {}
            sortedGroups.forEach(([val, groupRollouts]) => {
                initialCounts[`num_${val}`] = groupRollouts.length
                initialCounts[`len_${val}`] = groupRollouts.reduce((s, id) => s + getLen(id), 0)
            })

            const details = {
                mode: 'equal_count',
                property: enabledChecker,
                initial: initialCounts,
                final: finalCounts
            }
            return { rollouts: chosen, details }
        }

        if (balanceMode === 'equal_length') {
            // Compute totals for each group
            const getLen = (id: string) => {
                const rd = getRolloutData(id)
                return rd && rd.edges ? rd.edges.length : 0
            }
            const groupLengths = new Map<any, number>()
            sortedGroups.forEach(([val, groupRollouts]) => {
                groupLengths.set(val, groupRollouts.reduce((acc, id) => acc + getLen(id), 0))
            })

            // Find smallest and largest groups by length
            let smallestVal = sortedGroups[0][0]
            let smallestLen = groupLengths.get(smallestVal)!
            let largestVal = sortedGroups[0][0]
            let largestLen = groupLengths.get(largestVal)!
            sortedGroups.forEach(([val]) => {
                const len = groupLengths.get(val)!
                if (len < smallestLen) {
                    smallestLen = len
                    smallestVal = val
                }
                if (len > largestLen) {
                    largestLen = len
                    largestVal = val
                }
            })

            // Keep smaller group fully
            const keptSmallest = [...sortedGroups.find(g => g[0] === smallestVal)![1]]
            const target = smallestLen
            const largerGroupRollouts = [...sortedGroups.find(g => g[0] === largestVal)![1]]

            // Sort larger group by cluster size sum (largest first), then drop until we reach target
            const withClusterSums = largerGroupRollouts.map(rid => ({
                id: rid,
                clusterSum: getRolloutClusterSizeSum(rid),
                len: getLen(rid)
            }))
            withClusterSums.sort((a, b) => {
                if (b.clusterSum !== a.clusterSum) return b.clusterSum - a.clusterSum
                if (b.len !== a.len) return b.len - a.len
                return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
            })

            let remainingTotal = withClusterSums.reduce((s, x) => s + x.len, 0)
            const keptLargerIds: string[] = []

            for (const item of withClusterSums) {
                if (remainingTotal <= target) {
                    keptLargerIds.push(item.id)
                } else {
                    remainingTotal -= item.len
                }
            }
            keptLargerIds.sort()

            // Also keep all other groups fully
            const keptOther: string[] = []
            sortedGroups.forEach(([val, groupRollouts]) => {
                if (val !== smallestVal && val !== largestVal) {
                    keptOther.push(...groupRollouts)
                }
            })

            const chosen = [...keptSmallest, ...keptLargerIds, ...keptOther]

            // Build details
            const finalCounts: any = {}
            const initialCounts: any = {}
            sortedGroups.forEach(([val, groupRollouts]) => {
                const kept = val === smallestVal
                    ? keptSmallest
                    : val === largestVal
                        ? keptLargerIds
                        : groupRollouts
                initialCounts[`len_${val}`] = groupLengths.get(val)
                initialCounts[`num_${val}`] = groupRollouts.length
                finalCounts[`len_${val}`] = kept.reduce((s, id) => s + getLen(id), 0)
                finalCounts[`num_${val}`] = kept.length
            })
            finalCounts.keep_smaller_side = smallestVal

            const details = {
                mode: 'equal_length',
                property: enabledChecker,
                initial: initialCounts,
                final: finalCounts
            }
            return { rollouts: chosen, details }
        }

        return { rollouts, details: null }
    }


    // Calculate cluster sizes (exclude START/response nodes) and filter
    const getClusterSizes = () => {
        const sizes = data.nodes
            .map(node => {
                const { clusterId, nodeData } = getNodeInfo(node)
                if (clusterId === 'START' || isResponseNodeId(clusterId)) return undefined
                return nodeData.freq
            })
            .filter((v: any) => typeof v === 'number') as number[]
        return {
            min: Math.min(...sizes),
            max: Math.max(...sizes),
            sizes
        }
    }

    // Calculate entropy values for clusters
    const getClusterEntropies = () => {
        const entropies = data.nodes
            .map(node => {
                const { clusterId, nodeData } = getNodeInfo(node)
                if (clusterId === 'START' || isResponseNodeId(clusterId)) return undefined
                return nodeData.entropy
            })
            .filter((v: any) => typeof v === 'number' && !isNaN(v)) as number[]

        if (entropies.length === 0) {
            return { min: 0, max: 1, entropies: [0, 1] }
        }

        const minEntropy = Math.min(...entropies)
        const maxEntropy = Math.max(...entropies)

        // If all entropies are the same, create a small range
        if (minEntropy === maxEntropy) {
            return {
                min: Math.max(0, minEntropy - 0.1),
                max: minEntropy + 0.1,
                entropies: [minEntropy]
            }
        }

        return {
            min: minEntropy,
            max: maxEntropy,
            entropies
        }
    }

    const { min: minSize, max: maxSize } = getClusterSizes()
    const { min: minEntropy, max: maxEntropyValue } = getClusterEntropies()


    // Better scaling function for cluster radius (logarithmic with minimum size)
    const getClusterRadius = (freq: number) => {
        const minRadius = 4
        const maxRadius = 30
        const scaled = Math.log(freq + 1) * 4
        return Math.max(minRadius, Math.min(maxRadius, scaled))
    }
    // Reset minClusterSize when data changes to ensure we show all clusters by default
    useEffect(() => {
        setMinClusterSize(0)
    }, [data])
    // Build available property checkers from props + data so newly added keys (e.g., single_algorithm) appear
    const availablePropertyCheckers = useMemo(() => {
        const set = new Set<string>(propertyCheckers || [])
        const responsesData: any = (data as any).responses || (data as any).rollouts || {}
        const entries: any[] = Array.isArray(responsesData)
            ? (isOldFormat ? responsesData : responsesData.map((o: any) => Object.values(o)[0]))
            : Object.values(responsesData)
        entries.forEach((rd) => {
            if (rd && typeof rd === 'object') {
                Object.keys(rd).forEach(k => {
                    if (k === 'edges') return
                    if (k === 'index' || k === 'seed') return
                    if (k === 'answer' || k === 'rollout_id') return
                    set.add(k)
                })
            }
        })
        // Don't add hardcoded checkers - only show what's in the config
        return Array.from(set).sort()
    }, [data, propertyCheckers, isOldFormat])

    // Set initial maxEntropy to the maximum entropy value from current data
    useEffect(() => {
        setMaxEntropy(maxEntropyValue)
    }, [maxEntropyValue])
    // Compute rollout length stats
    useEffect(() => {
        const responsesData: any = (data as any).responses || (data as any).rollouts || {}
        const ids: string[] = Array.isArray(responsesData)
            ? (isOldFormat ? responsesData.map((r: any) => String(r.index)) : responsesData.map((o: any) => Object.keys(o)[0]))
            : Object.keys(responsesData)
        let maxLen = 0
        ids.forEach((rid) => {
            const rd = getRolloutData(String(rid))
            const len = rd && rd.edges ? rd.edges.length : (Array.isArray(rd) ? rd.length : 0)
            if (len > maxLen) maxLen = len
        })
        setMaxRolloutLength(maxLen)
        setSelectedMaxRolloutLength(maxLen)
    }, [data])


    const filteredNodes = useMemo(() => {
        return data.nodes.filter(node => {
            const { clusterId, nodeData } = getNodeInfo(node)
            // Always include START and response nodes
            if (isResponseNodeId(clusterId) || clusterId === 'START') return true
            // Filter by size threshold
            const clusterSize = nodeData.freq || 0
            if (clusterSize < minClusterSize) return false
            // Filter by entropy
            if (!(nodeData.entropy <= maxEntropy)) return false
            // Filter by question restatements if enabled
            if (skipQuestionRestatements) {
                const label = nodeData?.label
                if (label === 'question') return false
            }
            return true
        })
    }, [data, minClusterSize, maxEntropy, skipQuestionRestatements])

    // Debug: log filtering results
    console.log('DEBUG: Size filtering - minClusterSize:', minClusterSize, 'maxEntropy:', maxEntropy, 'filteredNodes count:', filteredNodes.length, 'total nodes:', data.nodes.length)

    const getRolloutPathWithTexts = (rolloutId: string) => {
        let rolloutEdges: { node_a: string; node_b: string }[] = []
        const rolloutData = getRolloutData(rolloutId)
        if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
            rolloutEdges = rolloutData.edges || []
        } else if (Array.isArray(rolloutData)) {
            rolloutEdges = rolloutData as any
        }

        // Normalize node IDs in edges to match the new format
        const normalizedEdges = rolloutEdges.map((edge: any) => ({
            node_a: normalizeNodeId(edge.node_a),
            node_b: normalizeNodeId(edge.node_b),
            step_text_a: (edge as any).step_text_a,
            step_text_b: (edge as any).step_text_b,
        }))

        type PathItem = { id: string, text?: string, idx: number }
        const orig: PathItem[] = []
        normalizedEdges.forEach((e, i) => {
            if (i === 0) orig.push({ id: e.node_a, text: e.step_text_a, idx: orig.length })
            orig.push({ id: e.node_b, text: e.step_text_b, idx: orig.length })
        })
        // If the FIRST STEP AFTER START has no text, prefer node_a of the second edge
        if (orig.length > 1 && (orig[1].text == null || String(orig[1].text).trim() === "")) {
            if (normalizedEdges.length >= 2) {
                const second = normalizedEdges[1]
                if (second && second.step_text_a != null && String(second.step_text_a).trim() !== "") {
                    orig[1].text = second.step_text_a
                }
            }
        }
        const displayedIds = new Set(filteredNodes.map(n => {
            const { clusterId } = getNodeInfo(n)
            return clusterId
        }))
        displayedIds.add('START')
        const kept: PathItem[] = orig.filter(item => displayedIds.has(item.id))
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
        return { seq: out.map(x => x.id), texts: out.map(x => x.text) }
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

    // Load graph layout from flowchart data
    useEffect(() => {
        // Compute/populate positions independently of SVG mount so first load works
        positionsRef.current.clear()
        globalEdgeCountRef.current.clear()
        rolloutEdgeCountRef.current.clear()
        setLayoutError(null)

        // Use embedded graph layout from flowchart
        console.log('DEBUG: data.graph_layout exists:', !!data.graph_layout)
        console.log('DEBUG: data.graph_layout keys length:', data.graph_layout ? Object.keys(data.graph_layout).length : 0)
        console.log('DEBUG: data.graph_layout sample:', data.graph_layout ? Object.entries(data.graph_layout).slice(0, 3) : 'none')

        if (data.graph_layout && Object.keys(data.graph_layout).length > 0) {
            console.log('Using embedded graph layout from flowchart')
            console.log('DEBUG: First 5 layout entries:', Object.entries(data.graph_layout).slice(0, 5))
            console.log('DEBUG: First 5 node cluster_ids:', data.nodes.slice(0, 5).map(n => isOldFormat ? n.cluster_id : Object.keys(n)[0]))
            const map = positionsRef.current
            const entries: Array<[string, { x: number; y: number }]> = Object.entries(data.graph_layout).map(([id, p]: [string, any]) => {
                const x = typeof p?.x === 'number' ? p.x : parseFloat(String(p?.x ?? 0))
                const y = typeof p?.y === 'number' ? p.y : parseFloat(String(p?.y ?? 0))
                return [id, { x, y }]
            })
            // Detect if positions are absolute pixels; normalize to [0,1]
            const xs = entries.map(([, p]) => p.x)
            const ys = entries.map(([, p]) => p.y)
            const minX = Math.min(...xs)
            const maxX = Math.max(...xs)
            const minY = Math.min(...ys)
            const maxY = Math.max(...ys)
            const rangeX = Math.max(1e-6, maxX - minX)
            const rangeY = Math.max(1e-6, maxY - minY)
            const looksPixel = maxX > 1.5 || maxY > 1.5 || minX < 0 || minY < 0
            entries.forEach(([id, p]) => {
                const nx = looksPixel ? (p.x - minX) / rangeX : p.x
                const ny = looksPixel ? (p.y - minY) / rangeY : p.y
                map.set(id, { x: nx, y: ny })
            })
            setPositionsReady(v => v + 1)
            return
        }

        // Fallback: try to load from cache directory
        console.log('No embedded graph layout found, trying cache directory...')
        const cacheFilename = `${datasetId}_sfdp.json`
        fetch(`/api/graph/layout/${cacheFilename}`)
            .then(response => {
                if (response.ok) {
                    return response.json()
                }
                throw new Error(`Cache file not found: ${cacheFilename}`)
            })
            .then(layoutData => {
                console.log('Using cached graph layout from:', cacheFilename)
                const map = positionsRef.current
                Object.entries(layoutData).forEach(([id, pos]: [string, any]) => {
                    map.set(id, pos)
                })
                setLayoutError(null)
                setPositionsReady(v => v + 1)
            })
            .catch(error => {
                console.error('Failed to load cached graph layout:', error)
                setLayoutError('No graph layout data available')
                setPositionsReady(v => v + 1)
            })
    }, [data])

    // Clear focus when rollout selection changes
    useEffect(() => {
        if (focusedClusters.size > 0) {
            setFocusedClusters(new Set())
        }
    }, [selectedRollouts])

    // Reset panel position when a new rollout is selected
    useEffect(() => {
        if (panelRollout) {
            const defaultPos = getDefaultPanelPosition()
            setPanelPosition(defaultPos)
        }
    }, [panelRollout])

    // Handle drag events
    const handleMouseDown = useCallback((event: React.MouseEvent) => {
        if (event.target === event.currentTarget || (event.target as HTMLElement).classList.contains('panelHeader')) {
            setIsDragging(true)
            // Calculate offset from current panel position to mouse position
            setDragOffset({
                x: event.clientX - panelPosition.x,
                y: event.clientY - panelPosition.y
            })
            event.preventDefault()
        }
    }, [panelPosition])

    // Handle resize events
    const handleResizeMouseDown = useCallback((event: React.MouseEvent) => {
        setIsResizing(true)
        setResizeStartY(event.clientY)
        setResizeStartHeight(panelHeight)
        event.preventDefault()
        event.stopPropagation()
    }, [panelHeight])

    const handleStrictMouseDown = useCallback((event: React.MouseEvent) => {
        if (event.target === event.currentTarget || (event.target as HTMLElement).classList.contains('panelHeader')) {
            setIsStrictDragging(true)
            setStrictDragOffset({
                x: event.clientX - strictPanelPosition.x,
                y: event.clientY - strictPanelPosition.y
            })
            event.preventDefault()
        }
    }, [strictPanelPosition])

    const handleMouseMove = useCallback((event: MouseEvent) => {
        if (isDragging) {
            setPanelPosition({
                x: event.clientX - dragOffset.x,
                y: event.clientY - dragOffset.y
            })
        } else if (isResizing) {
            const deltaY = event.clientY - resizeStartY
            const newHeight = Math.max(200, Math.min(800, resizeStartHeight + deltaY))
            setPanelHeight(newHeight)
        } else if (isStrictDragging) {
            setStrictPanelPosition({
                x: event.clientX - strictDragOffset.x,
                y: event.clientY - strictDragOffset.y
            })
        }
    }, [isDragging, isResizing, isStrictDragging, dragOffset, strictDragOffset, resizeStartY, resizeStartHeight])

    const handleMouseUp = useCallback(() => {
        setIsDragging(false)
        setIsResizing(false)
        setIsStrictDragging(false)
    }, [])

    // Add global mouse event listeners for dragging and resizing
    useEffect(() => {
        if (isDragging || isResizing || isStrictDragging) {
            document.addEventListener('mousemove', handleMouseMove)
            document.addEventListener('mouseup', handleMouseUp)
            return () => {
                document.removeEventListener('mousemove', handleMouseMove)
                document.removeEventListener('mouseup', handleMouseUp)
            }
        }
    }, [isDragging, isResizing, isStrictDragging, handleMouseMove, handleMouseUp])

    // When strict selection toggles on or sequence changes, show the panel and reset initial position once
    useEffect(() => {
        if (strictClusterSelection && clusterSelectedSeq && clusterSelectedSeq.length > 0) {
            setShowStrictPanel(true)
            if (typeof window !== 'undefined') {
                const top = (controlsRef.current?.getBoundingClientRect()?.height || 0) + 16
                const right = 20
                const x = Math.max(right, window.innerWidth - 340 - right)
                setStrictPanelPosition({ x, y: top })
            }
        }
    }, [strictClusterSelection, clusterSelectedSeq])

    // Precompute response-node outcome for correctness coloring once per data change
    const globalResponseNodeOutcome = useMemo(() => {
        const responsesData: any = (data as any).responses || (data as any).rollouts || {}
        const ids = Array.isArray(responsesData)
            ? (isOldFormat ? responsesData.map((r: any) => String(r.index)) : responsesData.map((o: any) => Object.keys(o)[0]))
            : Object.keys(responsesData)
        const counts = new Map<string, { correct: number; incorrect: number }>()
        ids.forEach((rid) => {
            const rolloutData = getRolloutData(String(rid))
            let rolloutEdges: { node_a: string; node_b: string }[] = []
            if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) rolloutEdges = rolloutData.edges || []
            else if (Array.isArray(rolloutData)) rolloutEdges = rolloutData as any
            const seq: string[] = []
            rolloutEdges.forEach((e: any, i: number) => { if (i === 0) seq.push(normalizeNodeId(e.node_a)); seq.push(normalizeNodeId(e.node_b)) })
            const finalResp = [...seq].reverse().find(id => isResponseNodeId(id))
            const corr = getRolloutCorrectness(String(rid))
            if (finalResp && typeof corr === 'boolean') {
                const cur = counts.get(finalResp) || { correct: 0, incorrect: 0 }
                if (corr) cur.correct += 1; else cur.incorrect += 1
                counts.set(finalResp, cur)
            }
        })
        const out = new Map<string, 'correct' | 'incorrect'>()
        counts.forEach((c, nid) => { out.set(nid, c.correct > 0 ? 'correct' : 'incorrect') })
        return out
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
        // Build nodeMap early for label lookups (used by distillation/bridging)
        const nodeMap = new Map<string, any>()
        data.nodes.forEach(node => {
            const { clusterId, nodeData } = getNodeInfo(node)
            nodeMap.set(clusterId, nodeData)
        })

        // Helper function to check if a property is a multi-algorithm property
        const isMultiAlgorithmProperty = (propertyName: string | null): boolean => {
            return propertyName !== null && propertyName.includes('multi_algorithm')
        }
        // Helper to get the active multi-algorithm property name
        const getActiveMultiAlgorithmProperty = (): string | null => {
            if (propertyFilter.propertyName && isMultiAlgorithmProperty(propertyFilter.propertyName)) {
                return propertyFilter.propertyName
            }
            for (const checker of enabledPropertyCheckers) {
                if (isMultiAlgorithmProperty(checker)) {
                    return checker
                }
            }
            return null
        }
        const activeMultiAlgorithmProperty = getActiveMultiAlgorithmProperty()
        const isMultiAlgorithmActive = activeMultiAlgorithmProperty !== null

        // DEBUG: Log multi-algorithm activation
        console.log('[DEBUG] Multi-algorithm activation:', {
            isMultiAlgorithmActive,
            activeMultiAlgorithmProperty,
            enabledPropertyCheckers: Array.from(enabledPropertyCheckers),
            propertyFilterPropertyName: propertyFilter.propertyName
        })

        // Function to check if a cluster matches the search term
        const clusterMatchesSearch = (clusterId: string, nodeData: any, searchTerm: string): boolean => {
            if (!searchTerm || searchTerm.trim() === '') return false
            if (clusterId === 'START' || isResponseNodeId(clusterId)) return false

            const searchLower = searchTerm.toLowerCase().trim()
            const representative = nodeData?.representative_sentence || ''
            if (representative.toLowerCase().includes(searchLower)) return true

            const sentences = nodeData?.sentences || []
            for (const sent of sentences) {
                const text = sent?.text || sent || ''
                if (typeof text === 'string' && text.toLowerCase().includes(searchLower)) return true
            }
            return false
        }

        // Apply balancing if enabled
        const result = getBalancedRolloutsAndDetails(selectedRollouts)
        let processedRollouts = Array.isArray(result as any) ? (result as any) : result.rollouts
        // Apply rollout length filter (keep rollouts with length <= selectedMaxRolloutLength)
        const getLen = (id: string) => {
            const rd = getRolloutData(id)
            return rd && rd.edges ? rd.edges.length : (Array.isArray(rd) ? rd.length : 0)
        }
        processedRollouts = processedRollouts.filter(rid => {
            const len = getLen(String(rid))
            return rolloutLengthGreaterMode ? (len > selectedMaxRolloutLength) : (len <= selectedMaxRolloutLength)
        })
        setBalanceStats(result.details)

        // When showing multiple property classes, sort to draw in a consistent order (last class on top)
        // This helps with visibility when multiple classes overlap
        if (propertyFilter.propertyName && propertyFilter.filterMode === 'both' && enabledPropertyCheckers.has(propertyFilter.propertyName)) {
            const values = new Set<any>()
            processedRollouts.forEach(rid => {
                const value = getRolloutPropertyValue(rid, propertyFilter.propertyName!)
                if (value !== undefined && value !== null) {
                    values.add(value)
                }
            })
            const uniqueValues = Array.from(values).sort()
            const sorted = uniqueValues.map(value =>
                processedRollouts.filter(rid => getRolloutPropertyValue(rid, propertyFilter.propertyName!) === value)
            )
            processedRollouts = sorted.flat()
        }

        // Build paths for each rollout
        const newValidRollouts: string[] = []
        processedRollouts.forEach((rolloutId) => {
            const rolloutData = getRolloutData(rolloutId)
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                rolloutEdges = rolloutData.edges || []
            } else if (Array.isArray(rolloutData)) {
                rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
            }

            if (rolloutEdges.length > 0) {
                newValidRollouts.push(rolloutId)

                // Normalize node IDs in edges to match the new format
                const normalizedEdges = rolloutEdges.map(edge => ({
                    node_a: normalizeNodeId(edge.node_a),
                    node_b: normalizeNodeId(edge.node_b)
                }))

                edges.push(...normalizedEdges)

                const path = new Set<string>()
                normalizedEdges.forEach(edge => {
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

        const applyNodePositions = (zoomLevel: number = 1) => {
            const { width, height, padding } = getCanvasSize()

            // Calculate node scale - nodes get smaller as zoom increases
            const nodeScale = Math.max(0.05, 1 / zoomLevel)

            const nodesSel = g.selectAll<SVGGElement, any>('.clusters .cluster')
            nodesSel.attr('transform', (d: any) => {
                const { clusterId } = getNodeInfo(d)
                const p = nodePositions.get(clusterId)
                // Scale normalized positions (0-1) to canvas dimensions
                const tx = p ? padding + p.x * (width - 2 * padding) : 0
                const ty = p ? padding + p.y * (height - 2 * padding) : 0
                return `translate(${tx}, ${ty}) scale(${nodeScale})`
            })
            const activeSel = g.selectAll<SVGGElement, any>('.clusters-active .cluster')
            activeSel.attr('transform', (d: any) => {
                const { clusterId } = getNodeInfo(d)
                const p = nodePositions.get(clusterId)
                // Scale normalized positions (0-1) to canvas dimensions
                const tx = p ? padding + p.x * (width - 2 * padding) : 0
                const ty = p ? padding + p.y * (height - 2 * padding) : 0
                return `translate(${tx}, ${ty}) scale(${nodeScale})`
            })
        }

        const zoom = d3.zoom<SVGSVGElement, unknown>()
            .scaleExtent([0.1, 10])
            .on('zoom', (event) => {
                g.attr('transform', event.transform)
                applyNodePositions(event.transform.k)
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

        // Get unique values for a property checker across all valid rollouts
        const getPropertyCheckerUniqueValues = (propertyName: string, rollouts: string[] = newValidRollouts) => {
            if (isMultiAlgorithmProperty(propertyName)) {
                const algos = new Set<string>()
                rollouts.forEach(rid => {
                    let v = getRolloutPropertyValue(rid, propertyName)
                    if (typeof v === 'string') {
                        try { v = JSON.parse(v) } catch { }
                    }
                    if (Array.isArray(v)) {
                        for (let i = 0; i < v.length; i += 2) {
                            const token = v[i]
                            if (typeof token === 'string') algos.add(token)
                        }
                    } else if (typeof v === 'string') {
                        algos.add(v)
                    }
                })
                return Array.from(algos).sort()
            }
            const values = new Set<any>()
            rollouts.forEach(rid => {
                const value = getRolloutPropertyValue(rid, propertyName)
                if (value !== undefined && value !== null) {
                    values.add(value)
                }
            })
            return Array.from(values).sort()
        }

        // Get color for a property checker value
        const getPropertyCheckerColor = (value: any, propertyName: string) => {
            const uniqueValues = getPropertyCheckerUniqueValues(propertyName)
            const index = uniqueValues.indexOf(value)
            if (index === -1) return rolloutColors[0] // fallback color

            // For boolean values (like correctness), use specific colors
            if (propertyName === 'correctness') {
                return value === true ? '#10b981' : '#ef4444' // green for correct, red for incorrect
            }

            // For resampled property checker, use different colors for each prefix
            if (propertyName === 'resampled') {
                if (value === false) {
                    return '#6b7280' // gray for non-resampled
                }
                // For prefix values, use sequential colors
                return rolloutColors[index % rolloutColors.length]
            }

            // For other property checkers, use sequential colors
            return rolloutColors[index % rolloutColors.length]
        }

        // nodeMap already built above

        const defs = svg.append('defs')

        // Add glow filter for search highlighting
        const glowFilter = defs.append('filter')
            .attr('id', 'search-glow')
            .attr('x', '-50%')
            .attr('y', '-50%')
            .attr('width', '200%')
            .attr('height', '200%')
        glowFilter.append('feGaussianBlur')
            .attr('stdDeviation', '3')
            .attr('result', 'coloredBlur')
        glowFilter.append('feMerge')
            .append('feMergeNode')
            .attr('in', 'coloredBlur')
        glowFilter.append('feMerge')
            .append('feMergeNode')
            .attr('in', 'SourceGraphic')

        const defaultColorMap = new Map<string, string>()
        newValidRollouts.forEach((rid, idx) => defaultColorMap.set(rid, rolloutColors[idx % rolloutColors.length]))
        const correctIds = newValidRollouts.filter(rid => getRolloutCorrectness(rid) === true)
        const incorrectIds = newValidRollouts.filter(rid => getRolloutCorrectness(rid) === false)

        // Get unique values for property checker coloring
        const getPropertyCheckerValues = (propertyName: string) => {
            const values = new Set<any>()
            newValidRollouts.forEach(rid => {
                const value = getRolloutPropertyValue(rid, propertyName)
                values.add(value)
            })
            return Array.from(values)
        }

        const colorForRollout = (rid: string, fallbackIndex?: number) => {
            // Skip rollout-level coloring when multi-algorithm property is active
            // (edges/nodes will be colored by algorithm instead)
            if (isMultiAlgorithmActive && activeMultiAlgorithmProperty) {
                // Return a neutral fallback color that won't be used for algorithm-specific coloring
                return rolloutColors[0]
            }

            // Use property checker coloring if any property checker is enabled
            if (enabledPropertyCheckers.size > 0) {
                // Use the first enabled property checker for coloring
                const enabledChecker = Array.from(enabledPropertyCheckers)[0]
                const propertyValue = getRolloutPropertyValue(rid, enabledChecker)
                if (propertyValue !== undefined && propertyValue !== null) {
                    return getPropertyCheckerColor(propertyValue, enabledChecker)
                }
            }

            // Default rollout coloring: every rollout gets a different color
            if (typeof fallbackIndex === 'number') {
                return rolloutColors[fallbackIndex % rolloutColors.length]
            }
            return defaultColorMap.get(rid) || rolloutColors[0]
        }

        const darkenColor = (hex: string, amount = 0.5) => {
            const clamp = (v: number) => Math.max(0, Math.min(255, v))
            const num = parseInt(hex.replace('#', ''), 16)
            const r = (num >> 16) & 0xff
            const g = (num >> 8) & 0xff
            const b = num & 0xff
            const dr = clamp(Math.round(r * (1 - amount)))
            const dg = clamp(Math.round(g * (1 - amount)))
            const db = clamp(Math.round(b * (1 - amount)))
            const toHex = (v: number) => v.toString(16).padStart(2, '0')
            return `#${toHex(dr)}${toHex(dg)}${toHex(db)}`
        }

        const colorFromSeed = (seed: string) => {
            let hash = 0
            for (let i = 0; i < seed.length; i++) hash = (hash * 31 + seed.charCodeAt(i)) >>> 0
            const hue = hash % 360
            return `hsl(${hue}, 70%, 55%)`
        }

        // Build and persist color map for legend and pies to stay consistent
        const newColorMap = new Map<string, string>()
        newValidRollouts.forEach((rid, idx) => {
            newColorMap.set(rid, colorForRollout(rid, idx))
        })
        setRolloutColorMap(newColorMap)

        // Petal coloring maps (only when a single rollout is selected and focus is active)
        const petalEdgeColor = new Map<string, string>()
        const petalNodeColor = new Map<string, string>()

        // Precompute symmetric-difference sets for nodes and edges (for delta mode)
        // Maps property value -> Set of nodes/edges seen by rollouts with that value
        const nodeSeenByClass = new Map<any, Set<string>>()
        const edgeSeenByClass = new Map<any, Set<string>>()
        const classColors = new Map<any, string>()

        if (propertyFilter.propertyName && propertyFilter.filterMode === 'delta' && enabledPropertyCheckers.has(propertyFilter.propertyName)) {
            const uniqueValues = getPropertyCheckerUniqueValues(propertyFilter.propertyName, newValidRollouts)
            uniqueValues.forEach((value) => {
                nodeSeenByClass.set(value, new Set())
                edgeSeenByClass.set(value, new Set())
                classColors.set(value, getPropertyCheckerColor(value, propertyFilter.propertyName!))
            })

            newValidRollouts.forEach((rid) => {
                const rolloutData = getRolloutData(rid)
                let rolloutEdges: { node_a: string; node_b: string }[] = []
                if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                    rolloutEdges = rolloutData.edges || []
                } else if (Array.isArray(rolloutData)) {
                    rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
                }
                const normalizedEdges = rolloutEdges.map(edge => ({
                    node_a: normalizeNodeId(edge.node_a),
                    node_b: normalizeNodeId(edge.node_b)
                }))
                const fullSequence: string[] = []
                normalizedEdges.forEach((e, i) => { if (i === 0) fullSequence.push(e.node_a); fullSequence.push(e.node_b) })
                const displayedIds = new Set(filteredNodes.map(n => getNodeInfo(n).clusterId))
                displayedIds.add('START')
                const compressedSeq = fullSequence.filter(id => displayedIds.has(id))

                if (isMultiAlgorithmProperty(propertyFilter.propertyName)) {
                    // Build per-position algorithm mapping for this rollout
                    let raw = getRolloutPropertyValue(String(rid), propertyFilter.propertyName!)
                    if (typeof raw === 'string') { try { raw = JSON.parse(raw) } catch { } }
                    const seqForAlgo = compressedSeq.filter(id => id !== 'START')
                    const L = seqForAlgo.length
                    const posToAlgo: (string | null)[] = new Array(L).fill(null)
                    const valueArr: any[] = Array.isArray(raw) ? raw : []
                    if (valueArr.length > 0 && typeof valueArr[0] === 'string') {
                        const algos: string[] = []
                        const cuts: number[] = []
                        for (let i = 0; i < valueArr.length; i++) {
                            const v = valueArr[i]
                            if (i % 2 === 0 && typeof v === 'string') algos.push(v)
                            if (i % 2 === 1 && typeof v === 'number') cuts.push(v)
                        }
                        let start = 1
                        for (let i = 0; i < algos.length; i++) {
                            const algo = algos[i]
                            const endExclusive = (i < cuts.length) ? cuts[i] : (L + 1)
                            const end = Math.max(start, Math.min(L, endExclusive - 1))
                            if (start <= end) {
                                for (let p = start; p <= end; p++) posToAlgo[p - 1] = algo
                            }
                            start = endExclusive
                        }
                    }
                    for (let i = 0; i < seqForAlgo.length; i++) {
                        const algo = posToAlgo[i]
                        if (!algo || !nodeSeenByClass.has(algo)) continue
                        nodeSeenByClass.get(algo)!.add(seqForAlgo[i])
                        if (i < seqForAlgo.length - 1) {
                            const ek = `${seqForAlgo[i]}->${seqForAlgo[i + 1]}`
                            edgeSeenByClass.get(algo)!.add(ek)
                        }
                    }
                } else {
                    // Default behavior: group by rollout-level property value
                    const propertyValue = getRolloutPropertyValue(rid, propertyFilter.propertyName!)
                    if (propertyValue === undefined || propertyValue === null || !nodeSeenByClass.has(propertyValue)) return
                    const nodeSet = nodeSeenByClass.get(propertyValue)!
                    const edgeSet = edgeSeenByClass.get(propertyValue)!
                    for (let i = 0; i < compressedSeq.length; i++) {
                        const nid = compressedSeq[i]
                        nodeSet.add(nid)
                        if (i < compressedSeq.length - 1) {
                            const ek = `${compressedSeq[i]}->${compressedSeq[i + 1]}`
                            edgeSet.add(ek)
                        }
                    }
                }
            })
        }

        // Compute symmetric difference: nodes/edges that appear in ONLY one class
        const nodeOnlyInClass = new Map<any, Set<string>>()
        const edgeOnlyInClass = new Map<any, Set<string>>()

        if (propertyFilter.propertyName && propertyFilter.filterMode === 'delta') {
            nodeSeenByClass.forEach((nodeSet, classValue) => {
                const exclusiveNodes = new Set<string>()
                nodeSet.forEach(nid => {
                    let appearsInOtherClass = false
                    nodeSeenByClass.forEach((otherSet, otherValue) => {
                        if (otherValue !== classValue && otherSet.has(nid)) {
                            appearsInOtherClass = true
                        }
                    })
                    if (!appearsInOtherClass) {
                        exclusiveNodes.add(nid)
                    }
                })
                nodeOnlyInClass.set(classValue, exclusiveNodes)
            })

            edgeSeenByClass.forEach((edgeSet, classValue) => {
                const exclusiveEdges = new Set<string>()
                edgeSet.forEach(ek => {
                    let appearsInOtherClass = false
                    edgeSeenByClass.forEach((otherSet, otherValue) => {
                        if (otherValue !== classValue && otherSet.has(ek)) {
                            appearsInOtherClass = true
                        }
                    })
                    if (!appearsInOtherClass) {
                        exclusiveEdges.add(ek)
                    }
                })
                edgeOnlyInClass.set(classValue, exclusiveEdges)
            })
        }

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
        // Special multi-algorithm coloring mode helpers
        // Get algorithm colors - use same colors as legend for consistency
        const getAlgorithmColor = (algoId: string) => {
            if (algoId === '0') return rolloutColors[0]
            if (algoId === '1') return rolloutColors[1]
            // For multi-algorithm properties, get unique values and assign consistent colors
            if (activeMultiAlgorithmProperty) {
                const uniqueAlgos = getPropertyCheckerUniqueValues(activeMultiAlgorithmProperty, newValidRollouts)
                const algoIndex = uniqueAlgos.indexOf(algoId)
                if (algoIndex >= 0) {
                    // Use the same color palette as the legend (rolloutColors)
                    return rolloutColors[algoIndex % rolloutColors.length]
                }
            }
            // Fallback to seed-based color if not found in unique values
            return colorFromSeed(`algo-${algoId}`)
        }
        // rolloutId -> nodeId -> algoId (first occurrence along path)
        const nodeAlgoByRollout = new Map<string, Map<string, string>>()
        const displayedPathNodeSet = new Set<string>()
        const incomingByNode = new Map<string, Map<string, number>>()
        // Track which response node corresponds to correct vs incorrect outcomes based on rollouts
        const responseNodeCounts = new Map<string, { correct: number; incorrect: number }>()
        const addIncoming = (nodeId: string, rolloutKey: string, weight: number) => {
            if (!incomingByNode.has(nodeId)) incomingByNode.set(nodeId, new Map())
            const m = incomingByNode.get(nodeId)!
            m.set(rolloutKey, (m.get(rolloutKey) || 0) + weight)
        }
        // Precompute darkened segments and nodes for hovered rollout up to hovered step
        const darkEdgesByRollout = new Map<string, Set<string>>()
        const darkNodesByRollout = new Map<string, Set<string>>()
        if (panelHover) {
            const rid = String(panelHover.rolloutId)
            const seq = getCompressedSequenceForRollout(rid)
            const cutoff = Math.max(0, Math.min(panelHover.index, seq.length - 1))
            const eSet = new Set<string>()
            const nSet = new Set<string>()
            for (let i = 0; i <= cutoff; i++) nSet.add(seq[i])
            for (let i = 0; i < cutoff; i++) eSet.add(`${seq[i]}->${seq[i + 1]}`)
            darkEdgesByRollout.set(rid, eSet)
            darkNodesByRollout.set(rid, nSet)
        }

        // Property filter predicate
        const passesPropertyFilter = (rid: string) => {
            if (!propertyFilter.propertyName || propertyFilter.filterMode === 'both' || propertyFilter.filterMode === 'delta') return true
            if (isMultiAlgorithmProperty(propertyFilter.propertyName)) return true // handle at segment/pie level
            const propertyValue = getRolloutPropertyValue(rid, propertyFilter.propertyName)
            const uniqueValues = getPropertyCheckerUniqueValues(propertyFilter.propertyName, newValidRollouts)
            const targetValue = uniqueValues[propertyFilter.filterMode as number]
            return propertyValue === targetValue
        }

        newValidRollouts.forEach((rolloutId, index) => {
            if (!passesPropertyFilter(String(rolloutId))) return
            const rolloutData = getRolloutData(rolloutId)
            let rolloutEdges: { node_a: string; node_b: string }[] = []

            if (rolloutData && typeof rolloutData === 'object' && 'edges' in rolloutData) {
                rolloutEdges = rolloutData.edges || []
            } else if (Array.isArray(rolloutData)) {
                rolloutEdges = rolloutData as { node_a: string; node_b: string }[]
            }

            // Normalize node IDs in edges to match the new format
            const normalizedEdges = rolloutEdges.map(edge => ({
                node_a: normalizeNodeId(edge.node_a),
                node_b: normalizeNodeId(edge.node_b)
            }))

            // Build ordered path from full edge list, then compress by filter
            const nextMap = new Map<string, string[]>()
            const edgeOrderIndex = new Map<string, number>()
            normalizedEdges.forEach((e, i) => {
                if (!nextMap.has(e.node_a)) nextMap.set(e.node_a, [])
                nextMap.get(e.node_a)!.push(e.node_b)
                edgeOrderIndex.set(`${e.node_a}->${e.node_b}`, i)
                // Defer incoming counts to when segments are actually drawn to avoid ghost nodes
            })
            const incomingFull = new Set<string>()
            normalizedEdges.forEach(e => incomingFull.add(e.node_b))
            const allSourcesFull = new Set(normalizedEdges.map(e => e.node_a))
            let startNodeId = Array.from(allSourcesFull).find(a => !incomingFull.has(a)) || (normalizedEdges[0] ? normalizedEdges[0].node_a : undefined)

            // Build sequence and draw paths
            const fullSequence: string[] = []
            normalizedEdges.forEach((e, i) => {
                if (i === 0) fullSequence.push(e.node_a)
                fullSequence.push(e.node_b)
            })

            const displayedIds = new Set(filteredNodes.map(n => {
                const { clusterId } = getNodeInfo(n)
                return clusterId
            }))
            displayedIds.add('START')
            let compressedSeq = fullSequence.filter(id => displayedIds.has(id))
            // Helper label checks
            const isAnswerLabel = (cid: string) => {
                const nd = nodeMap.get(cid)
                const label = nd ? nd.label : undefined
                return label === 'answer' || (typeof label === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(label))
            }
            if (skipQuestionRestatements) {
                // Remove consecutive runs of question-labeled clusters and bridge A->B
                const isQuestion = (cid: string) => {
                    const nd = nodeMap.get(cid)
                    return nd && nd.label === 'question'
                }
                const bridged: string[] = []
                for (let i = 0; i < compressedSeq.length; i++) {
                    const cur = compressedSeq[i]
                    if (!isQuestion(cur)) {
                        bridged.push(cur)
                        continue
                    }
                    // we're at Q run; find end
                    let j = i
                    while (j + 1 < compressedSeq.length && isQuestion(compressedSeq[j + 1])) j++
                    const prev = bridged.length > 0 ? bridged[bridged.length - 1] : null
                    const next = (j + 1 < compressedSeq.length) ? compressedSeq[j + 1] : null
                    if (prev && next && prev !== next) {
                        // Add edge prev->next via drawSegment later by inserting next if needed; avoid duplicate consecutive
                        if (bridged.length === 0 || bridged[bridged.length - 1] !== prev) bridged.push(prev)
                        if (bridged[bridged.length - 1] !== next) bridged.push(next)
                    }
                    i = j
                }
                if (bridged.length > 0) compressedSeq = bridged
            }
            const shouldCollapseAll = collapseAllCyclesExceptQuestion
            if (collapseAnswerCycles || shouldCollapseAll) {
                const isEligible = (cid: string) => {
                    const nd = nodeMap.get(cid)
                    const lbl = nd ? nd.label : undefined
                    const isQuestion = lbl === 'question'
                    const isAnswer = isAnswerLabel(cid)
                    return shouldCollapseAll ? !isQuestion : isAnswer
                }
                const out: string[] = []
                let i = 0
                while (i < compressedSeq.length) {
                    const cur = compressedSeq[i]
                    out.push(cur)
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
                            for (let t = i + 1; t <= firstOther; t++) out.push(compressedSeq[t])
                        }
                        const after = (lastSame + 1 < compressedSeq.length) ? compressedSeq[lastSame + 1] : null
                        if (after && isResponseNodeId(after)) {
                            if (out[out.length - 1] !== after) out.push(after)
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
                out.forEach(id => { if (dedup.length === 0 || dedup[dedup.length - 1] !== id) dedup.push(id) })
                compressedSeq = dedup
            }
            // Remove consecutive duplicates to avoid self-edges when gaps are skipped
            const dedupedSeq: string[] = []
            compressedSeq.forEach(id => {
                if (dedupedSeq.length === 0 || dedupedSeq[dedupedSeq.length - 1] !== id) {
                    dedupedSeq.push(id)
                }
            })

            // Build multi-algorithm mapping from positions to algorithm (1-based positions excluding START)
            if (isMultiAlgorithmActive && activeMultiAlgorithmProperty) {
                let raw = getRolloutPropertyValue(String(rolloutId), activeMultiAlgorithmProperty)
                console.log(`[DEBUG] Rollout ${rolloutId}: raw property value:`, raw, 'type:', typeof raw)

                if (typeof raw === 'string') {
                    try {
                        raw = JSON.parse(raw)
                        console.log(`[DEBUG] Rollout ${rolloutId}: parsed to:`, raw)
                    } catch (e) {
                        console.error(`[DEBUG] Rollout ${rolloutId}: JSON parse failed:`, e)
                    }
                }

                const pathForAlgo = dedupedSeq.filter(cid => cid !== 'START')
                const L = pathForAlgo.length
                console.log(`[DEBUG] Rollout ${rolloutId}: pathForAlgo length:`, L, 'pathForAlgo:', pathForAlgo.slice(0, 10))

                const posToAlgo: (string | null)[] = new Array(L).fill(null)
                const valueArr: any[] = Array.isArray(raw) ? raw : []
                console.log(`[DEBUG] Rollout ${rolloutId}: valueArr length:`, valueArr.length, 'isArray:', Array.isArray(raw), 'valueArr:', valueArr)

                if (valueArr.length > 0 && typeof valueArr[0] === 'string') {
                    const algos: string[] = []
                    const cuts: number[] = []
                    for (let i = 0; i < valueArr.length; i++) {
                        const v = valueArr[i]
                        if (i % 2 === 0 && typeof v === 'string') algos.push(v)
                        if (i % 2 === 1 && typeof v === 'number') cuts.push(v)
                    }
                    console.log(`[DEBUG] Rollout ${rolloutId}: extracted algos:`, algos, 'cuts:', cuts)

                    let start = 1
                    for (let i = 0; i < algos.length; i++) {
                        const algo = algos[i]
                        const endExclusive = (i < cuts.length) ? cuts[i] : (L + 1)
                        const end = Math.max(start, Math.min(L, endExclusive - 1))
                        console.log(`[DEBUG] Rollout ${rolloutId}: algo ${algo} from pos ${start} to ${end} (exclusive: ${endExclusive})`)
                        if (start <= end) {
                            for (let p = start; p <= end; p++) posToAlgo[p - 1] = algo
                        }
                        start = endExclusive
                    }
                    console.log(`[DEBUG] Rollout ${rolloutId}: posToAlgo:`, posToAlgo.filter((x, i) => x !== null).map((x, i) => `[${i + 1}]=${x}`).join(', '))
                } else {
                    console.warn(`[DEBUG] Rollout ${rolloutId}: Skipping algorithm mapping - valueArr.length:`, valueArr.length, 'first type:', typeof valueArr[0])
                }

                const map = new Map<string, string>()
                pathForAlgo.forEach((cid, idx) => {
                    const algo = posToAlgo[idx]
                    if (algo && !map.has(cid)) {
                        map.set(cid, algo)
                        console.log(`[DEBUG] Rollout ${rolloutId}: mapped cluster ${cid} at idx ${idx} -> algo ${algo}`)
                    }
                })
                console.log(`[DEBUG] Rollout ${rolloutId}: Final mapping size:`, map.size, 'entries:', Array.from(map.entries()))
                nodeAlgoByRollout.set(String(rolloutId), map)
            }

            // Track final response node outcome for correctness coloring
            const finalResponseId = [...dedupedSeq].reverse().find(id => isResponseNodeId(id))
            const corr = getRolloutCorrectness(String(rolloutId))
            if (finalResponseId && typeof corr === 'boolean') {
                const cur = responseNodeCounts.get(finalResponseId) || { correct: 0, incorrect: 0 }
                if (corr) cur.correct += 1
                else cur.incorrect += 1
                responseNodeCounts.set(finalResponseId, cur)
            }

            const drawSegment = (u: string, v: string) => {
                if (u === v) return
                const a = nodePositions.get(u)
                const b = nodePositions.get(v)
                if (!a || !b) return
                if (propertyFilter.propertyName && propertyFilter.filterMode === 'delta') {
                    // Only draw edges whose endpoints are exclusive nodes (appear in only one class)
                    let isExclusiveU = false
                    let isExclusiveV = false
                    nodeOnlyInClass.forEach((nodeSet) => {
                        if (nodeSet.has(u)) isExclusiveU = true
                        if (nodeSet.has(v)) isExclusiveV = true
                    })
                    if (!isExclusiveU || !isExclusiveV) return
                }
                const { width, height, padding } = getCanvasSize()
                const ax = padding + a.x * (width - 2 * padding)
                const ay = padding + a.y * (height - 2 * padding)
                const bx = padding + b.x * (width - 2 * padding)
                const by = padding + b.y * (height - 2 * padding)
                const weight = 1
                const base = colorForRollout(rolloutId, index)
                const edgeKey = `${u}->${v}`
                // When multi-algorithm is active, start with neutral color (edges will be colored by algorithm)
                let stroke = isMultiAlgorithmActive ? '#6b7280' : base

                // Algorithm variables - need to be in function scope for access later
                let au: string | null = null
                let av: string | null = null

                // DEBUG: Log initial stroke color
                if (isMultiAlgorithmActive && index === 0) {
                    console.log(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: initial stroke = ${stroke}, base = ${base}, isMultiAlgorithmActive = ${isMultiAlgorithmActive}`)
                }
                // Force START edges to first algorithm color only in 'Both' view; allow class filters otherwise
                let startForced = false
                if (isMultiAlgorithmActive && activeMultiAlgorithmProperty && (u === 'START' || v === 'START') && propertyFilter.filterMode === 'both') {
                    let raw = getRolloutPropertyValue(String(rolloutId), activeMultiAlgorithmProperty)
                    if (typeof raw === 'string') {
                        try { raw = JSON.parse(raw) } catch { }
                    }
                    let firstAlgo: string | null = null
                    if (Array.isArray(raw) && typeof raw[0] === 'string') firstAlgo = raw[0]
                    else if (typeof raw === 'string') firstAlgo = raw
                    if (firstAlgo) {
                        stroke = getAlgorithmColor(firstAlgo)
                        startForced = true
                    }
                }
                if (!startForced && propertyFilter.propertyName && propertyFilter.filterMode === 'delta') {
                    // Find which class this edge belongs to (if any)
                    let foundClass: any = null
                    edgeOnlyInClass.forEach((edgeSet, classValue) => {
                        if (edgeSet.has(edgeKey)) {
                            foundClass = classValue
                        }
                    })
                    if (foundClass !== null) {
                        stroke = classColors.get(foundClass) || base
                    } else {
                        return // skip non-exclusive edges
                    }
                } else if (!startForced) {
                    // Multi-algorithm per-edge stroke override (half/half if crossing algorithms)
                    if (isMultiAlgorithmActive) {
                        // Use first occurrence positions along the deduped path (excluding START)
                        const pathForAlgo = dedupedSeq.filter(cid => cid !== 'START')
                        const firstPos = new Map<string, number>()
                        pathForAlgo.forEach((cid, i) => { if (!firstPos.has(cid)) firstPos.set(cid, i + 1) })
                        const getAlgoAt = (cid: string): string | null => {
                            const rid = String(rolloutId)
                            const map = nodeAlgoByRollout.get(rid)
                            if (!map) {
                                console.warn(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: No algorithm map found`)
                                return null
                            }
                            const algo = map.get(cid) || null
                            if (!algo && (index === 0 || index === 1 || index === 2)) {
                                console.warn(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: Cluster ${cid} not in map. Map has:`, Array.from(map.keys()).slice(0, 10))
                            }
                            return algo
                        }
                        au = getAlgoAt(u)
                        av = getAlgoAt(v)

                        // DEBUG: Log algorithm detection for first few edges - ALWAYS log first edges
                        if (index === 0 || index === 1 || index === 2) {
                            console.log(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId} (index=${index}): au=${au}, av=${av}, map exists: ${!!nodeAlgoByRollout.get(String(rolloutId))}`)
                            if (nodeAlgoByRollout.get(String(rolloutId))) {
                                const map = nodeAlgoByRollout.get(String(rolloutId))!
                                console.log(`[DEBUG] Edge ${u}->${v}: map size=${map.size}, u in map=${map.has(u)}, v in map=${map.has(v)}, map keys sample:`, Array.from(map.keys()).slice(0, 5))
                            }
                        }
                        // If user selected a specific class, draw only that class segments
                        if (propertyFilter.filterMode !== 'both' && typeof propertyFilter.filterMode === 'number' && activeMultiAlgorithmProperty) {
                            const classes = getPropertyCheckerUniqueValues(activeMultiAlgorithmProperty, newValidRollouts)
                            const target = classes[propertyFilter.filterMode as number]
                            if (!(au === target && av === target)) return
                        }
                        if (au && av) {
                            const algoColor = getAlgorithmColor(au)
                            if (au === av) {
                                stroke = algoColor
                                if (index < 3) {
                                    console.log(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: Same algo ${au}, color = ${algoColor}`)
                                }
                            } else {
                                const gid = `ma-grad-${rolloutId}-${u}-${v}`
                                const existing = svg.select(`#${gid}`)
                                if (existing.empty()) {
                                    const lg = defs.append('linearGradient')
                                        .attr('id', gid)
                                        .attr('gradientUnits', 'userSpaceOnUse')
                                        .attr('x1', ax)
                                        .attr('y1', ay)
                                        .attr('x2', bx)
                                        .attr('y2', by)
                                    const cu = getAlgorithmColor(au)
                                    const cv = getAlgorithmColor(av)
                                    lg.append('stop').attr('offset', '0%').attr('stop-color', cu)
                                    lg.append('stop').attr('offset', '50%').attr('stop-color', cu)
                                    lg.append('stop').attr('offset', '50%').attr('stop-color', cv)
                                    lg.append('stop').attr('offset', '100%').attr('stop-color', cv)
                                } else {
                                    existing
                                        .attr('gradientUnits', 'userSpaceOnUse')
                                        .attr('x1', ax)
                                        .attr('y1', ay)
                                        .attr('x2', bx)
                                        .attr('y2', by)
                                }
                                stroke = `url(#${gid})`
                                if (index < 3) {
                                    console.log(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: Gradient ${au}->${av}, color ${getAlgorithmColor(au)}->${getAlgorithmColor(av)}`)
                                }
                            }
                        } else {
                            // DEBUG: Log when algorithms aren't detected
                            if (index < 3) {
                                console.warn(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: Algorithms NOT detected! au=${au}, av=${av}. Stroke remains ${stroke}`)
                            }
                        }
                        // If algorithms weren't detected, stroke remains gray (set above)
                    }
                    const petalStroke = (selectedRollouts.length === 1 && focusedClusters.size > 0) ? petalEdgeColor.get(edgeKey) : undefined
                    if (petalStroke) {
                        stroke = petalStroke
                    } else if (isMultiAlgorithmActive) {
                        // When multi-algorithm is active, don't darken using base (rollout color)
                        // Only darken if we have an algorithm color
                        const darkSet = darkEdgesByRollout.get(String(rolloutId))
                        if (darkSet && darkSet.has(edgeKey) && (au || av)) {
                            // Darken the current algorithm color, not the base rollout color
                            stroke = darkenColor(stroke, 0.35)
                            if (index < 3) {
                                console.log(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId}: Darkened algorithm color to ${stroke}`)
                            }
                        }
                    } else {
                        // When multi-algorithm is NOT active, use base color darkening
                        const darkSet = darkEdgesByRollout.get(String(rolloutId))
                        if (darkSet && darkSet.has(edgeKey)) {
                            stroke = darkenColor(base, 0.35)
                        }
                    }

                    // DEBUG: Final stroke color for first few edges - ALWAYS log
                    if (isMultiAlgorithmActive && (index === 0 || index === 1 || index === 2)) {
                        console.log(`[DEBUG] Edge ${u}->${v} for rollout ${rolloutId} (index=${index}): FINAL stroke = ${stroke}, au=${au}, av=${av}`)
                    }
                }
                const pathSel = trajectoryGroup.append('path')
                    .attr('d', `M ${ax} ${ay} L ${bx} ${by}`)
                    .attr('fill', 'none')
                    .attr('stroke', stroke)
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
                // Accumulate incoming counts only for edges actually drawn
                addIncoming(v, String(rolloutId), 1)
                if (directedEdges) {
                    const midx = (ax + bx) / 2
                    const midy = (ay + by) / 2
                    trajectoryGroup.append('path')
                        .attr('d', `M ${ax} ${ay} L ${midx} ${midy} L ${bx} ${by}`)
                        .attr('fill', 'none')
                        .attr('stroke', 'transparent')
                        .attr('stroke-width', Math.min(6, 3 + Math.log2(1 + weight) * 1.5))
                        .attr('vector-effect', 'non-scaling-stroke')
                        .attr('opacity', 1)
                        .attr('pointer-events', 'none')
                        .attr('marker-mid', `url(#arrow-${rolloutId})`)
                }
            }

            if (focusMode && focusedClusters.size > 0) {
                // Build intervals [start, end] for each focused cluster within compressedSeq
                const intervals: Array<[number, number]> = []
                focusedClusters.forEach(focusId => {
                    const startIdx = compressedSeq.indexOf(focusId)
                    const endIdx = compressedSeq.lastIndexOf(focusId)
                    if (startIdx >= 0 && endIdx > startIdx) {
                        intervals.push([startIdx, endIdx])
                    }
                })
                // When single rollout and focus, also compute petal colors between consecutive occurrences
                if (focusMode && selectedRollouts.length === 1) {
                    focusedClusters.forEach(focusId => {
                        const idxs: number[] = []
                        compressedSeq.forEach((cid, i) => { if (cid === focusId) idxs.push(i) })
                        for (let k = 0; k < idxs.length - 1; k++) {
                            const s = idxs[k]
                            const e = idxs[k + 1]
                            if (e > s) {
                                const seed = `${focusId}-${s}-${e}`
                                const color = colorFromSeed(seed)
                                for (let i = s; i < e; i++) petalEdgeColor.set(`${compressedSeq[i]}->${compressedSeq[i + 1]}`, color)
                                for (let i = s; i <= e; i++) {
                                    const nid = compressedSeq[i]
                                    if (nid !== focusId) petalNodeColor.set(nid, color)
                                }
                            }
                        }
                    })
                }
                // Draw segments that fall fully within any interval; collect nodes for highlighting
                for (let i = 0; i < compressedSeq.length - 1; i++) {
                    const inInterval = intervals.some(([s, e]) => i >= s && (i + 1) <= e)
                    if (!inInterval) continue
                    const u = compressedSeq[i]
                    const v = compressedSeq[i + 1]
                    displayedPathNodeSet.add(u)
                    displayedPathNodeSet.add(v)
                    drawSegment(u, v)
                }
            } else {
                // No focus: draw the full deduped path in order, adding nodes only for drawn segments
                for (let i = 0; i < dedupedSeq.length - 1; i++) {
                    const u = dedupedSeq[i]
                    const v = dedupedSeq[i + 1]
                    displayedPathNodeSet.add(u)
                    displayedPathNodeSet.add(v)
                    drawSegment(u, v)
                }
            }

            // drawSegment defined above

            // drawing handled above depending on focus

            // Also draw START -> first displayed node only if rollout lacks a START edge (fallback)
            const hasStartEdge = normalizedEdges.some(e => e.node_a === 'START')
            const startPos = nodePositions.get('START')
            if (propertyFilter.filterMode !== 'delta' && (!focusMode || focusedClusters.size === 0) && !hasStartEdge && startPos) {
                const displayedIdsForStart = new Set(filteredNodes.map(n => {
                    const { clusterId } = getNodeInfo(n)
                    return clusterId
                }))
                const firstDisplayedNodeId = (function () {
                    const seq: string[] = []
                    normalizedEdges.forEach((e, i) => {
                        if (i === 0) seq.push(e.node_a)
                        seq.push(e.node_b)
                    })
                    const compressed = seq.filter(id => displayedIdsForStart.has(id))
                    return compressed.find(id => id !== 'START')
                })()

                if (firstDisplayedNodeId) {
                    const firstPos = nodePositions.get(firstDisplayedNodeId)
                    if (firstPos) {
                        const countsForRollout = rolloutEdgeCountRef.current.get(String(rolloutId)) || new Map()
                        const weight = countsForRollout.get('START->' + firstDisplayedNodeId) || 1
                        const { width, height, padding } = getCanvasSize()
                        const startX = padding + startPos.x * (width - 2 * padding)
                        const startY = padding + startPos.y * (height - 2 * padding)
                        const firstX = padding + firstPos.x * (width - 2 * padding)
                        const firstY = padding + firstPos.y * (height - 2 * padding)
                        const lineStart = d3.line<{ x: number; y: number }>()
                            .x(d => d.x)
                            .y(d => d.y)
                        // If multi_algorithm class is selected, only draw if first step's algorithm matches selected class
                        if (isMultiAlgorithmActive && activeMultiAlgorithmProperty && propertyFilter.filterMode !== 'both' && typeof propertyFilter.filterMode === 'number') {
                            let raw = getRolloutPropertyValue(String(rolloutId), activeMultiAlgorithmProperty)
                            if (typeof raw === 'string') { try { raw = JSON.parse(raw) } catch { } }
                            let firstAlgo: string | null = null
                            if (Array.isArray(raw) && typeof raw[0] === 'string') firstAlgo = raw[0]
                            else if (typeof raw === 'string') firstAlgo = raw
                            const classes = getPropertyCheckerUniqueValues(activeMultiAlgorithmProperty, newValidRollouts)
                            const target = classes[propertyFilter.filterMode as number]
                            if (!firstAlgo || firstAlgo !== target) {
                                // Skip drawing START edge if doesn't match selected class
                                return
                            }
                        }
                        const path = trajectoryGroup.append('path')
                            .attr('d', lineStart([{ x: startX, y: startY }, { x: firstX, y: firstY }]))
                            .attr('fill', 'none')
                            .attr('stroke', colorForRollout(rolloutId, index))
                            .attr('stroke-width', Math.min(6, 3 + Math.log2(1 + weight) * 1.5))
                            .attr('vector-effect', 'non-scaling-stroke')
                            .attr('opacity', 0.7)
                            .attr('class', `trajectory trajectory-start-${rolloutId}`)
                        if (directedEdges) {
                            const midx = (startX + firstX) / 2
                            const midy = (startY + firstY) / 2
                            trajectoryGroup.append('path')
                                .attr('d', `M ${startX} ${startY} L ${midx} ${midy} L ${firstX} ${firstY}`)
                                .attr('fill', 'none')
                                .attr('stroke', 'transparent')
                                .attr('stroke-width', Math.min(6, 3 + Math.log2(1 + weight) * 1.5))
                                .attr('vector-effect', 'non-scaling-stroke')
                                .attr('opacity', 1)
                                .attr('pointer-events', 'none')
                                .attr('marker-mid', `url(#arrow-${rolloutId})`)
                        }
                        addIncoming(firstDisplayedNodeId, String(rolloutId), weight)
                    }
                }
            }

            // note: polyline for entire chain removed; segments are drawn individually above
        })

        // Strict cluster selection overlay edges (drawn in specified order)
        if (strictClusterSelection && (clusterSelectedSeq && clusterSelectedSeq.length >= 2)) {
            const overlay = g.append('g').attr('class', 'cluster-selection-overlay')
            // Ensure overlay edges sit above other trajectories
            if ((overlay as any).raise) (overlay as any).raise()
            const { width, height, padding } = getCanvasSize()
            const displayedIds = new Set(filteredNodes.map(n => getNodeInfo(n).clusterId))
            for (let i = 0; i < clusterSelectedSeq.length - 1; i++) {
                const u = clusterSelectedSeq[i]
                const v = clusterSelectedSeq[i + 1]
                const pu = nodePositions.get(u)
                const pv = nodePositions.get(v)
                if (!pu || !pv) continue
                if (!displayedIds.has(u) || !displayedIds.has(v)) continue
                const ux = padding + pu.x * (width - 2 * padding)
                const uy = padding + pu.y * (height - 2 * padding)
                const vx = padding + pv.x * (width - 2 * padding)
                const vy = padding + pv.y * (height - 2 * padding)
                overlay.append('path')
                    .attr('d', `M ${ux} ${uy} L ${vx} ${vy}`)
                    .attr('fill', 'none')
                    .attr('stroke', '#000000')
                    .attr('stroke-width', 10)
                    .attr('vector-effect', 'non-scaling-stroke')
                    .attr('opacity', 1)
                    .attr('stroke-opacity', 1)
                    .attr('stroke-linecap', 'square')
                    .attr('stroke-linejoin', 'miter')
                    .attr('stroke-miterlimit', 10)
                    .attr('shape-rendering', 'crispEdges')
                    .style('mix-blend-mode', 'normal')
                    .attr('pointer-events', 'none')
            }
        }

        // Decide per-response node correctness label globally across ALL rollouts (computed above)

        const nodeGroup = g.append('g').attr('class', 'clusters')
        // In delta mode, only display exclusive nodes; omit START
        // Otherwise, include all filtered nodes plus START
        const startNode: any = isOldFormat
            ? { cluster_id: 'START', freq: 0, representative_sentence: '', sentences: [] }
            : { 'START': { freq: 0, representative_sentence: '', sentences: [] } }
        const displayedNodes = (propertyFilter.propertyName && propertyFilter.filterMode === 'delta')
            ? filteredNodes.filter(n => {
                const { clusterId } = getNodeInfo(n)
                // Check if node appears in any exclusive class
                let isExclusive = false
                nodeOnlyInClass.forEach((nodeSet) => {
                    if (nodeSet.has(clusterId)) isExclusive = true
                })
                return isExclusive
            })
            : [...filteredNodes, startNode]

        // Find all matching cluster IDs for search highlighting (after displayedNodes is defined)
        const matchingClusterIds = new Set<string>()
        if (propClusterSearchTerm.trim()) {
            displayedNodes.forEach(node => {
                const { clusterId, nodeData } = getNodeInfo(node)
                if (clusterMatchesSearch(clusterId, nodeData, propClusterSearchTerm)) {
                    matchingClusterIds.add(clusterId)
                }
            })
        }
        // Selected clusters set (cluster-selection mode)
        const selectedClusterIds = new Set<string>(clusterSelectedSeq || [])
        const nodes = nodeGroup.selectAll('.cluster')
            .data(displayedNodes)
            .enter()
            .append('g')
            .attr('class', 'cluster')
            .attr('transform', d => {
                const { clusterId } = getNodeInfo(d)
                const p = nodePositions.get(clusterId)
                if (!p) {
                    console.log('DEBUG: No position found for clusterId:', clusterId)
                }
                // Scale normalized positions (0-1) to canvas dimensions
                const { width, height, padding } = getCanvasSize()
                const tx = p ? padding + p.x * (width - 2 * padding) : 0
                const ty = p ? padding + p.y * (height - 2 * padding) : 0
                return `translate(${tx}, ${ty})`
            })
            .style('cursor', 'pointer')
            .on('click', (event, d) => {
                const { clusterId, nodeData } = getNodeInfo(d)
                setSelectedNode({ ...nodeData, cluster_id: clusterId })
                if (focusMode) {
                    setFocusedClusters(prev => {
                        const next = new Set(prev)
                        if (next.has(clusterId)) {
                            next.delete(clusterId)
                        } else {
                            next.add(clusterId)
                        }
                        return next
                    })
                }
            })
            .on('mouseenter', function (event, d) {
                const { clusterId } = getNodeInfo(d)
                setHighlightClusterId(clusterId)
            })
            .on('mouseleave', function (event, d) {
                setHighlightClusterId(null)
            })

        nodes.append('circle')
            .attr('r', d => {
                const { clusterId, nodeData } = getNodeInfo(d)
                return (clusterId === 'START' || isResponseNodeId(clusterId)) ? 20 : getClusterRadius(nodeData.freq)
            })
            .attr('fill', d => {
                const { clusterId } = getNodeInfo(d)
                if (clusterId === 'START') return '#000000'
                if (isResponseNodeId(clusterId)) {
                    const outcome = globalResponseNodeOutcome.get(clusterId)
                    if (outcome === 'correct') return '#059669' // darker green
                    // Default any non-correct (including unknown) to darker red
                    return '#dc2626'
                }
                // Label-based coloring
                const nodeData = nodeMap.get(clusterId)
                const label = nodeData ? nodeData.label : undefined
                const isNumericAnswer = typeof label === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(label)
                if ((label === 'answer' || isNumericAnswer) && showAnswerLabel) return '#8b5cf6'
                if (label === 'question' && showQuestionLabel) return '#FFD700'
                if (label === 'self-checking' && showSelfCheckingLabel) return '#f59e0b'
                if (propertyFilter.propertyName && propertyFilter.filterMode === 'delta') {
                    // Find which class this node belongs to (if any)
                    let foundClass: any = null
                    nodeOnlyInClass.forEach((nodeSet, classValue) => {
                        if (nodeSet.has(clusterId)) {
                            foundClass = classValue
                        }
                    })
                    if (foundClass !== null) {
                        return classColors.get(foundClass) || '#d1d5db'
                    }
                }
                // Petal node colors (single rollout + focus) take precedence; focus node stays original
                if (selectedRollouts.length === 1 && focusedClusters.size > 0) {
                    if (!focusedClusters.has(clusterId)) {
                        const c = petalNodeColor.get(clusterId)
                        if (c) return c
                    }
                }
                // darken hovered rollout nodes up to hovered step
                for (const rid of newValidRollouts) {
                    const dn = darkNodesByRollout.get(String(rid))
                    if (dn && dn.has(clusterId)) return darkenColor(colorForRollout(rid), 0.35)
                }
                if (focusedClusters.size > 0) {
                    return displayedPathNodeSet.has(clusterId) ? '#ffffff' : '#d1d5db'
                }
                const counts = incomingByNode.get(clusterId) || new Map()
                let total = 0
                newValidRollouts.forEach(rid => { total += (counts.get(String(rid)) || 0) })
                return total === 0 ? '#d1d5db' : '#ffffff'
            })
            .attr('stroke', d => {
                const { clusterId } = getNodeInfo(d)
                if (selectedClusterIds.size > 0 && selectedClusterIds.has(clusterId)) return '#000000'
                return matchingClusterIds.has(clusterId) ? '#fbbf24' : '#ffffff' // yellow highlight for matches
            })
            .attr('stroke-width', d => {
                const { clusterId } = getNodeInfo(d)
                if (selectedClusterIds.size > 0 && selectedClusterIds.has(clusterId)) return 4
                return matchingClusterIds.has(clusterId) ? 4 : 2
            })
            .attr('vector-effect', 'non-scaling-stroke')
            .attr('opacity', 0.95)
            .attr('filter', d => {
                const { clusterId } = getNodeInfo(d)
                return matchingClusterIds.has(clusterId) ? 'url(#search-glow)' : 'none'
            })

        // Pie slices for nodes with incoming counts
        nodes.each(function (d: any) {
            const { clusterId, nodeData } = getNodeInfo(d)
            if (clusterId === 'START' || isResponseNodeId(clusterId)) return
            // If label overrides are active for this node, skip pies so label color remains dominant
            const nodeLabel = (nodeData && (nodeData as any).label) ? (nodeData as any).label : undefined
            const isNumericAnswer = typeof nodeLabel === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(nodeLabel)
            if (((nodeLabel === 'answer' || isNumericAnswer) && showAnswerLabel) || (nodeLabel === 'question' && showQuestionLabel) || (nodeLabel === 'self-checking' && showSelfCheckingLabel)) {
                return
            }
            const counts = incomingByNode.get(clusterId) || new Map()
            let total = 0
            newValidRollouts.forEach(rid => { total += (counts.get(String(rid)) || 0) })
            if (total === 0) return
            const group = d3.select(this)
            const r = getClusterRadius(nodeData.freq)
            let dataSlices: Array<{ key: string; value: number; color: string }>
            if (isMultiAlgorithmActive) {
                // Aggregate counts by algorithm at this node across selected rollouts
                const algoTotals = new Map<string, number>()
                newValidRollouts.forEach(rid => {
                    const id = String(rid)
                    const w = counts.get(id) || 0
                    if (w <= 0) return
                    const map = nodeAlgoByRollout.get(id)
                    const algo = map ? map.get(clusterId) : null
                    // Ignore segments with no resolved algorithm (can happen after collapsing)
                    if (!algo) return
                    const key = algo
                    // If a specific class is selected, only count that class
                    if (propertyFilter.filterMode !== 'both' && typeof propertyFilter.filterMode === 'number' && activeMultiAlgorithmProperty) {
                        const classes = getPropertyCheckerUniqueValues(activeMultiAlgorithmProperty, newValidRollouts)
                        const target = classes[propertyFilter.filterMode as number]
                        if (key === target) algoTotals.set(key, (algoTotals.get(key) || 0) + w)
                    } else {
                        algoTotals.set(key, (algoTotals.get(key) || 0) + w)
                    }
                })
                dataSlices = Array.from(algoTotals.entries()).map(([algo, value]) => ({ key: algo, value, color: getAlgorithmColor(algo) }))
            } else {
                dataSlices = newValidRollouts.map((rid, idx) => ({ key: String(rid), value: counts.get(String(rid)) || 0, color: colorForRollout(String(rid), idx) }))
            }
            const pieGen = d3.pie<any>().value((x: any) => x.value).sort(null)
            const arcs = pieGen(dataSlices as any)
            const arcGen = d3.arc<any>().innerRadius(0).outerRadius(r)
            const hoveredRid = panelHover ? String(panelHover.rolloutId) : null
            const hoveredNodeDark = hoveredRid ? (darkNodesByRollout.get(hoveredRid)?.has(clusterId) || false) : false
            arcs.forEach((a: any) => {
                if (a.data.value <= 0) return
                const sliceColor = (!isMultiAlgorithmActive && hoveredNodeDark && a.data.key === hoveredRid) ? darkenColor(a.data.color, 0.5) : a.data.color
                group.append('path')
                    .attr('d', arcGen(a) as any)
                    .attr('fill', sliceColor)
                    .attr('opacity', 0.95)
            })
        })

        nodes.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('font-size', d => {
                const { clusterId, nodeData } = getNodeInfo(d)
                if (clusterId === 'START') return 10
                if (isResponseNodeId(clusterId)) return 12
                return nodeData.freq >= 5 ? Math.min(12, Math.max(8, getClusterRadius(nodeData.freq) * 0.4)) : 0
            })
            .attr('font-weight', '600')
            .attr('fill', d => {
                const { clusterId } = getNodeInfo(d)
                return matchingClusterIds.has(clusterId) ? '#fbbf24' : '#ffffff' // yellow text for matches
            })
            .attr('text-shadow', d => {
                const { clusterId } = getNodeInfo(d)
                return matchingClusterIds.has(clusterId) ? '2px 2px 4px rgba(0,0,0,0.9)' : '1px 1px 2px rgba(0,0,0,0.7)'
            })
            .text(d => {
                const { clusterId, nodeData } = getNodeInfo(d)
                if (clusterId === 'START') return 'START'
                if (isResponseNodeId(clusterId)) {
                    // Extract just the answer from response node ID (e.g., "response-19" -> "19")
                    return clusterId.replace('response-', '')
                }
                return nodeData.freq >= 5 ? nodeData.freq.toString() : ''
            })

        // Layering: gray nodes (bottom) < trajectories (middle) < active nodes (top)
        // 1) Move trajectories above the initial clusters layer
        trajectoryGroup.raise()
        // 2) Reparent active nodes into a new top layer so they stay above trajectories
        const activeLayer = g.append('g').attr('class', 'clusters-active')
        nodes.filter(function (d: any) {
            const { clusterId } = getNodeInfo(d)
            if (focusMode && focusedClusters.size > 0) {
                return displayedPathNodeSet.has(clusterId)
            }
            if (clusterId === 'START') return true
            const counts = incomingByNode.get(clusterId) || new Map()
            let total = 0
            newValidRollouts.forEach(rid => { total += (counts.get(String(rid)) || 0) })
            return total > 0
        }).each(function () {
            const parent = activeLayer.node()
            if (parent) parent.appendChild(this as any)
        })

        // ensure hover highlighting applies to entire group, not just circle
        nodes.on('mouseenter.highlight', function (event, d: any) {
            const { clusterId, nodeData } = getNodeInfo(d)
            setHighlightClusterId(clusterId)
            d3.select(this).select('circle')
                .attr('stroke', '#000000')
                .attr('stroke-width', 5)
                .attr('opacity', 1)
            // try immediate scroll for rollout panel
            if (rolloutPanelRef.current) {
                const row = rolloutPanelRef.current.querySelector(`#rollout-row-${clusterId}`) as HTMLElement | null
                if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
            }
        })
        nodes.on('mouseleave.highlight', function () {
            setHighlightClusterId(null)
            const grp = d3.select(this)
            const datum: any = (grp as any).datum()
            const { nodeData } = getNodeInfo(datum)
            const label = (nodeData && (nodeData as any).label) ? (nodeData as any).label : undefined
            const circle = grp.select('circle')
            circle
                .attr('stroke', function () {
                    const d: any = (grp as any).datum()
                    const { clusterId } = getNodeInfo(d)
                    const inSelected = (clusterSelectedSeq && clusterSelectedSeq.length > 0) && (new Set(clusterSelectedSeq)).has(clusterId)
                    if (inSelected) return '#000000'
                    const isNumericAnswer = typeof label === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(label)
                    if ((label === 'answer' || isNumericAnswer) && showAnswerLabel) return '#8b5cf6'
                    if (label === 'question' && showQuestionLabel) return '#FFD700'
                    if (label === 'self-checking' && showSelfCheckingLabel) return '#f59e0b'
                    return '#ffffff'
                })
                .attr('stroke-width', function () {
                    const d: any = (grp as any).datum()
                    const { clusterId } = getNodeInfo(d)
                    const inSelected = (clusterSelectedSeq && clusterSelectedSeq.length > 0) && (new Set(clusterSelectedSeq)).has(clusterId)
                    if (inSelected) return 4
                    return (((((label === 'answer') || (typeof label === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(label))) && showAnswerLabel) || (label === 'question' && showQuestionLabel) || (label === 'self-checking' && showSelfCheckingLabel)) ? 3 : 2)
                })
                .attr('opacity', 0.95)
        })

        // Apply initial node positions
        const initialZoom = currentTransform ? currentTransform.k : 1
        applyNodePositions(initialZoom)

    }, [data, selectedRollouts, minClusterSize, maxEntropy, positionsReady, propertyFilter, directedEdges, enabledPropertyCheckers, balanceMode, focusMode, focusedClusters, panelHover, showAnswerLabel, showQuestionLabel, skipQuestionRestatements, collapseAnswerCycles, collapseAllCyclesExceptQuestion, propClusterSearchTerm, clusterSelectedSeq, strictClusterSelection, selectedMaxRolloutLength, rolloutLengthGreaterMode])

    // Bidirectional hover sync: update node styles and auto-scroll rollout panel
    useEffect(() => {
        if (!svgRef.current) return
        const svg = d3.select(svgRef.current)
        const k = currentTransform ? currentTransform.k : 1
        const groups = svg.selectAll('.clusters .cluster, .clusters-active .cluster') as any

        // Calculate node scale using the same logic as in applyNodePositions
        const nodeScale = Math.max(0.05, 1 / k)

        const selectedClusterIds = new Set<string>(clusterSelectedSeq || [])
        groups.each(function (d: any) {
            const group = d3.select(this)
            const { clusterId } = getNodeInfo(d)
            const p = positionsRef.current.get(clusterId)
            const hover = !!(highlightClusterId && d && clusterId === highlightClusterId)
            const hoverScale = hover ? 1.6 : 1
            // Scale normalized positions (0-1) to canvas dimensions
            const { width, height, padding } = getCanvasSize()
            const tx = p ? padding + p.x * (width - 2 * padding) : 0
            const ty = p ? padding + p.y * (height - 2 * padding) : 0
            group.attr('transform', `translate(${tx}, ${ty}) scale(${nodeScale * hoverScale})`)
            const circle = group.select('circle')
            if (!circle.empty()) {
                if (hover) {
                    circle.attr('stroke', '#000000').attr('stroke-width', 5).attr('opacity', 1)
                } else {
                    const inSelected = selectedClusterIds.size > 0 && selectedClusterIds.has(clusterId)
                    circle.attr('stroke', inSelected ? '#000000' : '#ffffff').attr('stroke-width', inSelected ? 4 : 2).attr('opacity', 0.95)
                }
            }
        })
        if (panelHover && rolloutPanelRef.current) {
            const container = rolloutPanelRef.current
            const doScroll = () => {
                const row = container.querySelector(`#rollout-row-${panelHover.rolloutId ? '' : ''}rollout-row-${highlightClusterId}-${panelHover.index}`) as HTMLElement | null
                if (row) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
            }
            if (typeof requestAnimationFrame !== 'undefined') {
                requestAnimationFrame(doScroll)
            } else {
                setTimeout(doScroll, 0)
            }
        }
    }, [panelHover, panelRollout, minClusterSize, maxEntropy, currentTransform, highlightClusterId, showAnswerLabel, showQuestionLabel, showSelfCheckingLabel, skipQuestionRestatements, clusterSelectedSeq])

    if (selectedRollouts.length === 0) {
        return (
            <div className="emptyState">
                <div className="emptyStateText">
                    Select rollouts to see the cluster trajectory visualization
                </div>
            </div>
        )
    }

    return (
        <div className="container">
            {/* Controls */}
            <GraphizControls
                minClusterSize={minClusterSize}
                maxClusterSize={maxSize}
                filteredNodesCount={filteredNodes.length}
                onMinClusterSizeChange={setMinClusterSize}
                maxEntropy={maxEntropy}
                minEntropy={minEntropy}
                maxEntropyData={maxEntropyValue}
                minEntropyData={minEntropy}
                onMaxEntropyChange={setMaxEntropy}
                directedEdges={directedEdges}
                onDirectedEdgesChange={setDirectedEdges}
                propertyCheckers={availablePropertyCheckers}
                enabledPropertyCheckers={enabledPropertyCheckers}
                onPropertyCheckerChange={(checker, enabled) => {
                    if (enabled) {
                        setEnabledPropertyCheckers(new Set([checker]))
                        // Automatically set property filter to use this checker
                        setPropertyFilter({ propertyName: checker, filterMode: 'both' })
                    } else {
                        setEnabledPropertyCheckers(new Set())
                        setPropertyFilter({ propertyName: null, filterMode: 'both' })
                        setBalanceMode('none') // reset when none selected
                    }
                }}
                balanceMode={balanceMode}
                onBalanceModeChange={(mode) => {
                    setBalanceMode(mode)
                }}
                hasFocus={focusedClusters.size > 0}
                onClearFocus={() => setFocusedClusters(new Set())}
                focusedClusterIds={Array.from(focusedClusters)}
                onClearSingleFocus={(cid) => {
                    setFocusedClusters(prev => {
                        const next = new Set(prev)
                        next.delete(cid)
                        return next
                    })
                }}
                propertyFilter={propertyFilter}
                onPropertyFilterChange={setPropertyFilter}
                enabledPropertyCheckers={enabledPropertyCheckers}
                getRolloutPropertyValue={getRolloutPropertyValue}
                validRollouts={validRollouts}
                focusMode={focusMode}
                onFocusModeChange={(val) => {
                    setFocusMode(val)
                    if (!val) {
                        setFocusedClusters(new Set())
                    }
                }}
                containerRef={controlsRef}
                showAnswerLabel={showAnswerLabel}
                showQuestionLabel={showQuestionLabel}
                showSelfCheckingLabel={showSelfCheckingLabel}
                onShowAnswerLabelChange={setShowAnswerLabel}
                onShowQuestionLabelChange={setShowQuestionLabel}
                onShowSelfCheckingLabelChange={setShowSelfCheckingLabel}
                skipQuestionRestatements={skipQuestionRestatements}
                onSkipQuestionRestatementsChange={setSkipQuestionRestatements}
                collapseAnswerCycles={collapseAnswerCycles}
                onCollapseAnswerCyclesChange={setCollapseAnswerCycles}
                collapseAllCyclesExceptQuestion={collapseAllCyclesExceptQuestion}
                onCollapseAllCyclesExceptQuestionChange={setCollapseAllCyclesExceptQuestion}
                maxRolloutLength={maxRolloutLength}
                selectedMaxRolloutLength={selectedMaxRolloutLength}
                onSelectedMaxRolloutLengthChange={setSelectedMaxRolloutLength}
                rolloutLengthGreaterMode={rolloutLengthGreaterMode}
                onRolloutLengthModeChange={setRolloutLengthGreaterMode}
            />

            {/* Details toggle under controls */}
            {balanceMode !== 'none' && enabledPropertyCheckers.size > 0 && (
                <div style={{ position: 'absolute', right: 20, top: (controlsRef.current?.getBoundingClientRect()?.height || 0) + 8, zIndex: 5 }}>
                    <details open={showDetails} onToggle={(e) => setShowDetails((e.target as any).open)}>
                        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Details</summary>
                        {balanceStats && (
                            <div style={{ background: '#111827', color: '#e5e7eb', padding: 8, borderRadius: 6, minWidth: 280 }}>
                                <div>Mode: {balanceStats.mode}</div>
                                {balanceStats.property && <div>Property: {balanceStats.property}</div>}
                                {balanceStats.initial && (
                                    <div style={{ marginTop: 6 }}>
                                        <div style={{ fontWeight: 600 }}>Initial</div>
                                        {Object.entries(balanceStats.initial).map(([key, value]) => {
                                            if (key.startsWith('len_')) {
                                                const classVal = key.replace('len_', '')
                                                return <div key={key}>L_{classVal}: {value}</div>
                                            } else if (key.startsWith('num_')) {
                                                const classVal = key.replace('num_', '')
                                                return <div key={key}># {classVal}: {value}</div>
                                            }
                                            return null
                                        })}
                                    </div>
                                )}
                                {balanceStats.final && (
                                    <div style={{ marginTop: 6 }}>
                                        <div style={{ fontWeight: 600 }}>Final</div>
                                        {Object.entries(balanceStats.final).map(([key, value]) => {
                                            if (key === 'keep_smaller_side') {
                                                return <div key={key}>Kept smaller side: {value}</div>
                                            } else if (key.startsWith('len_')) {
                                                const classVal = key.replace('len_', '')
                                                return <div key={key}>L_{classVal}: {value}</div>
                                            } else if (key.startsWith('num_')) {
                                                const classVal = key.replace('num_', '')
                                                return <div key={key}># {classVal}: {value}</div>
                                            }
                                            return null
                                        })}
                                    </div>
                                )}
                            </div>
                        )}
                    </details>
                </div>
            )}

            {/* Rollout Legend (right) */}
            {validRollouts.length > 0 && (
                <GraphizLegend
                    validRollouts={validRollouts}
                    rolloutColorMap={rolloutColorMap}
                    rolloutColors={rolloutColors}
                    hoveredRollout={hoveredRollout}
                    onRolloutHover={setHoveredRollout}
                    onRolloutClick={setPanelRollout}
                    enabledPropertyCheckers={enabledPropertyCheckers}
                    getRolloutPropertyValue={getRolloutPropertyValue}
                />
            )}

            {/* Rollout Details Panel (draggable and resizable) */}
            {panelRollout && (
                <div
                    ref={rolloutPanelContainerRef}
                    style={{
                        position: 'absolute',
                        left: panelPosition.x,
                        top: panelPosition.y,
                        width: 320,
                        height: panelHeight,
                        zIndex: 1000,
                        userSelect: 'none',
                    }}
                >
                    <GraphizRolloutPanel
                        panelRollout={panelRollout}
                        onClose={() => setPanelRollout(null)}
                        rolloutPanelRef={rolloutPanelRef}
                        data={data}
                        getRolloutPathWithTexts={getRolloutPathWithTexts}
                        highlightClusterId={highlightClusterId}
                        onClusterHover={setHighlightClusterId}
                        onRowHover={(rid, idx, cid) => setPanelHover(idx >= 0 ? { rolloutId: rid, index: idx } : null)}
                        showAnswerLabel={showAnswerLabel}
                        showQuestionLabel={showQuestionLabel}
                        skipQuestionRestatements={skipQuestionRestatements}
                        collapseAnswerCycles={collapseAnswerCycles}
                        collapseAllCyclesExceptQuestion={collapseAllCyclesExceptQuestion}
                        isDragging={isDragging}
                        onResizeMouseDown={handleResizeMouseDown}
                        onDragMouseDown={handleMouseDown}
                        getNodeLabel={(cid) => {
                            // Lookup label directly from data.nodes
                            const entry = (data.nodes || []).find((n: any) => {
                                if (n && n.cluster_id) {
                                    return n.cluster_id === cid
                                } else if (n && typeof n === 'object') {
                                    const k = Object.keys(n)[0]
                                    return k === cid
                                }
                                return false
                            }) as any
                            if (!entry) return undefined
                            if (entry.cluster_id) {
                                return entry.label
                            } else {
                                const k = Object.keys(entry)[0]
                                return entry[k]?.label
                            }
                        }}
                        hoveredIndex={panelHover ? panelHover.index : null}
                    />
                </div>
            )}

            {/* Strict cluster selection panel */}
            {strictClusterSelection && showStrictPanel && clusterSelectedSeq && clusterSelectedSeq.length > 0 && (
                <div
                    style={{
                        position: 'absolute',
                        left: strictPanelPosition.x,
                        top: strictPanelPosition.y,
                        width: 340,
                        maxHeight: '60%',
                        overflowY: 'auto',
                        zIndex: 1100,
                        userSelect: 'none'
                    }}
                    onMouseDown={handleStrictMouseDown}
                >
                    <div className="rolloutPanel" style={{ height: '100%' }}>
                        <div className="panelHeader" style={{ cursor: isStrictDragging ? 'grabbing' : 'grab' }}>
                            <div className="panelTitle">Selected clusters (strict)</div>
                            <button onClick={() => setShowStrictPanel(false)} className="closeButton">×</button>
                        </div>
                        <div>
                            {clusterSelectedSeq.map((cid, idx) => {
                                const nodeEntry: any = (data.nodes || []).find((n: any) => {
                                    if (n && n.cluster_id) return n.cluster_id === cid
                                    const k = Object.keys(n || {})[0]
                                    return k === cid
                                })
                                let rep: string = ''
                                if (nodeEntry) {
                                    if (nodeEntry.cluster_id) {
                                        rep = nodeEntry.representative_sentence || ''
                                    } else {
                                        const k = Object.keys(nodeEntry)[0]
                                        rep = (nodeEntry[k]?.representative_sentence) || ''
                                    }
                                }
                                return (
                                    <div key={`${cid}-${idx}`} className="rolloutRow">
                                        <div className="clusterCircle">{(() => {
                                            if (!nodeEntry) return ''
                                            if (nodeEntry.cluster_id) return nodeEntry.freq
                                            const k = Object.keys(nodeEntry)[0]
                                            return nodeEntry[k]?.freq
                                        })()}</div>
                                        <div className="clusterInfo">
                                            <div className="clusterId">Cluster {cid}</div>
                                            <div className="clusterSentence">{rep}</div>
                                        </div>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                </div>
            )}

            <svg ref={svgRef} style={{ width: '100%', height: '100%' }} />

            {/* Node Details Modal */}
            {selectedNode && (
                <div className="nodeModal">
                    <div className="modalHeader">
                        <h3 className="modalTitle">
                            Cluster {selectedNode.cluster_id}
                        </h3>
                        <button
                            onClick={() => setSelectedNode(null)}
                            className="modalCloseButton"
                        >
                            ×
                        </button>
                    </div>

                    <div className="modalContent">
                        <div className="modalSize">
                            Size: {selectedNode.freq} sentences
                        </div>
                        <div className="modalRepresentative">
                            Representative:
                        </div>
                        <p className="modalText">
                            {selectedNode.representative_sentence}
                        </p>
                    </div>

                    <div className="modalSentencesTitle">
                        <strong>
                            Sentences ({selectedNode.sentences?.length || 0} total):
                        </strong>
                    </div>

                    <div className="sentencesContainer">
                        {(selectedNode.sentences || []).map((sentence, index) => (
                            <div key={index} className="sentenceItem">
                                <div className="sentenceCount">
                                    Count: {sentence.count}
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

'use client'

import { RefObject, useEffect, useMemo, useState } from 'react'
import './graph.css'

interface GraphizRolloutPanelProps {
    panelRollout: string | null
    onClose: () => void
    rolloutPanelRef: RefObject<HTMLDivElement>
    getRolloutPathWithTexts: (rolloutId: string) => { seq: string[]; texts: (string | undefined)[] }
    data: any
    highlightClusterId: string | null
    onClusterHover: (clusterId: string | null) => void
    onRowHover?: (rolloutId: string, index: number, clusterId: string | null) => void
    hoveredIndex?: number | null
    showAnswerLabel?: boolean
    showQuestionLabel?: boolean
    getNodeLabel?: (cid: string) => string | undefined
    skipQuestionRestatements?: boolean
    collapseAnswerCycles?: boolean
    collapseAllCyclesExceptQuestion?: boolean
    isDragging?: boolean
    onResizeMouseDown?: (event: React.MouseEvent) => void
    onDragMouseDown?: (event: React.MouseEvent) => void
}

export default function GraphizRolloutPanel({
    panelRollout,
    onClose,
    rolloutPanelRef,
    getRolloutPathWithTexts,
    data,
    highlightClusterId,
    onClusterHover,
    onRowHover,
    hoveredIndex = null,
    showAnswerLabel = false,
    showQuestionLabel = false,
    getNodeLabel,
    skipQuestionRestatements = false,
    collapseAnswerCycles = false,
    collapseAllCyclesExceptQuestion = false,
    isDragging = false,
    onResizeMouseDown,
    onDragMouseDown
}: GraphizRolloutPanelProps) {
    if (!panelRollout) return null

    const promptIndex: string | undefined = (data && (data as any).prompt_index) ? (data as any).prompt_index : undefined
    const modelName: string | undefined = (data && (data as any).models && (data as any).models.length > 0) ? (data as any).models[0] : ((data && (data as any).model) ? (data as any).model : undefined)

    const [rolloutJson, setRolloutJson] = useState<any | null>(null)
    const [loadingRollout, setLoadingRollout] = useState<boolean>(false)

    useEffect(() => {
        if (!panelRollout || !promptIndex || !modelName) {
            setRolloutJson(null)
            return
        }
        setLoadingRollout(true)
        fetch(`/api/rollout/${encodeURIComponent(promptIndex)}/${encodeURIComponent(modelName)}/${encodeURIComponent(panelRollout)}`)
            .then(r => r.ok ? r.json() : null)
            .then(j => setRolloutJson(j))
            .catch(() => setRolloutJson(null))
            .finally(() => setLoadingRollout(false))
    }, [panelRollout, promptIndex, modelName])

    const path = getRolloutPathWithTexts(panelRollout)
    let seq = path.seq
    let stepTexts = path.texts
    if (skipQuestionRestatements) {
        const isQuestion = (cid: string) => {
            const lbl = getNodeLabel ? getNodeLabel(cid) : undefined
            return lbl === 'question'
        }
        const bridged: string[] = []
        for (let i = 0; i < seq.length; i++) {
            const cur = seq[i]
            if (!isQuestion(cur)) {
                bridged.push(cur)
                continue
            }
            let j = i
            while (j + 1 < seq.length && isQuestion(seq[j + 1])) j++
            const prev = bridged.length > 0 ? bridged[bridged.length - 1] : null
            const next = (j + 1 < seq.length) ? seq[j + 1] : null
            if (prev && next && prev !== next) {
                if (bridged.length === 0 || bridged[bridged.length - 1] !== prev) bridged.push(prev)
                if (bridged[bridged.length - 1] !== next) bridged.push(next)
            }
            i = j
        }
        if (bridged.length > 0) seq = bridged
    }

    if (collapseAnswerCycles || collapseAllCyclesExceptQuestion) {
        const isAnswer = (cid: string) => {
            const lbl = getNodeLabel ? getNodeLabel(cid) : undefined
            return lbl === 'answer' || (typeof lbl === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(lbl))
        }
        const isEligible = (cid: string) => {
            const lbl = getNodeLabel ? getNodeLabel(cid) : undefined
            if (collapseAllCyclesExceptQuestion) return lbl !== 'question'
            return isAnswer(cid)
        }
        const out: string[] = []
        let i = 0
        while (i < seq.length) {
            const cur = seq[i]
            out.push(cur)
            if (!isEligible(cur)) { i++; continue }
            let lastSame = -1
            for (let j = seq.length - 1; j > i; j--) {
                if (seq[j] === cur) { lastSame = j; break }
            }
            if (lastSame > i) {
                let firstOther = -1
                for (let k = i + 1; k < lastSame; k++) {
                    const c = seq[k]
                    const lbl = getNodeLabel ? getNodeLabel(c) : undefined
                    const eligible = collapseAllCyclesExceptQuestion ? (lbl !== 'question') : isAnswer(c)
                    if (eligible && c !== cur) { firstOther = k; break }
                }
                if (firstOther !== -1) {
                    for (let t = i + 1; t <= firstOther; t++) out.push(seq[t])
                }
                const after = (lastSame + 1 < seq.length) ? seq[lastSame + 1] : null
                if (after && /^response-/.test(after)) {
                    if (out[out.length - 1] !== after) out.push(after)
                    i = lastSame + 2
                } else {
                    i = lastSame + 1
                }
                continue
            }
            i++
        }
        const dedup: string[] = []
        out.forEach(id => { if (dedup.length === 0 || dedup[dedup.length - 1] !== id) dedup.push(id) })
        seq = dedup
    }

    // Build helper for finding exact chunk text per cluster for this rollout
    const chunkList: string[] = useMemo(() => {
        if (!rolloutJson) return []
        if (Array.isArray(rolloutJson.chunked_cot_content)) return rolloutJson.chunked_cot_content as string[]
        // Fallback: split CoT content into sentences if chunks not present
        const content: string = (rolloutJson.cot_content || '').toString()
        if (!content) return []
        // naive split by newline as content often includes line breaks per chunk
        return content.split(/\n+/).map((s: string) => s.trim()).filter(Boolean)
    }, [rolloutJson])

    const responseText: string | undefined = rolloutJson ? (rolloutJson.processed_response_content || rolloutJson.response_content || rolloutJson.answer || undefined) : undefined

    // Compute repeated clusters and assign stable, distinct colors per cluster within this rollout
    const counts = new Map<string, number>()
    seq.forEach(cid => counts.set(cid, (counts.get(cid) || 0) + 1))
    const repeatedIds = new Set<string>(Array.from(counts.entries()).filter(([_, c]) => c && c > 1).map(([cid]) => cid))
    const repeatedList = Array.from(repeatedIds)
    const hash = (s: string) => {
        let h = 0
        for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
        return h >>> 0
    }
    const hueOffset = panelRollout ? (hash(panelRollout) % 360) : 0
    const colorByClusterId = new Map<string, string>()
    if (repeatedList.length > 0) {
        const step = 360 / repeatedList.length
        repeatedList.sort() // stable order
        repeatedList.forEach((cid, idx) => {
            const hue = Math.round((hueOffset + idx * step) % 360)
            colorByClusterId.set(cid, `hsl(${hue}, 70%, 85%)`)
        })
    }

    return (
        <div
            className={`rolloutPanel ${isDragging ? 'dragging' : ''}`}
            ref={rolloutPanelRef}
            style={{ height: '100%', overflowY: 'auto' }}
        >
            <div
                className="panelHeader"
                style={{
                    cursor: isDragging ? 'grabbing' : 'grab',
                    opacity: isDragging ? 0.8 : 1,
                    transition: isDragging ? 'none' : 'opacity 0.2s ease'
                }}
                onMouseDown={onDragMouseDown}
            >
                <div className="panelTitle">Response {panelRollout}</div>
                <button onClick={onClose} className="closeButton">×</button>
            </div>
            <div>
                {seq.map((cid, idx) => {
                    const node = data.nodes.find((n: any) => {
                        // Handle both formats: new format has cluster keys, old format has direct properties
                        if (n.cluster_id) {
                            // Old format: direct properties
                            return n.cluster_id === cid
                        } else {
                            // New format: objects with cluster keys
                            const clusterKey = Object.keys(n)[0]
                            return clusterKey === cid
                        }
                    })
                    if (!node) return null
                    const isHovered = hoveredIndex !== null && hoveredIndex !== undefined && hoveredIndex === idx
                    const isHighlighted = isHovered ? true : (highlightClusterId === cid)
                    const label = getNodeLabel ? getNodeLabel(cid) : undefined
                    const labelBg = (function () {
                        const isNumericAnswer = typeof label === 'string' && /^[-+]?\d+(?:\.\d+)?$/.test(label)
                        if ((label === 'answer' || isNumericAnswer) && showAnswerLabel) return '#ede9fe' // light purple
                        if (label === 'question' && showQuestionLabel) return '#fff6bf' // light gold
                        return undefined
                    })()
                    // Use precomputed step text aligned with filtered path
                    const displayText = (stepTexts && stepTexts[idx]) ? String(stepTexts[idx]) : ''
                    return (
                        <div
                            key={`${cid}-${idx}`}
                            id={`rollout-row-${cid}-${idx}`}
                            className={`rolloutRow ${isHighlighted ? 'rolloutRowHighlighted' : ''}`}
                            style={{ backgroundColor: isHighlighted ? undefined : (labelBg || (repeatedIds.has(cid) ? colorByClusterId.get(cid) : undefined)) }}
                            onMouseEnter={() => { onClusterHover(cid); onRowHover && panelRollout && onRowHover(panelRollout, idx, cid) }}
                            onMouseLeave={() => { onClusterHover(null); onRowHover && panelRollout && onRowHover(panelRollout, -1, null) }}
                        >
                            <div className="clusterCircle">
                                {(() => {
                                    // Handle both formats: new format has cluster keys, old format has direct properties
                                    if (node.cluster_id) {
                                        // Old format: direct properties
                                        return node.freq
                                    } else {
                                        // New format: objects with cluster keys
                                        const clusterKey = Object.keys(node)[0]
                                        return node[clusterKey].freq
                                    }
                                })()}
                            </div>
                            <div className="clusterInfo">
                                <div className="clusterId">Cluster {cid}</div>
                                <div className="clusterSentence">{displayText}</div>
                            </div>
                        </div>
                    )
                })}
                {seq.length === 0 && (
                    <div className="noClustersMessage">No clusters at current size</div>
                )}
            </div>
            {/* Resize handle */}
            {onResizeMouseDown && (
                <div
                    className="resizeHandle"
                    onMouseDown={onResizeMouseDown}
                />
            )}
        </div>
    )
}

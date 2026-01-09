'use client'

import { useState, useEffect } from 'react'
import { FlowchartData } from '@/types/flowchart'
import GraphizVisualization from '@/components/GraphizVisualization'
import AlgorithmStructure from '@/components/AlgorithmStructure'
import ClusterAlgorithmStructure from '@/components/ClusterAlgorithmStructure'
import { getEnvironmentInfo } from '@/utils/flowchartConfig'

interface FileOption {
    label: string
    value: string
    isFolder?: boolean
    children?: Array<{ label: string, value: string }>
}

export default function Home() {
    const [files, setFiles] = useState<FileOption[]>([])
    const [selectedFile, setSelectedFile] = useState<string>('')
    const [data, setData] = useState<FlowchartData | null>(null)
    const [loading, setLoading] = useState(false)
    const [rolloutInput, setRolloutInput] = useState('')
    const [clusterSelectInput, setClusterSelectInput] = useState('')
    const [strictClusterSelection, setStrictClusterSelection] = useState(false)
    const [clusterSelectedSeq, setClusterSelectedSeq] = useState<string[]>([])
    const [selectedRollouts, setSelectedRollouts] = useState<string[]>([])
    const [activeTab, setActiveTab] = useState<'graphviz' | 'algorithm' | 'cluster-algorithm'>('graphviz')
    const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
    const [dropdownOpen, setDropdownOpen] = useState(false)
    const [propertyCheckers, setPropertyCheckers] = useState<string[]>([])
    const [promptText, setPromptText] = useState<string>('')
    const [envInfo, setEnvInfo] = useState<any>(null)
    const [clusterSearchTerm, setClusterSearchTerm] = useState<string>('')

    // Load available files and environment info
    useEffect(() => {
        fetch('/api/files')
            .then(res => res.json())
            .then(data => {
                // Handle both old format (array) and new format (object with files and environment)
                if (Array.isArray(data)) {
                    setFiles(data)
                } else {
                    setFiles(data.files || [])
                    setEnvInfo(data.environment)
                }
            })
            .catch(console.error)
    }, [])

    // Load flowchart data when file is selected
    useEffect(() => {
        if (selectedFile) {
            setLoading(true)
            const apiUrl = `/api/flowchart/${selectedFile}`
            console.log('Frontend - Requesting URL:', apiUrl)
            console.log('Frontend - Selected file:', selectedFile)
            fetch(apiUrl)
                .then(res => res.json())
                .then((data: FlowchartData) => {
                    console.log('Frontend - Received data with', data.nodes?.length, 'nodes')
                    setData(data)

                    // Extract property checkers from flowchart data
                    const checkers = (data as any).property_checkers || []
                    console.log('Frontend - Extracted property checkers:', checkers)
                    setPropertyCheckers(checkers)

                    // Check if prompt exists directly in the JSON data
                    const directPrompt = (data as any).prompt
                    if (directPrompt && typeof directPrompt === 'string') {
                        setPromptText(directPrompt)
                        setLoading(false)
                    } else {
                        // Fetch prompt text using prompt_index in flowchart data
                        const pidx = (data as any).prompt_index
                        if (pidx) {
                            fetch(`/api/prompts/${encodeURIComponent(pidx)}`)
                                .then(r => r.json())
                                .then(obj => {
                                    if (obj && typeof obj.prompt === 'string') {
                                        setPromptText(obj.prompt)
                                    } else {
                                        setPromptText('')
                                    }
                                })
                                .catch(() => setPromptText(''))
                                .finally(() => setLoading(false))
                        } else {
                            setPromptText('')
                            setLoading(false)
                        }
                    }
                })
                .catch(err => {
                    console.error(err)
                    setLoading(false)
                })
        } else {
            setData(null)
            setPropertyCheckers([])
            setPromptText('')
        }
    }, [selectedFile])

    // Parse rollout selection
    const parseRolloutSelection = (input: string) => {
        if (!input.trim()) {
            setSelectedRollouts([])
            return
        }

        const ids = new Set<string>()
        const parts = input.split(',')

        parts.forEach(part => {
            part = part.trim()
            if (part.includes('-')) {
                // Handle ranges like "3-10"
                const [start, end] = part.split('-').map(n => n.trim())
                const startNum = parseInt(start)
                const endNum = parseInt(end)
                if (!isNaN(startNum) && !isNaN(endNum)) {
                    for (let i = Math.min(startNum, endNum); i <= Math.max(startNum, endNum); i++) {
                        ids.add(i.toString())
                    }
                }
            } else {
                // Handle single numbers
                const num = parseInt(part)
                if (!isNaN(num)) {
                    ids.add(num.toString())
                }
            }
        })

        setSelectedRollouts(Array.from(ids).sort((a, b) => parseInt(a) - parseInt(b)))
    }

    const handleRolloutInputChange = (value: string) => {
        setRolloutInput(value)
        parseRolloutSelection(value)
    }

    // Parse cluster shorthand like "3,5-8,12" into ordered cluster IDs ["cluster-3", ...]
    const parseClusterShorthand = (input: string): string[] => {
        const out: string[] = []
        const parts = input.split(',').map(p => p.trim()).filter(p => p.length > 0)
        parts.forEach(part => {
            if (part.includes('-')) {
                const [a, b] = part.split('-').map(s => s.trim())
                const sa = parseInt(a)
                const sb = parseInt(b)
                if (!isNaN(sa) && !isNaN(sb)) {
                    const lo = Math.min(sa, sb)
                    const hi = Math.max(sa, sb)
                    for (let i = lo; i <= hi; i++) out.push(`cluster-${i}`)
                }
            } else {
                const n = parseInt(part)
                if (!isNaN(n)) out.push(`cluster-${n}`)
            }
        })
        return out
    }

    // Build raw cluster sequence for a rollout (no filtering, no dedupe, excludes START/response nodes)
    const getRawClusterSequence = (rid: string): string[] => {
        if (!data) return []
        const responses: any = (data as any).responses || (data as any).rollouts
        const rdata: any = Array.isArray(responses)
            ? (responses.find((r: any) => (r.index && r.index.toString() === rid)) || (responses.find((x: any) => x[rid]) ? responses.find((x: any) => x[rid])[rid] : null))
            : responses[rid]
        const edges: Array<{ node_a: string; node_b: string }> = (rdata && Array.isArray(rdata.edges)) ? rdata.edges : (Array.isArray(rdata) ? rdata : [])
        const seq: string[] = []
        const norm = (id: string) => (/^\d+$/.test(id) ? `cluster-${id}` : id)
        const isResp = (id: string) => id.startsWith('response-')
        edges.forEach((e, i) => {
            const a = norm(e.node_a)
            const b = norm(e.node_b)
            if (i === 0 && a !== 'START' && !isResp(a)) seq.push(a)
            if (b !== 'START' && !isResp(b)) seq.push(b)
        })
        return seq
    }

    // Apply cluster-based selection (AND semantics; strict = contiguous subsequence match)
    const applyClusterSelection = (input: string, strict: boolean) => {
        const req = parseClusterShorthand(input)
        if (req.length === 0 || !data) {
            setSelectedRollouts([])
            return
        }
        const responses: any = (data as any).responses || (data as any).rollouts || {}
        const rolloutIds: string[] = Array.isArray(responses)
            ? responses.map((r: any) => (r.index ? String(r.index) : Object.keys(r)[0])).filter(Boolean)
            : Object.keys(responses)

        const reqSet = new Set(req)
        const matches: string[] = []
        rolloutIds.forEach(rid => {
            const seq = getRawClusterSequence(rid)
            if (strict) {
                if (seq.length < req.length) return
                let ok = false
                const m = req.length
                for (let i = 0; i + m <= seq.length; i++) {
                    let all = true
                    for (let j = 0; j < m; j++) {
                        if (seq[i + j] !== req[j]) { all = false; break }
                    }
                    if (all) { ok = true; break }
                }
                if (ok) matches.push(rid)
            } else {
                // AND: every unique required cluster must appear somewhere in seq
                const seen = new Set(seq)
                let all = true
                reqSet.forEach(cid => { if (!seen.has(cid)) all = false })
                if (all) matches.push(rid)
            }
        })
        matches.sort((a, b) => parseInt(a) - parseInt(b))
        setSelectedRollouts(matches)
    }

    const toggleFolder = (folderValue: string) => {
        const newExpanded = new Set(expandedFolders)
        if (newExpanded.has(folderValue)) {
            newExpanded.delete(folderValue)
        } else {
            newExpanded.add(folderValue)
        }
        setExpandedFolders(newExpanded)
    }

    const toggleDropdown = () => {
        setDropdownOpen(!dropdownOpen)
    }

    const handleFileSelect = (fileValue: string) => {
        setSelectedFile(fileValue)
        setDropdownOpen(false)
    }

    const renderFileOption = (file: FileOption, level: number = 0) => {
        const isExpanded = expandedFolders.has(file.value)

        if (file.isFolder) {
            return (
                <div key={file.value} className="file-option">
                    <div
                        className="folder-option"
                        style={{ paddingLeft: `${level * 16}px`, cursor: 'pointer' }}
                        onClick={() => toggleFolder(file.value)}
                    >
                        <span className="folder-icon">{isExpanded ? '▼' : '▶'}</span> {file.label}
                    </div>
                    {isExpanded && file.children && (
                        <div className="folder-children">
                            {file.children.map(child => renderFileOption(child, level + 1))}
                        </div>
                    )}
                </div>
            )
        }

        return (
            <div
                key={file.value}
                className={`file-option ${selectedFile === file.value ? 'selected' : ''}`}
                style={{ paddingLeft: `${level * 16}px`, cursor: 'pointer' }}
                onClick={() => handleFileSelect(file.value)}
            >
                {file.label}
            </div>
        )
    }

    return (
        <div className="container">
            <div className="header">
                <div>
                    <h1>Flowchart Visualizer</h1>
                </div>
                <div className="header-controls">
                    <div className="file-selector">
                        <label>Select Flowchart:</label>
                        <div className={`dropdown-container ${dropdownOpen ? 'open' : ''}`}>
                            <button className="dropdown-trigger" onClick={toggleDropdown}>
                                {selectedFile ? selectedFile.split('/').pop() : 'Choose file...'}
                            </button>
                            <div className="file-browser">
                                {files.length === 0 ? (
                                    <div className="no-files">No flowchart files found</div>
                                ) : (
                                    files.map(file => renderFileOption(file))
                                )}
                            </div>
                        </div>
                    </div>
                    <div className="rollout-selector">
                        <label htmlFor="rolloutInput">Rollouts:</label>
                        <input
                            id="rolloutInput"
                            type="text"
                            className="rollout-input"
                            placeholder="e.g., 5, 7-9, 11, 19, 100-102"
                            value={rolloutInput}
                            onChange={(e) => handleRolloutInputChange(e.target.value)}
                            style={{ width: '120px' }}
                        />
                    </div>
                    <div className="rollout-selector">
                        <label htmlFor="clusterSelectInput" style={{ whiteSpace: 'nowrap' }}>Clusters:</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <input
                                id="clusterSelectInput"
                                type="text"
                                className="rollout-input"
                                placeholder="e.g., 3,5-8,12"
                                value={clusterSelectInput}
                                onChange={(e) => {
                                    const v = e.target.value
                                    setClusterSelectInput(v)
                                    const req = parseClusterShorthand(v)
                                    setClusterSelectedSeq(req)
                                    applyClusterSelection(v, strictClusterSelection)
                                }}
                                style={{ width: '140px' }}
                            />
                            <label className="checkboxLabel" title="Contiguous subsequence match in order">
                                <input
                                    type="checkbox"
                                    checked={strictClusterSelection}
                                    onChange={(e) => {
                                        const next = e.target.checked
                                        setStrictClusterSelection(next)
                                        applyClusterSelection(clusterSelectInput, next)
                                    }}
                                />
                                <span style={{ fontWeight: 600 }}>strict cluster selection</span>
                            </label>
                        </div>
                    </div>
                    <div className="rollout-selector">
                        <label htmlFor="clusterSearchInput" style={{ whiteSpace: 'nowrap' }}>Search clusters:</label>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                            <input
                                id="clusterSearchInput"
                                type="text"
                                className="rollout-input"
                                placeholder="Search text..."
                                value={clusterSearchTerm}
                                onChange={(e) => setClusterSearchTerm(e.target.value)}
                                style={{ width: '140px' }}
                            />
                            {clusterSearchTerm && (
                                <button
                                    onClick={() => setClusterSearchTerm('')}
                                    style={{
                                        padding: '2px 6px',
                                        fontSize: '11px',
                                        border: 'none',
                                        borderRadius: '3px',
                                        backgroundColor: '#ef4444',
                                        color: 'white',
                                        cursor: 'pointer',
                                        height: 'fit-content',
                                        lineHeight: '1.2'
                                    }}
                                >
                                    ✕
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>


            {/* Tab Navigation */}
            {data && (
                <div className="tab-navigation">
                    <button
                        className={`tab-button ${activeTab === 'graphviz' ? 'active' : ''}`}
                        onClick={() => setActiveTab('graphviz')}
                    >
                        Graph Visualization
                    </button>
                    <button
                        className={`tab-button ${activeTab === 'algorithm' ? 'active' : ''}`}
                        onClick={() => setActiveTab('algorithm')}
                    >
                        Algorithm Structure
                    </button>
                    <button
                        className={`tab-button ${activeTab === 'cluster-algorithm' ? 'active' : ''}`}
                        onClick={() => setActiveTab('cluster-algorithm')}
                    >
                        Cluster Algorithm Structure
                    </button>
                </div>
            )}

            <div className="main-content">
                <div className="visualization-area" style={{ height: 'calc(100vh - 140px)', position: 'relative', overflow: 'visible' }}>
                    {loading ? (
                        <div className="loading">
                            <div className="spinner"></div>
                            <span>Loading...</span>
                        </div>
                    ) : !data ? (
                        <div className="empty-state">
                            Select a flowchart file to begin
                        </div>
                    ) : (
                        <>
                            {promptText && (
                                <div style={{ position: 'absolute', top: '12px', left: 0, right: 0, textAlign: 'center', padding: '12px 16px', fontWeight: 600, fontSize: '18px' }}>
                                    {promptText}
                                </div>
                            )}
                            <div style={{ position: 'absolute', top: promptText ? '56px' : 0, left: 0, right: 0, bottom: 0 }}>
                                {activeTab === 'graphviz' ? (
                                    <GraphizVisualization
                                        data={data}
                                        selectedRollouts={selectedRollouts}
                                        datasetId={selectedFile ? selectedFile.split('/').pop() : 'default'}
                                        propertyCheckers={propertyCheckers}
                                        clusterSearchTerm={clusterSearchTerm}
                                        clusterSelectedSeq={clusterSelectedSeq}
                                        strictClusterSelection={strictClusterSelection}
                                    />
                                ) : activeTab === 'algorithm' ? (
                                    <AlgorithmStructure
                                        data={data}
                                        promptIndex={(data as any).prompt_index}
                                    />
                                ) : (
                                    <ClusterAlgorithmStructure
                                        data={data}
                                        promptIndex={(data as any).prompt_index}
                                        selectedRollouts={selectedRollouts}
                                    />
                                )}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}


'use client'

import { useState, useEffect } from 'react'
import { FlowchartData } from '@/types/flowchart'
import GraphizVisualization from '@/components/GraphizReference'

interface FileOption {
    label: string
    value: string
    isFolder?: boolean
    children?: Array<{ label: string, value: string }>
}

export default function ReferencePage() {
    const [files, setFiles] = useState<FileOption[]>([])
    const [selectedFile, setSelectedFile] = useState<string>('')
    const [data, setData] = useState<FlowchartData | null>(null)
    const [loading, setLoading] = useState(false)
    const [rolloutInput, setRolloutInput] = useState('')
    const [selectedRollouts, setSelectedRollouts] = useState<string[]>([])
    const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
    const [dropdownOpen, setDropdownOpen] = useState(false)
    const [promptText, setPromptText] = useState<string>('')

    // Load available files
    useEffect(() => {
        fetch('/api/files')
            .then(res => res.json())
            .then(setFiles)
            .catch(console.error)
    }, [])

    // Load flowchart data when file is selected
    useEffect(() => {
        if (selectedFile) {
            setLoading(true)
            const apiUrl = `/api/flowchart/${selectedFile}`
            console.log('Reference Page - Requesting URL:', apiUrl)
            console.log('Reference Page - Selected file:', selectedFile)
            fetch(apiUrl)
                .then(res => res.json())
                .then((data: FlowchartData) => {
                    console.log('Reference Page - Received data with', data.nodes?.length, 'nodes')
                    setData(data)
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
                })
                .catch(err => {
                    console.error(err)
                    setLoading(false)
                })
        } else {
            setData(null)
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
                <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                    <h1>Reference Visualization</h1>
                    <a 
                        href="/" 
                        style={{ 
                            padding: '8px 16px', 
                            backgroundColor: '#3b82f6', 
                            color: 'white', 
                            textDecoration: 'none', 
                            borderRadius: '6px',
                            fontSize: '14px'
                        }}
                    >
                        ← Back to Main
                    </a>
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
                        />
                    </div>
                </div>
            </div>

            <div className="main-content">
                <div className="visualization-area" style={{ position: 'relative', overflow: 'visible' }}>
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
                                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, textAlign: 'center', padding: '8px 16px', fontWeight: 600 }}>
                                    {promptText}
                                </div>
                            )}
                            <div style={{ position: 'absolute', top: promptText ? 40 : 0, left: 0, right: 0, bottom: 0 }}>
                                <GraphizVisualization
                                    data={data}
                                    selectedRollouts={selectedRollouts}
                                    datasetId={selectedFile ? selectedFile.split('/').pop() : 'default'}
                                />
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}

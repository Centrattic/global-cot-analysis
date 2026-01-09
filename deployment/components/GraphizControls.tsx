'use client'

import { useState } from 'react'
import './graph.css'

interface GraphizControlsProps {
    minClusterSize: number // Minimum cluster size threshold
    maxClusterSize: number // Maximum cluster size in data
    filteredNodesCount: number
    onMinClusterSizeChange: (size: number) => void
    maxEntropy: number
    minEntropy: number
    maxEntropyData: number
    minEntropyData: number
    onMaxEntropyChange: (entropy: number) => void
    directedEdges: boolean
    onDirectedEdgesChange: (directed: boolean) => void
    propertyCheckers: string[]
    enabledPropertyCheckers: Set<string>
    onPropertyCheckerChange: (checker: string, enabled: boolean) => void
    balanceMode?: 'none' | 'equal_count' | 'equal_length'
    onBalanceModeChange?: (mode: 'none' | 'equal_count' | 'equal_length') => void
    hasFocus?: boolean
    onClearFocus?: () => void
    focusedClusterIds?: string[]
    onClearSingleFocus?: (clusterId: string) => void
    propertyFilter?: { propertyName: string | null; filterMode: 'both' | number | 'delta' }
    onPropertyFilterChange?: (filter: { propertyName: string | null; filterMode: 'both' | number | 'delta' }) => void
    getRolloutPropertyValue?: (rid: string, propertyName: string) => any
    validRollouts?: string[]
    focusMode?: boolean
    onFocusModeChange?: (val: boolean) => void
    containerRef?: React.RefObject<HTMLDivElement>
    showAnswerLabel?: boolean
    showQuestionLabel?: boolean
    showSelfCheckingLabel?: boolean
    onShowAnswerLabelChange?: (v: boolean) => void
    onShowQuestionLabelChange?: (v: boolean) => void
    onShowSelfCheckingLabelChange?: (v: boolean) => void
    skipQuestionRestatements?: boolean
    onSkipQuestionRestatementsChange?: (v: boolean) => void
    collapseAnswerCycles?: boolean
    onCollapseAnswerCyclesChange?: (v: boolean) => void
    collapseAllCyclesExceptQuestion?: boolean
    onCollapseAllCyclesExceptQuestionChange?: (v: boolean) => void
    // Rollout length filter
    maxRolloutLength?: number
    selectedMaxRolloutLength?: number
    onSelectedMaxRolloutLengthChange?: (len: number) => void
    rolloutLengthGreaterMode?: boolean
    onRolloutLengthModeChange?: (v: boolean) => void
}

export default function GraphizControls({
    minClusterSize,
    maxClusterSize,
    filteredNodesCount,
    onMinClusterSizeChange,
    maxEntropy,
    minEntropy,
    maxEntropyData,
    minEntropyData,
    onMaxEntropyChange,
    directedEdges,
    onDirectedEdgesChange,
    propertyCheckers,
    enabledPropertyCheckers,
    onPropertyCheckerChange,
    balanceMode = 'none',
    onBalanceModeChange,
    hasFocus = false,
    onClearFocus,
    focusedClusterIds = [],
    onClearSingleFocus,
    propertyFilter = { propertyName: null, filterMode: 'both' },
    onPropertyFilterChange,
    getRolloutPropertyValue,
    validRollouts = [],
    focusMode = true,
    onFocusModeChange,
    containerRef,
    showAnswerLabel = false,
    showQuestionLabel = false,
    showSelfCheckingLabel = false,
    onShowAnswerLabelChange,
    onShowQuestionLabelChange,
    onShowSelfCheckingLabelChange
    , skipQuestionRestatements = false
    , onSkipQuestionRestatementsChange
    , collapseAnswerCycles = false
    , onCollapseAnswerCyclesChange
    , collapseAllCyclesExceptQuestion = false
    , onCollapseAllCyclesExceptQuestionChange
    , maxRolloutLength = 0
    , selectedMaxRolloutLength = 0
    , onSelectedMaxRolloutLengthChange
    , rolloutLengthGreaterMode = false
    , onRolloutLengthModeChange
}: GraphizControlsProps) {
    const SLIDER_MAX = 100
    const entropyDenom = Math.max(0.001, maxEntropyData - minEntropyData)

    // Size threshold slider mapping (linear: left=0, right=maxClusterSize)
    const fromSize = (size: number) => {
        const clamped = Math.max(0, Math.min(maxClusterSize, size))
        return Math.round((clamped / Math.max(1, maxClusterSize)) * SLIDER_MAX)
    }

    const toSize = (val: number) => {
        const t = Math.max(0, Math.min(1, val / SLIDER_MAX))
        const size = Math.round(t * maxClusterSize)
        return size
    }

    const fromEntropy = (entropy: number) => {
        const norm = Math.max(0, Math.min(1, (entropy - minEntropyData) / entropyDenom))
        return Math.round(norm * SLIDER_MAX)
    }

    const toEntropy = (val: number) => {
        const t = Math.max(0, Math.min(1, val / SLIDER_MAX))
        const mapped = minEntropyData + entropyDenom * t
        return Math.round(mapped * 1000) / 1000 // Round to 3 decimal places
    }

    const sizeSliderValue = fromSize(minClusterSize)
    const entropySliderValue = fromEntropy(maxEntropy)
    const fromLen = (len: number) => {
        if (!maxRolloutLength || maxRolloutLength <= 0) return 0
        const clamped = Math.max(0, Math.min(maxRolloutLength, len))
        return Math.round((clamped / Math.max(1, maxRolloutLength)) * SLIDER_MAX)
    }
    const toLen = (val: number) => {
        if (!maxRolloutLength || maxRolloutLength <= 0) return 0
        const t = Math.max(0, Math.min(1, val / SLIDER_MAX))
        return Math.round(t * maxRolloutLength)
    }
    const lengthSliderValue = fromLen(selectedMaxRolloutLength)

    // Compute property filter UI data if a property checker is enabled
    let propertyFilterData: { activeProperty: string; uniqueValues: any[]; isBoolean: boolean } | null = null
    if (enabledPropertyCheckers.size > 0 && getRolloutPropertyValue && validRollouts.length > 0) {
        const activeProperty = Array.from(enabledPropertyCheckers)[0]
        // Helper to check if a property is a multi-algorithm property
        const isMultiAlgorithmProperty = (propertyName: string): boolean => {
            return propertyName.includes('multi_algorithm')
        }
        // Special-case multi_algorithm: treat classes as unique algorithm ids
        if (isMultiAlgorithmProperty(activeProperty)) {
            const algos = new Set<string>()
            validRollouts.forEach(rid => {
                let v = getRolloutPropertyValue(rid, activeProperty)
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
            const uniqueValues = Array.from(algos).sort()
            const isBoolean = uniqueValues.length === 2
            propertyFilterData = { activeProperty, uniqueValues, isBoolean }
        } else {
            const values = new Set<any>()
            validRollouts.forEach(rid => {
                const value = getRolloutPropertyValue(rid, activeProperty)
                if (value !== undefined && value !== null) {
                    values.add(value)
                }
            })
            const uniqueValues = Array.from(values).sort()
            const isBoolean = uniqueValues.length === 2 &&
                uniqueValues.includes(true) && uniqueValues.includes(false)
            propertyFilterData = { activeProperty, uniqueValues, isBoolean }
        }
    }

    // Render property filter UI
    const renderPropertyFilter = () => {
        if (!propertyFilterData) return null
        const { activeProperty, uniqueValues, isBoolean } = propertyFilterData
        const isActive = propertyFilter.propertyName === activeProperty
        const updateFilter = (filterMode: 'both' | number | 'delta') => {
            if (onPropertyFilterChange) {
                onPropertyFilterChange({ propertyName: activeProperty, filterMode })
            }
        }
        return (
            <div className="checkboxContainer" style={{ marginTop: 6, marginLeft: 16 }}>
                <div className="checkboxLabel">
                    View
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
                    <label className="checkboxLabel">
                        <input
                            type="radio"
                            name={`propertyFilter-${activeProperty}`}
                            checked={isActive && propertyFilter.filterMode === 'both'}
                            onChange={() => updateFilter('both')}
                        />
                        Both
                    </label>
                    {uniqueValues.map((value, idx) => (
                        <label key={idx} className="checkboxLabel">
                            <input
                                type="radio"
                                name={`propertyFilter-${activeProperty}`}
                                checked={isActive && propertyFilter.filterMode === idx}
                                onChange={() => updateFilter(idx)}
                            />
                            {isBoolean
                                ? (idx === 0 ? 'Class 0 Only' : 'Class 1 Only')
                                : `Class ${idx} Only`
                            }
                        </label>
                    ))}
                </div>
                <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                    <label className="checkboxLabel">
                        <input
                            type="radio"
                            name={`propertyFilter-${activeProperty}`}
                            checked={isActive && propertyFilter.filterMode === 'delta'}
                            onChange={() => updateFilter('delta')}
                        />
                        Symmetric diff
                    </label>
                </div>
            </div>
        )
    }

    return (
        <div className="controls" ref={containerRef}>
            <div className="controlsTitle">
                Size Threshold Filter
            </div>

            {/* Clear Focus will be shown below property checkers; keep top area minimal */}
            <div className="sliderContainer">
                <input
                    type="range"
                    min={0}
                    max={SLIDER_MAX}
                    value={sizeSliderValue}
                    onChange={(e) => onMinClusterSizeChange(toSize(parseInt(e.target.value)))}
                    className="slider"
                />
            </div>
            <div className="sliderInfo">
                Min size: {minClusterSize} | Range: 0 - {maxClusterSize} | {filteredNodesCount} clusters shown
            </div>

            <div className="controlsTitle" style={{ marginTop: 16 }}>
                Entropy Filter
            </div>

            <div className="sliderContainer">
                <input
                    type="range"
                    min={0}
                    max={SLIDER_MAX}
                    value={entropySliderValue}
                    onChange={(e) => onMaxEntropyChange(toEntropy(parseInt(e.target.value)))}
                    className="slider"
                />
            </div>
            <div className="sliderInfo">
                Max entropy: {maxEntropy.toFixed(3)} | Range: {minEntropyData.toFixed(3)} - {maxEntropyData.toFixed(3)}
            </div>

            {/* Rollout length filter */}
            {maxRolloutLength > 0 && (
                <>
                    <div className="controlsTitle" style={{ marginTop: 16, display: 'flex', alignItems: 'center', justifyContent: 'flex-start', gap: 8 }}>
                        <span>Rollout Length Filter</span>
                        <button
                            type="button"
                            aria-label="Toggle rollout length filter direction"
                            onClick={() => {
                                const newMode = !rolloutLengthGreaterMode
                                onRolloutLengthModeChange && onRolloutLengthModeChange(newMode)
                                // Reset slider to a value that shows all rollouts immediately
                                if (onSelectedMaxRolloutLengthChange) {
                                    onSelectedMaxRolloutLengthChange(newMode ? 0 : maxRolloutLength)
                                }
                            }}
                            style={{
                                fontSize: 12,
                                lineHeight: '16px',
                                padding: '2px 6px',
                                borderRadius: 4,
                                border: '1px solid #374151',
                                background: '#111827',
                                color: '#e5e7eb',
                                cursor: 'pointer'
                            }}
                            title={rolloutLengthGreaterMode ? 'Currently > N. Click to switch to ≤ N' : 'Currently ≤ N. Click to switch to > N'}
                        >
                            {rolloutLengthGreaterMode ? '>' : '≤'}
                        </button>
                    </div>
                    <div className="sliderContainer">
                        <input
                            type="range"
                            min={0}
                            max={SLIDER_MAX}
                            value={lengthSliderValue}
                            onChange={(e) => onSelectedMaxRolloutLengthChange && onSelectedMaxRolloutLengthChange(toLen(parseInt(e.target.value)))}
                            className="slider"
                        />
                    </div>
                    <div className="sliderInfo">
                        {rolloutLengthGreaterMode ? 'Show >' : 'Show ≤'} {selectedMaxRolloutLength} | Range: 0 - {maxRolloutLength}
                    </div>
                </>
            )}
            <div className="checkboxContainer">
                <label className="checkboxLabel">
                    <input
                        type="checkbox"
                        checked={directedEdges}
                        onChange={(e) => onDirectedEdgesChange(e.target.checked)}
                    />
                    Directed edges
                </label>
            </div>

            {/* Focus mode toggle */}
            <div className="checkboxContainer" style={{ marginTop: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <label className="checkboxLabel">
                        <input
                            type="checkbox"
                            checked={focusMode}
                            onChange={(e) => onFocusModeChange && onFocusModeChange(e.target.checked)}
                        />
                        Focus mode
                    </label>
                    <button
                        className="clearFocusButton"
                        onClick={() => onClearFocus && onClearFocus()}
                        disabled={!hasFocus}
                    >
                        Clear focus
                    </button>
                </div>
                {hasFocus && focusedClusterIds.length > 0 && (
                    <div className="focusedChipsContainer" style={{ marginTop: 8 }}>
                        {focusedClusterIds.map((cid) => {
                            const label = cid.startsWith('cluster-')
                                ? cid.replace('cluster-', '')
                                : cid.startsWith('response-')
                                    ? cid.replace('response-', '')
                                    : cid
                            return (
                                <button
                                    key={cid}
                                    className="clearFocusChip"
                                    onClick={() => onClearSingleFocus && onClearSingleFocus(cid)}
                                    title={`Clear focus on ${cid}`}
                                >
                                    {label}
                                </button>
                            )
                        })}
                    </div>
                )}
            </div>

            {/* Property Checker Controls (hide 'resampled') */}
            {propertyCheckers.filter(c => c !== 'resampled').length > 0 && (
                <div className="propertyCheckersSection">
                    {propertyCheckers.filter(c => c !== 'resampled').map(checker => (
                        <div key={checker} className="propertyCheckerItem">
                            <label className="checkboxLabel">
                                <input
                                    type="checkbox"
                                    checked={enabledPropertyCheckers.has(checker)}
                                    onChange={(e) => onPropertyCheckerChange(checker, e.target.checked)}
                                />
                                {checker}
                            </label>
                        </div>
                    ))}
                    {enabledPropertyCheckers.size > 0 && (
                        <div className="checkboxContainer" style={{ marginTop: 12, marginLeft: 16 }}>
                            <div className="checkboxLabel">Balancing</div>
                            <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
                                <label className="checkboxLabel">
                                    <input
                                        type="radio"
                                        name="balanceMode"
                                        checked={balanceMode === 'none'}
                                        onChange={() => onBalanceModeChange && onBalanceModeChange('none')}
                                    />
                                    None
                                </label>
                                <label className="checkboxLabel">
                                    <input
                                        type="radio"
                                        name="balanceMode"
                                        checked={balanceMode === 'equal_count'}
                                        onChange={() => onBalanceModeChange && onBalanceModeChange('equal_count')}
                                    />
                                    Same number per class
                                </label>
                                <label className="checkboxLabel">
                                    <input
                                        type="radio"
                                        name="balanceMode"
                                        checked={balanceMode === 'equal_length'}
                                        onChange={() => onBalanceModeChange && onBalanceModeChange('equal_length')}
                                    />
                                    Similar total length per class
                                </label>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Property filter - only show when a property checker is enabled */}
            {renderPropertyFilter()}

            {/* Details toggle for balancing info placeholder; content provided by parent */}
            {/* The actual values (L_C, L_I, final totals) will be rendered by the parent alongside controls */}

            {/* Labels section */}
            <div className="checkboxContainer" style={{ marginTop: 12 }}>
                <div className="checkboxLabel">Show labels:</div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6 }}>
                    <label className="checkboxLabel" title="State answer">
                        <input
                            type="checkbox"
                            checked={showAnswerLabel}
                            onChange={(e) => onShowAnswerLabelChange && onShowAnswerLabelChange(e.target.checked)}
                        />
                        <span style={{ color: '#8b5cf6', fontWeight: 600 }}>state answer</span>
                    </label>
                    <label className="checkboxLabel" title="Restate question">
                        <input
                            type="checkbox"
                            checked={showQuestionLabel}
                            onChange={(e) => onShowQuestionLabelChange && onShowQuestionLabelChange(e.target.checked)}
                        />
                        <span style={{ color: '#FFD700', fontWeight: 600 }}>restate question</span>
                    </label>
                    <label className="checkboxLabel" title="Self-checking">
                        <input
                            type="checkbox"
                            checked={showSelfCheckingLabel}
                            onChange={(e) => onShowSelfCheckingLabelChange && onShowSelfCheckingLabelChange(e.target.checked)}
                        />
                        <span style={{ color: '#f59e0b', fontWeight: 600 }}>self-checking</span>
                    </label>
                </div>
            </div>

            {/* Distill section */}
            <div className="checkboxContainer" style={{ marginTop: 12 }}>
                <div className="checkboxLabel">Distill:</div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6 }}>
                    <label className="checkboxLabel" title="Collapse cycles through answer nodes">
                        <input
                            type="checkbox"
                            checked={collapseAnswerCycles}
                            onChange={(e) => onCollapseAnswerCyclesChange && onCollapseAnswerCyclesChange(e.target.checked)}
                        />
                        <span style={{ color: '#8b5cf6', fontWeight: 600 }}>collapse answer cycles</span>
                    </label>
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6 }}>
                    <label className="checkboxLabel" title="Collapse cycles for all non-question nodes">
                        <input
                            type="checkbox"
                            checked={collapseAllCyclesExceptQuestion}
                            onChange={(e) => onCollapseAllCyclesExceptQuestionChange && onCollapseAllCyclesExceptQuestionChange(e.target.checked)}
                        />
                        <span style={{ color: '#3b82f6', fontWeight: 600 }}>collapse all cycles (except answer restatement nodes)</span>
                    </label>
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6 }}>
                    <label className="checkboxLabel" title="Skip clusters labeled question">
                        <input
                            type="checkbox"
                            checked={skipQuestionRestatements}
                            onChange={(e) => onSkipQuestionRestatementsChange && onSkipQuestionRestatementsChange(e.target.checked)}
                        />
                        <span style={{ color: '#FFD700', fontWeight: 600 }}>skip question restatements</span>
                    </label>
                </div>
            </div>
        </div >
    )
}

'use client'

import './graph.css'

interface GraphizLegendProps {
    validRollouts: string[]
    rolloutColorMap: Map<string, string>
    rolloutColors: string[]
    hoveredRollout: string | null
    onRolloutHover: (rolloutId: string | null) => void
    onRolloutClick: (rolloutId: string) => void
    enabledPropertyCheckers: Set<string>
    getRolloutPropertyValue: (rolloutId: string, propertyName: string) => any
}

export default function GraphizLegend({
    validRollouts,
    rolloutColorMap,
    rolloutColors,
    hoveredRollout,
    onRolloutHover,
    onRolloutClick,
    enabledPropertyCheckers,
    getRolloutPropertyValue
}: GraphizLegendProps) {
    const renderPropertyValues = () => {
        if (enabledPropertyCheckers.size === 0) return null

        const enabledChecker = Array.from(enabledPropertyCheckers)[0]
        // Helper to check if a property is a multi-algorithm property
        const isMultiAlgorithmProperty = (propertyName: string): boolean => {
            return propertyName.includes('multi_algorithm')
        }
        // Special handling for multi_algorithm: collapse to unique algorithm ids (string tokens)
        if (isMultiAlgorithmProperty(enabledChecker)) {
            const algos = new Set<string>()
            validRollouts.forEach(rid => {
                let v = getRolloutPropertyValue(rid, enabledChecker)
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
            const uniqueAlgos = Array.from(algos).sort()
            return (
                <>
                    <div className="propertyValuesSection">
                        Property Values
                    </div>
                    {uniqueAlgos.map((algo, index) => {
                        const color = rolloutColors[index % rolloutColors.length]
                        return (
                            <div key={`ma-${algo}`} className="propertyValueItem">
                                <div
                                    className="propertyValueColor"
                                    style={{ backgroundColor: color }}
                                />
                                <span className="propertyValueLabel">
                                    {enabledChecker}: {JSON.stringify(algo)}
                                </span>
                            </div>
                        )
                    })}
                </>
            )
        }

        const values = new Set<any>()
        validRollouts.forEach(rid => {
            const value = getRolloutPropertyValue(rid, enabledChecker)
            if (value !== undefined && value !== null) {
                values.add(value)
            }
        })
        const uniqueValues = Array.from(values).sort()

        return (
            <>
                <div className="propertyValuesSection">
                    Property Values
                </div>
                {uniqueValues.map((value, index) => {
                    let color = rolloutColors[index % rolloutColors.length]
                    if (enabledChecker === 'correctness') {
                        color = value === true ? '#10b981' : '#ef4444'
                    } else if (enabledChecker === 'resampled') {
                        if (value === false) {
                            color = '#6b7280' // gray for non-resampled
                        } else {
                            color = rolloutColors[index % rolloutColors.length]
                        }
                    }

                    return (
                        <div key={value} className="propertyValueItem">
                            <div
                                className="propertyValueColor"
                                style={{ backgroundColor: color }}
                            />
                            <span className="propertyValueLabel">
                                {enabledChecker}: {String(value)}
                            </span>
                        </div>
                    )
                })}
            </>
        )
    }

    return (
        <div className="legend">
            <div className="legendTitle">
                Response Trajectories
            </div>
            <div className="rolloutTrajectoriesSection">
                {validRollouts.map((rolloutId, index) => (
                    <div
                        key={rolloutId}
                        className="rolloutItem"
                        style={{ opacity: hoveredRollout === rolloutId ? 1 : 0.9 }}
                        onMouseEnter={() => onRolloutHover(rolloutId)}
                        onMouseLeave={() => onRolloutHover(null)}
                        onClick={() => onRolloutClick(rolloutId)}
                    >
                        <div
                            className="rolloutColor"
                            style={{
                                backgroundColor: rolloutColorMap.get(rolloutId) || rolloutColors[index % rolloutColors.length]
                            }}
                        />
                        <span className="rolloutLabel">
                            Response {rolloutId}
                        </span>
                    </div>
                ))}
            </div>

            {renderPropertyValues()}
        </div>
    )
}

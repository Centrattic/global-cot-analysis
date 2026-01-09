import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { getFlowchartsPath } from '../../../utils/flowchartConfig'

export async function GET() {
    try {
        const flowchartsDir = getFlowchartsPath()

        // Debug logging
        console.log('Environment check:', {
            VERCEL: process.env.VERCEL,
            FLOWCHARTS_FOLDER: process.env.FLOWCHARTS_FOLDER,
            NODE_ENV: process.env.NODE_ENV,
            flowchartsDir: flowchartsDir,
            exists: fs.existsSync(flowchartsDir)
        })

        if (!fs.existsSync(flowchartsDir)) {
            console.log('Flowcharts directory not found:', flowchartsDir)
            return NextResponse.json({
                files: [],
                environment: {
                    isVercel: process.env.VERCEL === '1',
                    flowchartFolder: process.env.FLOWCHARTS_FOLDER || (process.env.VERCEL === '1' ? 'flowcharts_public' : 'flowcharts'),
                    nodeEnv: process.env.NODE_ENV,
                    vercelEnv: process.env.VERCEL_ENV
                }
            })
        }

        const result: Array<{ label: string, value: string, isFolder?: boolean, children?: Array<{ label: string, value: string }> }> = []

        // Read all items in the flowcharts directory
        const allItems = fs.readdirSync(flowchartsDir, { withFileTypes: true })

        // Separate files and directories
        const rootFiles = allItems
            .filter(item => item.isFile() && item.name.endsWith('.json'))
            .map(item => item.name)
            .sort()

        const directories = allItems
            .filter(item => item.isDirectory())
            .map(item => item.name)
            .sort()

        // Add root level files
        rootFiles.forEach(file => {
            result.push({
                label: file,
                value: file
            })
        })

        // Add all directories dynamically with nicer display names
        const folderDisplayNames: Record<string, string> = {
            'hypothesis_generation': 'Hypothesis Generation',
            'algorithms': 'Algorithms',
            'intuition_building': 'Intuition Building',
            'predictive_power': 'Predictive Power'
        }

        directories.forEach(dirName => {
            const dirPath = path.join(flowchartsDir, dirName)
            const dirFiles = fs.readdirSync(dirPath)
                .filter(file => file.endsWith('.json'))
                .sort()

            if (dirFiles.length > 0) {
                const displayName = folderDisplayNames[dirName] || dirName
                result.push({
                    label: `${displayName}/`,
                    value: `${dirName}/`,
                    isFolder: true,
                    children: dirFiles.map(file => ({
                        label: file,
                        value: `${dirName}/${file}`
                    }))
                })
            }
        })

        // Add environment info to the response
        const response = {
            files: result,
            environment: {
                isVercel: process.env.VERCEL === '1',
                flowchartFolder: process.env.FLOWCHARTS_FOLDER || (process.env.VERCEL === '1' ? 'flowcharts_public' : 'flowcharts'),
                nodeEnv: process.env.NODE_ENV,
                vercelEnv: process.env.VERCEL_ENV
            }
        }

        return NextResponse.json(response)
    } catch (error) {
        console.error('Error reading files:', error)
        return NextResponse.json([], { status: 500 })
    }
}

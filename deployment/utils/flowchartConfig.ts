/**
 * Utility functions for flowchart configuration
 */

// Determine which flowchart folder to use based on environment
export function getFlowchartsDir(): string {
    const isVercel = process.env.VERCEL === '1'
    const folderName = process.env.FLOWCHARTS_FOLDER || (isVercel ? 'flowcharts_public' : 'flowcharts')
    return folderName
}

// Get the full path to the flowcharts directory
export function getFlowchartsPath(): string {
    const path = require('path')
    const fs = require('fs')
    const isVercel = process.env.VERCEL === '1'
    const folderName = getFlowchartsDir()

    let fullPath: string

    if (isVercel) {
        // On Vercel, try multiple possible locations
        // The flowcharts_public folder should be in the deployment root
        const possiblePaths = [
            path.join(process.cwd(), folderName), // Current working directory
            path.join(process.cwd(), '..', folderName), // One level up
            path.join('/var/task', folderName), // Vercel serverless function default
            path.join('/var/task', '..', folderName), // Alternative Vercel path
        ]

        // Find the first path that exists
        fullPath = path.join(process.cwd(), folderName) // Default fallback
        for (const testPath of possiblePaths) {
            const normalized = path.normalize(testPath)
            try {
                if (fs.existsSync(normalized)) {
                    fullPath = normalized
                    console.log('getFlowchartsPath - Found at:', fullPath)
                    break
                }
            } catch (e) {
                // Continue to next path
            }
        }

        if (!fs.existsSync(fullPath)) {
            console.log('getFlowchartsPath - Using fallback:', fullPath)
        }
    } else {
        // For local development, go up one directory from deployment folder
        fullPath = path.join(process.cwd(), '..', folderName)
    }

    const normalizedPath = path.normalize(fullPath)

    console.log('getFlowchartsPath debug:', {
        isVercel,
        folderName,
        normalizedPath,
        cwd: process.cwd(),
        VERCEL: process.env.VERCEL,
        exists: fs.existsSync(normalizedPath),
        parentExists: fs.existsSync(path.dirname(normalizedPath))
    })

    return normalizedPath
}

// Check if we're running on Vercel
export function isVercelEnvironment(): boolean {
    return process.env.VERCEL === '1'
}

// Get environment info for debugging
export function getEnvironmentInfo() {
    return {
        isVercel: isVercelEnvironment(),
        flowchartFolder: getFlowchartsDir(),
        nodeEnv: process.env.NODE_ENV,
        vercelEnv: process.env.VERCEL_ENV
    }
}

import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import { getFlowchartsPath } from '../../../../utils/flowchartConfig'

export async function GET(
    request: Request,
    { params }: { params: { filename: string[] } }
) {
    // Log that the route handler was called
    console.log('=== FLOWCHART API ROUTE CALLED ===')
    console.log('URL:', request.url)
    console.log('Params:', params)
    
    try {
        if (!params || !params.filename || !Array.isArray(params.filename)) {
            console.error('Invalid params:', params)
            return NextResponse.json({ error: 'Invalid filename parameter' }, { status: 400 })
        }
        
        const filename = params.filename.join('/')
        console.log('Joined filename:', filename)
        
        const flowchartsPath = getFlowchartsPath()
        console.log('Flowcharts base path:', flowchartsPath)
        console.log('Flowcharts path exists:', fs.existsSync(flowchartsPath))
        
        const filePath = path.join(flowchartsPath, filename)
        console.log('Full file path:', filePath)
        console.log('File exists:', fs.existsSync(filePath))

        // List base directory contents
        if (fs.existsSync(flowchartsPath)) {
            try {
                const dirContents = fs.readdirSync(flowchartsPath, { withFileTypes: true })
                console.log('Base directory contents:', dirContents.map(item => ({
                    name: item.name,
                    isDirectory: item.isDirectory(),
                    isFile: item.isFile()
                })))
                
                // If the path has subdirectories, check them too
                const pathParts = filename.split('/')
                if (pathParts.length > 1) {
                    let currentPath = flowchartsPath
                    for (let i = 0; i < pathParts.length - 1; i++) {
                        currentPath = path.join(currentPath, pathParts[i])
                        console.log(`Checking subdirectory ${i}:`, currentPath, 'exists:', fs.existsSync(currentPath))
                        if (fs.existsSync(currentPath)) {
                            const subContents = fs.readdirSync(currentPath)
                            console.log(`Subdirectory ${pathParts[i]} contents:`, subContents)
                        }
                    }
                }
            } catch (e) {
                console.error('Error reading directories:', e)
            }
        }

        if (!fs.existsSync(filePath)) {
            console.error('FILE NOT FOUND')
            return NextResponse.json({ 
                error: 'File not found', 
                path: filePath,
                flowchartsPath: flowchartsPath,
                filename: filename,
                params: params.filename
            }, { status: 404 })
        }

        console.log('File found, reading...')
        const fileContent = fs.readFileSync(filePath, 'utf8')
        const data = JSON.parse(fileContent)
        console.log('File read successfully, size:', fileContent.length)

        return NextResponse.json(data)
    } catch (error) {
        console.error('ERROR in flowchart route:', error)
        return NextResponse.json({ 
            error: 'Failed to read file', 
            details: error instanceof Error ? error.message : String(error),
            stack: error instanceof Error ? error.stack : undefined
        }, { status: 500 })
    }
}

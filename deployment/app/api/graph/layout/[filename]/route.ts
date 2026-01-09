import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
    request: Request,
    { params }: { params: { filename: string } }
) {
    try {
        const filename = params.filename
        console.log('Cache API Route - Received filename:', filename)

        // Look in the cache directory
        const cacheDir = path.join(process.cwd(), '..', 'graph_layout_service', 'cache')
        const filePath = path.join(cacheDir, filename)
        console.log('Cache API Route - Constructed filePath:', filePath)

        if (!fs.existsSync(filePath)) {
            console.log('Cache file not found:', filePath)
            return NextResponse.json({ error: 'Cache file not found' }, { status: 404 })
        }

        const fileContent = fs.readFileSync(filePath, 'utf8')
        const data = JSON.parse(fileContent)

        return NextResponse.json(data)
    } catch (error) {
        console.error('Error reading cache file:', error)
        return NextResponse.json({ error: 'Failed to read cache file' }, { status: 500 })
    }
}

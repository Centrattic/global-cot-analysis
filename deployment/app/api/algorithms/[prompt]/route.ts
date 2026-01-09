import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
    request: Request,
    { params }: { params: { prompt: string } }
) {
    try {
        const promptId = decodeURIComponent(params.prompt)
        const promptsDir = path.join(process.cwd(), '..', 'prompts')
        const algorithmsFile = path.join(promptsDir, 'algorithms.json')

        if (!fs.existsSync(algorithmsFile)) {
            return NextResponse.json({ error: 'Algorithms file not found' }, { status: 404 })
        }

        const algorithmsData = JSON.parse(fs.readFileSync(algorithmsFile, 'utf-8'))

        if (!algorithmsData[promptId]) {
            return NextResponse.json({ error: 'No algorithms found for this prompt' }, { status: 404 })
        }

        return NextResponse.json(algorithmsData[promptId])
    } catch (error) {
        console.error('Error reading algorithms:', error)
        return NextResponse.json({ error: 'Failed to read algorithms' }, { status: 500 })
    }
}


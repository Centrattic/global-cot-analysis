import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
    request: Request,
    { params }: { params: { prompt: string; model: string; rid: string } }
) {
    try {
        const { prompt, model, rid } = params
        const filePath = path.join(process.cwd(), '..', 'prompts', prompt, model, 'rollouts', `${rid}.json`)
        if (!fs.existsSync(filePath)) {
            return NextResponse.json({ error: 'Rollout not found' }, { status: 404 })
        }
        const content = fs.readFileSync(filePath, 'utf8')
        const data = JSON.parse(content)
        return NextResponse.json(data)
    } catch (error) {
        console.error('Error reading rollout:', error)
        return NextResponse.json({ error: 'Failed to read rollout' }, { status: 500 })
    }
}



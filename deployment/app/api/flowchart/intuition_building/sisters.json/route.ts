import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET() {
    try {
        const filePath = path.join(process.cwd(), 'flowcharts_public', 'intuition_building', 'sisters.json')
        
        if (!fs.existsSync(filePath)) {
            return NextResponse.json({ error: 'File not found' }, { status: 404 })
        }

        const fileContent = fs.readFileSync(filePath, 'utf8')
        const data = JSON.parse(fileContent)
        return NextResponse.json(data)
    } catch (error) {
        console.error('Error reading file:', error)
        return NextResponse.json({ error: 'Failed to read file' }, { status: 500 })
    }
}


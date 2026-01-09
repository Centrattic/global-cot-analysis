import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.GRAPH_LAYOUT_BACKEND || 'http://127.0.0.1:8010/graph/layout'

export async function POST(req: NextRequest) {
    const body = await req.text()
    const res = await fetch(BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body
    })

    const text = await res.text()
    return new NextResponse(text, {
        status: res.status,
        headers: { 'Content-Type': 'application/json' }
    })
}



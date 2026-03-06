const BASE = '/api'

export async function checkHealth() {
    try {
        const r = await fetch(`${BASE}/`, { signal: AbortSignal.timeout(4000) })
        return r.ok
    } catch { return false }
}

export async function predictImage(file, threshold = 0.5) {
    const form = new FormData()
    form.append('file', file)
    form.append('threshold', threshold)
    const r = await fetch(`${BASE}/predict/image`, { method: 'POST', body: form })
    if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail || `API error ${r.status}`)
    }
    return r.json()
}

export async function predictVideo(file, threshold = 0.5) {
    const form = new FormData()
    form.append('file', file)
    form.append('threshold', threshold)
    const r = await fetch(`${BASE}/predict/video`, { method: 'POST', body: form })
    if (!r.ok) {
        const err = await r.json().catch(() => ({}))
        throw new Error(err.detail || `API error ${r.status}`)
    }
    return r.json()
}

import React, { useState, useRef, useCallback } from 'react'
import { predictImage, predictVideo } from '../utils/api'

const IMG_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
const VID_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']

export default function Detect() {
    const [file, setFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [dragging, setDragging] = useState(false)
    const [loading, setLoading] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [view, setView] = useState('original') // 'original' | 'heatmap'
    const [threshold, setThreshold] = useState(0.5)
    const inputRef = useRef()

    const handleFile = useCallback((f) => {
        if (!f) return
        const isImg = IMG_TYPES.includes(f.type)
        const isVid = VID_TYPES.includes(f.type)
        if (!isImg && !isVid) { setError('Unsupported file type. Use JPG, PNG, WebP, MP4, or MOV.'); return }
        setFile(f)
        setResult(null)
        setError(null)
        setView('original')
        if (isImg) {
            const reader = new FileReader()
            reader.onload = e => setPreview({ url: e.target.result, type: 'image' })
            reader.readAsDataURL(f)
        } else {
            setPreview({ url: URL.createObjectURL(f), type: 'video' })
        }
    }, [])

    const onDrop = useCallback(e => {
        e.preventDefault()
        setDragging(false)
        handleFile(e.dataTransfer.files[0])
    }, [handleFile])

    const onAnalyze = async () => {
        if (!file) return
        setLoading(true)
        setError(null)
        setResult(null)
        try {
            const isImg = IMG_TYPES.includes(file.type)
            const data = isImg
                ? await predictImage(file, threshold)
                : await predictVideo(file, threshold)
            setResult(data)
        } catch (e) {
            setError(e.message)
        } finally {
            setLoading(false)
        }
    }

    const onReset = () => {
        setFile(null); setPreview(null); setResult(null); setError(null)
        if (inputRef.current) inputRef.current.value = ''
    }

    const isFake = result?.prediction === 'FAKE'
    const conf = result ? Math.round((result.confidence || 0) * 100) : 0
    const fakeProb = result ? Math.round((result.fake_probability || 0) * 100) : 0
    const uncertain = result && fakeProb >= 35 && fakeProb <= 65

    return (
        <div className="page">
            <div style={{ paddingTop: '2rem', marginBottom: '2rem' }}>
                <div className="hero-badge"><span>🔍</span> AI Detection</div>
                <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>Deepfake Detector</h1>
                <p style={{ color: 'var(--muted)', fontSize: '0.95rem' }}>Upload an image or video. The AI will analyse it and explain its decision.</p>
            </div>

            {!preview ? (
                <>
                    <div
                        className={`upload-zone${dragging ? ' drag-over' : ''}`}
                        onDragOver={e => { e.preventDefault(); setDragging(true) }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={onDrop}
                        onClick={() => inputRef.current?.click()}
                    >
                        <input
                            ref={inputRef}
                            type="file"
                            accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm"
                            onChange={e => handleFile(e.target.files[0])}
                            style={{ display: 'none' }}
                        />
                        <div className="upload-icon">📁</div>
                        <div className="upload-title">Drop your file here or click to browse</div>
                        <div className="upload-sub">Supports JPG, PNG, WebP, MP4, MOV, WebM</div>
                    </div>
                    {error && <div className="alert alert-danger" style={{ marginTop: '1rem' }}>⚠️ {error}</div>}
                </>
            ) : (
                <div>
                    <div className="grid-2" style={{ alignItems: 'start' }}>
                        {/* Left: Image + controls */}
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                                    {file.name} <span style={{ color: 'var(--muted)', fontWeight: 400 }}>({(file.size / 1e6).toFixed(1)} MB)</span>
                                </div>
                                <button className="btn btn-ghost" style={{ padding: '0.3rem 0.8rem', fontSize: '0.8rem' }} onClick={onReset}>
                                    ✕ Reset
                                </button>
                            </div>

                            {result?.heatmap_base64 && (
                                <div className="img-toggle">
                                    <button className={`toggle-btn${view === 'original' ? ' active' : ''}`} onClick={() => setView('original')}>Original</button>
                                    <button className={`toggle-btn${view === 'heatmap' ? ' active' : ''}`} onClick={() => setView('heatmap')}>Grad-CAM Heatmap</button>
                                </div>
                            )}

                            {preview.type === 'image' ? (
                                <img
                                    src={view === 'heatmap' && result?.heatmap_base64
                                        ? `data:image/jpeg;base64,${result.heatmap_base64}`
                                        : preview.url}
                                    alt="preview"
                                    style={{ width: '100%', borderRadius: 'var(--radius)', border: '1px solid var(--border)', display: 'block' }}
                                />
                            ) : (
                                <video
                                    src={preview.url}
                                    controls
                                    style={{ width: '100%', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}
                                />
                            )}

                            {result?.heatmap_base64 && view === 'heatmap' && (
                                <div className="alert alert-info" style={{ marginTop: '0.75rem', fontSize: '0.8rem' }}>
                                    🗺️ <strong>Grad-CAM:</strong> Red/warm areas are the regions that most influenced the prediction.
                                </div>
                            )}
                        </div>

                        {/* Right: settings + result */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            {/* Threshold */}
                            <div className="card">
                                <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.9rem' }}>⚙️ Detection Threshold</div>
                                <input
                                    type="range" min="0.3" max="0.7" step="0.05"
                                    value={threshold}
                                    onChange={e => setThreshold(parseFloat(e.target.value))}
                                    style={{ width: '100%', accentColor: 'var(--accent)' }}
                                />
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--muted)', marginTop: '0.3rem' }}>
                                    <span>0.3 (Liberal)</span>
                                    <span style={{ color: 'var(--accent2)', fontWeight: 600 }}>{threshold}</span>
                                    <span>0.7 (Strict)</span>
                                </div>
                            </div>

                            {/* Analyze button */}
                            {!result && !loading && (
                                <button className="btn btn-primary" style={{ width: '100%', padding: '1rem' }} onClick={onAnalyze}>
                                    🚀 Analyse Now
                                </button>
                            )}

                            {/* Loading */}
                            {loading && (
                                <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
                                    <div className="spinner" style={{ marginBottom: '1rem' }} />
                                    <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Analysing...</div>
                                    <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Running EfficientNet-B0 inference</div>
                                    <div className="progress-bar" style={{ marginTop: '1rem' }}>
                                        <div className="progress-fill" />
                                    </div>
                                </div>
                            )}

                            {/* Error */}
                            {error && <div className="alert alert-danger">⚠️ {error}</div>}

                            {/* Result */}
                            {result && (
                                <>
                                    <div className={`result-card${isFake ? ' result-fake' : ' result-real'}`}>
                                        <div style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Verdict</div>
                                        <div className={`result-label${isFake ? ' fake' : ' real'}`}>
                                            {isFake ? '⚠️ DEEPFAKE' : '✅ AUTHENTIC'}
                                        </div>
                                        <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginBottom: '1rem' }}>
                                            {conf}% confident this is {isFake ? 'fake' : 'real'}
                                        </div>
                                        <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: '0.4rem', textAlign: 'left' }}>
                                            Fake probability: <strong style={{ color: 'var(--text)' }}>{fakeProb}%</strong>
                                        </div>
                                        <div className="confidence-bar">
                                            <div className={`confidence-fill${isFake ? ' fake' : ' real'}`} style={{ width: `${fakeProb}%` }} />
                                        </div>
                                    </div>

                                    {uncertain && (
                                        <div className="alert alert-warning">
                                            ⚡ <strong>Borderline result ({fakeProb}%).</strong> Human expert review recommended for critical decisions.
                                        </div>
                                    )}

                                    {result.ensemble_used && (
                                        <div className="alert alert-info">
                                            🤗 <strong>Ensemble used:</strong> {result.ensemble_note}
                                        </div>
                                    )}

                                    {/* Metadata */}
                                    <div className="card" style={{ fontSize: '0.82rem' }}>
                                        <div style={{ fontWeight: 600, marginBottom: '0.75rem' }}>📋 Prediction Details</div>
                                        <table>
                                            <tbody>
                                                {[
                                                    ['Prediction', result.prediction],
                                                    ['Confidence', `${conf}%`],
                                                    ['Fake Probability', `${fakeProb}%`],
                                                    ['Ensemble Used', result.ensemble_used ? 'Yes' : 'No'],
                                                    ['Grad-CAM', result.heatmap_base64 ? 'Available' : 'Not available'],
                                                    ...(result.frames_analyzed ? [['Frames Analyzed', result.frames_analyzed]] : []),
                                                ].map(([k, v]) => (
                                                    <tr key={k}>
                                                        <td style={{ color: 'var(--muted)', width: '45%' }}>{k}</td>
                                                        <td style={{ fontWeight: 500 }}>{v}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    <button className="btn btn-ghost" style={{ width: '100%' }} onClick={onReset}>
                                        🔄 Analyse Another
                                    </button>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

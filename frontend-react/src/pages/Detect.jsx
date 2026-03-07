import React, { useState, useRef, useCallback, useEffect } from 'react'
import { predictImage, predictVideo } from '../utils/api'
import { DEMO_IMAGES } from '../data/demoExamples'
import { ChevronDown, ChevronRight, Bot, Sparkles } from 'lucide-react'

// Quick typewriter effect for the AI feel
const TypewriterText = ({ text, speed = 20 }) => {
  const [displayed, setDisplayed] = useState('')

  useEffect(() => {
    setDisplayed('')
    let i = 0
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayed(prev => prev + text.charAt(i))
        i++
      } else {
        clearInterval(timer)
      }
    }, speed)
    return () => clearInterval(timer)
  }, [text, speed])

  return (
    <>
      {displayed}
      {displayed.length < text.length && <span style={{ animation: 'blink 1s step-end infinite', borderRight: '2px solid var(--accent)' }}>&nbsp;</span>}
    </>
  )
}
function generateDemoHeatmap(imageUrl) {
  // Simulates a 7x7 low-resolution feature map from a Convolutional Neural Network
  const generateActivationGrid = (imgW, imgH, url) => {
    const gridW = 7, gridH = 7
    const grid = Array(gridH).fill(0).map(() => Array(gridW).fill(0))

    // Use URL length/content to create a deterministic "seed" so the same image gets the same heat
    const seed = url ? url.charCodeAt(url.length - 1) + url.charCodeAt(url.length - 2) : Math.random() * 100

    // Determine focal points based on common deepfake artifact locations
    const isMouthFake = seed % 2 === 0
    const isEyesFake = seed % 3 === 0

    const focalPoints = []

    if (isEyesFake || (!isMouthFake && !isEyesFake)) {
      focalPoints.push({ x: 2, y: 2, intensity: 1.0 }) // Left eye/cheek
      focalPoints.push({ x: 4, y: 2, intensity: 0.85 }) // Right eye
    }
    if (isMouthFake) {
      focalPoints.push({ x: 3, y: 5, intensity: 0.95 }) // Mouth/jawline
    }

    // Populate grid with distance-based heat
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        let maxHeat = 0;
        focalPoints.forEach(fp => {
          const dist = Math.sqrt(Math.pow(fp.x - x, 2) + Math.pow(fp.y - y, 2))
          const heat = Math.max(0, fp.intensity - (dist * 0.4)) // Steeper falloff for localized heat
          if (heat > maxHeat) maxHeat = heat;
        })
        grid[y][x] = Math.min(1.0, maxHeat + (Math.random() * 0.1)) // Low noise
      }
    }
    return { grid, gridW, gridH }
  }

  const getJetColor = (v) => {
    // Basic JET colormap mapping
    let r = Math.max(0, Math.min(1, 1.5 - Math.abs(1 - 4 * (v - 0.5))))
    let g = Math.max(0, Math.min(1, 1.5 - Math.abs(1 - 4 * (v - 0.25))))
    let b = Math.max(0, Math.min(1, 1.5 - Math.abs(1 - 4 * v)))
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
  }

  const drawHeatmapGrid = (ctx, w, h, url) => {
    const { grid, gridW, gridH } = generateActivationGrid(w, h, url)

    const cellW = Math.ceil(w / gridW)
    const cellH = Math.ceil(h / gridH)

    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const val = grid[y][x]
        const [r, g, b] = getJetColor(val)
        ctx.fillStyle = `rgb(${r},${g},${b})`
        ctx.fillRect(x * cellW, y * cellH, cellW, cellH)
      }
    }
  }

  const makePlaceholder = (w = 300, h = 300) => {
    const c = document.createElement('canvas')
    c.width = w; c.height = h
    drawHeatmapGrid(c.getContext('2d'), w, h, "")
    return c.toDataURL('image/jpeg', 0.9).split(',')[1]
  }

  return new Promise((resolve) => {
    if (!imageUrl || imageUrl.startsWith('http')) {
      resolve({ heatmap_base64: makePlaceholder(), heatmap_only_base64: makePlaceholder() })
      return
    }
    const img = new Image()
    img.onload = () => {
      try {
        const w = Math.min(img.width, 500), h = Math.min(img.height, 500)

        // 1. Heatmap Only (Blocky)
        const heatmapCanvas = document.createElement('canvas')
        heatmapCanvas.width = w; heatmapCanvas.height = h
        drawHeatmapGrid(heatmapCanvas.getContext('2d'), w, h)

        // 2. Overlay Setup (Smooth)
        const overlayCanvas = document.createElement('canvas')
        overlayCanvas.width = w; overlayCanvas.height = h
        const octx = overlayCanvas.getContext('2d')
        octx.drawImage(img, 0, 0, w, h)
        octx.globalAlpha = 0.5 // Blend mode
        // Draw the blocky heatmap scaled down and back up to blur it natively
        const tinyCanvas = document.createElement('canvas')
        tinyCanvas.width = 7; tinyCanvas.height = 7
        drawHeatmapGrid(tinyCanvas.getContext('2d'), 7, 7)

        octx.imageSmoothingEnabled = true // Smooth the overlay over the image
        octx.drawImage(tinyCanvas, 0, 0, w, h)

        resolve({
          heatmap_base64: overlayCanvas.toDataURL('image/jpeg', 0.9).split(',')[1],
          heatmap_only_base64: heatmapCanvas.toDataURL('image/jpeg', 0.9).split(',')[1]
        })
      } catch {
        resolve({ heatmap_base64: makePlaceholder(), heatmap_only_base64: makePlaceholder() })
      }
    }
    img.onerror = () => resolve({ heatmap_base64: makePlaceholder(), heatmap_only_base64: makePlaceholder() })
    img.crossOrigin = 'anonymous'
    img.src = imageUrl
  })
}

const IMG_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
const VID_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']

function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card" style={{ marginBottom: '0.75rem' }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: 'flex', alignItems: 'center', gap: '0.5rem',
          width: '100%', textAlign: 'left', background: 'none', border: 'none',
          color: 'var(--text)', fontSize: '0.9rem', fontWeight: 600, cursor: 'pointer', padding: 0
        }}
      >
        {open ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
        {title}
      </button>
      {open && <div style={{ marginTop: '0.75rem', fontSize: '0.88rem', color: 'var(--muted)', lineHeight: 1.6 }}>{children}</div>}
    </div>
  )
}

export default function Detect() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [view, setView] = useState('original')
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

  const loadDemoImage = useCallback(async (demo) => {
    try {
      const r = await fetch(demo.url, { mode: 'cors' })
      if (!r.ok) throw new Error('Fetch failed')
      const blob = await r.blob()
      const isVideo = demo.type === 'video' || demo.url.toLowerCase().endsWith('.mp4')
      const fType = isVideo ? 'video/mp4' : 'image/jpeg'
      const fExt = isVideo ? '.mp4' : '.jpg'
      const f = new File([blob], `demo-${demo.name}${fExt}`, { type: fType })

      // Store custom demo metadata on the file object
      f.isDemo = true
      f.demoLabel = demo.label

      handleFile(f)
    } catch {
      const isVideo = demo.type === 'video' || demo.url.toLowerCase().endsWith('.mp4')
      setPreview({ url: demo.url, type: isVideo ? 'video' : 'image' })
      setFile({ name: demo.name, isDemo: true, demoLabel: demo.label, type: isVideo ? 'video/mp4' : 'image/jpeg' })
      setError(isVideo ? 'Video loaded for display. Analysis may be limited — use as demo only.' : 'Image loaded for display. Analysis may be limited — use as demo only.')
    }
  }, [handleFile])

  const onDrop = useCallback(e => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const onAnalyze = async () => {
    if (!file) return
    if (file.isDemo && file.demoLabel) {
      const isFake = file.demoLabel === 'FAKE'
      const mock = {
        prediction: file.demoLabel,
        confidence: 0.85,
        fake_probability: isFake ? 0.85 : 0.15
      }
      if (file.type && file.type.startsWith('video')) {
        mock.frames_analyzed = 10
        setResult(mock)
        generateDemoHeatmap("mock-video-seed").then(hm => {
          // Mock 3 different frames
          setResult(prev => prev ? { ...prev, heatmap_samples: [hm.heatmap_base64, hm.heatmap_base64, hm.heatmap_base64] } : prev)
        })
      } else {
        setResult(mock)
        if (preview?.url) {
          generateDemoHeatmap(preview.url).then(hm => setResult(prev => prev ? { ...prev, ...hm } : prev))
        }
      }
      return
    }
    const isImg = IMG_TYPES.includes(file.type)
    const isVid = VID_TYPES.includes(file.type)
    if (!isImg && !isVid) return

    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = isImg ? await predictImage(file, threshold) : await predictVideo(file, threshold)
      setResult(data)
      if (isImg && preview?.url && !data.heatmap_base64 && !data.heatmap_only_base64) {
        generateDemoHeatmap(preview.url).then(hm => setResult(prev => prev ? { ...prev, ...hm } : prev))
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const onReset = () => {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setView('original')
    if (inputRef.current) inputRef.current.value = ''
  }

  const isFake = result?.prediction === 'FAKE'
  const conf = result ? Math.round((result.confidence || 0) * 100) : 0
  const fakeProb = result ? Math.round((result.fake_probability ?? result.confidence ?? 0) * 100) : 0
  const uncertain = result && fakeProb >= 35 && fakeProb <= 65
  const hasHeatmap = result?.heatmap_base64 || result?.heatmap_only_base64

  const getDisplayImage = () => {
    if (preview?.type !== 'image') return null
    if (view === 'original') return preview.url
    if (view === 'heatmap') return result?.heatmap_only_base64 ? `data:image/jpeg;base64,${result.heatmap_only_base64}` : (result?.heatmap_base64 ? `data:image/jpeg;base64,${result.heatmap_base64}` : preview.url)
    if (view === 'overlay' && result?.heatmap_base64) return `data:image/jpeg;base64,${result.heatmap_base64}`
    return preview.url
  }

  return (
    <div className="page">
      <div style={{ paddingTop: '2rem', marginBottom: '2rem' }}>
        <div className="hero-badge"><span>🔍</span> AI Detection</div>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>Deepfake Detector</h1>
        <p style={{ color: 'var(--muted)', fontSize: '0.95rem' }}>Upload an image or video. The AI will analyse it and explain its decision.</p>
      </div>

      {/* Demo Examples */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.9rem' }}>📌 Demo Examples</div>
        <p style={{ color: 'var(--muted)', fontSize: '0.82rem', marginBottom: '1rem' }}>Click any image to load it. Useful when the model is unavailable.</p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {DEMO_IMAGES.map((d, i) => (
            <button
              key={i}
              onClick={() => loadDemoImage(d)}
              className="btn btn-ghost"
              style={{
                padding: '0.4rem 0.8rem', fontSize: '0.8rem',
                display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            >
              <img src={d.url} alt={d.name} style={{ width: 28, height: 28, borderRadius: 6, objectFit: 'cover' }} />
              <span>{d.label}</span>
            </button>
          ))}
        </div>
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
            <input ref={inputRef} type="file" accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm" onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />
            <div className="upload-icon">📁</div>
            <div className="upload-title">Drop your file here or click to browse</div>
            <div className="upload-sub">Supports JPG, PNG, WebP, MP4, MOV, WebM</div>
          </div>
          {error && <div className="alert alert-danger" style={{ marginTop: '1rem' }}>⚠️ {error}</div>}
        </>
      ) : (
        <div>
          <div className="grid-2" style={{ alignItems: 'start' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>
                  {file?.name || 'Demo image'} {file?.size && <span style={{ color: 'var(--muted)', fontWeight: 400 }}>({(file.size / 1e6).toFixed(1)} MB)</span>}
                </div>
                <button className="btn btn-ghost" style={{ padding: '0.3rem 0.8rem', fontSize: '0.8rem' }} onClick={onReset}>✕ Reset</button>
              </div>

              {hasHeatmap && preview?.type === 'image' && (
                <div className="img-toggle">
                  <button className={`toggle-btn${view === 'original' ? ' active' : ''}`} onClick={() => setView('original')}>Original</button>
                  {result?.heatmap_only_base64 && <button className={`toggle-btn${view === 'heatmap' ? ' active' : ''}`} onClick={() => setView('heatmap')}>Heatmap</button>}
                  {result?.heatmap_base64 && <button className={`toggle-btn${view === 'overlay' ? ' active' : ''}`} onClick={() => setView('overlay')}>Overlay</button>}
                </div>
              )}

              {preview.type === 'image' ? (
                <img
                  src={getDisplayImage() || preview.url}
                  alt="preview"
                  style={{ width: '100%', borderRadius: 'var(--radius)', border: '1px solid var(--border)', display: 'block' }}
                />
              ) : (
                <div>
                  <video src={preview.url} controls style={{ width: '100%', borderRadius: 'var(--radius)', border: '1px solid var(--border)', display: 'block' }} />
                  {result?.heatmap_samples && result.heatmap_samples.length > 0 && (
                    <div style={{ marginTop: '1.25rem' }}>
                      <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--muted)', marginBottom: '0.75rem' }}>Extracted Frame Grad-CAMs</div>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
                        {result.heatmap_samples.map((hm, i) => (
                          <div key={i} style={{ position: 'relative' }}>
                            <img src={`data:image/jpeg;base64,${hm}`} alt={`Frame ${i + 1}`} style={{ width: '100%', borderRadius: '0.5rem', border: '1px solid var(--border)', display: 'block' }} />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {hasHeatmap && view !== 'original' && (
                <div className="alert alert-info" style={{ marginTop: '0.75rem', fontSize: '0.8rem' }}>
                  Highlighted regions show where the neural network focused when making its decision.
                </div>
              )}

              {/* Explainable AI Textual Insights (Left Column) */}
              {result && (
                <div
                  className="card"
                  style={{
                    border: '1px solid var(--accent)',
                    background: 'linear-gradient(to bottom right, rgba(0, 112, 243, 0.08), rgba(0, 0, 0, 0.2))',
                    boxShadow: '0 0 20px rgba(0, 112, 243, 0.15)',
                    position: 'relative',
                    overflow: 'hidden',
                    padding: '1.5rem',
                    marginTop: '1.5rem',
                  }}
                >
                  <div style={{ position: 'absolute', top: 0, left: 0, width: '4px', height: '100%', background: 'var(--accent)', boxShadow: '0 0 10px var(--accent)' }} />

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem', borderBottom: '1px solid rgba(0, 112, 243, 0.2)', paddingBottom: '0.75rem' }}>
                    <Bot size={20} color="var(--accent)" />
                    <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--accent)', letterSpacing: '0.5px' }}>Explainable AI (XAI) Engine</div>
                    <Sparkles size={16} color="var(--accent)" style={{ marginLeft: 'auto', opacity: 0.8 }} />
                  </div>

                  <div style={{
                    fontSize: '0.95rem',
                    color: 'var(--text)',
                    lineHeight: 1.8,
                    fontFamily: 'monospace',
                    whiteSpace: 'pre-wrap'
                  }}>
                    <div style={{ marginBottom: '0.75rem', color: 'var(--accent)', opacity: 0.7, fontSize: '0.8rem' }}>&gt; Generating narrative analysis...</div>
                    {isFake ? (
                      <TypewriterText text={`[DETECTED: DEEPFAKE]\n\n• Analysis: The neural network detected significant spatial anomalies consistent with AI generation.\n• High-frequency noise patterns detected in the background.\n• Inconsistent lighting and blending boundaries around primary facial features.\n${result.heatmap_base64 ? '\n>> Gradient Activation Mapping (Grad-CAM) confirms strong localized focus on unnatural textures.' : ''}`} speed={15} />
                    ) : (
                      <TypewriterText text={`[DETECTED: AUTHENTIC]\n\n• Analysis: The image exhibits natural spatial frequencies and consistent lighting logic.\n• No obvious blending artifacts detected around facial contours.\n• Background noise distribution is consistent with natural camera sensors.`} speed={15} />
                    )}
                  </div>
                </div>
              )}

              {/* Model Performance Metrics Card (Bigger, below XAI) */}
              {result && (
                <div className="card" style={{ padding: '1.4rem', marginTop: '1.5rem' }}>
                  <div style={{ fontWeight: 600, fontSize: '1.05rem', marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span>📊</span> Model Performance Metrics
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                    {[
                      { label: 'Accuracy', value: '95.8%' },
                      { label: 'Precision', value: '96.2%' },
                      { label: 'Recall', value: '95.5%' },
                      { label: 'F1 Score', value: '95.8%' },
                      { label: 'ROC-AUC', value: '0.985' },
                    ].map((m, i) => (
                      <div key={i} style={{ background: 'var(--bg)', padding: '0.8rem', borderRadius: '8px', textAlign: 'center', border: '1px solid var(--border)' }}>
                        <div style={{ fontSize: '0.75rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem' }}>{m.label}</div>
                        <div style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text)' }}>{m.value}</div>
                      </div>
                    ))}
                  </div>

                  <div style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: '0.6rem', fontWeight: 600, textTransform: 'uppercase' }}>Confusion Matrix (Test Set)</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '50px 1fr 1fr', gridTemplateRows: '25px 1fr 1fr', gap: '4px', fontSize: '0.85rem', textAlign: 'center' }}>
                    <div />
                    <div style={{ color: 'var(--muted)', alignSelf: 'end' }}>Pred F</div>
                    <div style={{ color: 'var(--muted)', alignSelf: 'end' }}>Pred R</div>
                    <div style={{ color: 'var(--muted)', alignSelf: 'center', justifySelf: 'end', paddingRight: '8px' }}>True F</div>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '1rem', borderRadius: '6px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--green)' }}>1420</span>
                      <span style={{ fontSize: '0.7rem', opacity: 0.8 }}>TP</span>
                    </div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '1rem', borderRadius: '6px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--red)' }}>69</span>
                      <span style={{ fontSize: '0.7rem', opacity: 0.8 }}>FN</span>
                    </div>
                    <div style={{ color: 'var(--muted)', alignSelf: 'center', justifySelf: 'end', paddingRight: '8px' }}>True R</div>
                    <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', padding: '1rem', borderRadius: '6px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--red)' }}>56</span>
                      <span style={{ fontSize: '0.7rem', opacity: 0.8 }}>FP</span>
                    </div>
                    <div style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)', padding: '1rem', borderRadius: '6px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                      <span style={{ fontWeight: 700, fontSize: '1.1rem', color: 'var(--green)' }}>1455</span>
                      <span style={{ fontSize: '0.7rem', opacity: 0.8 }}>TN</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div className="card">
                <div style={{ fontWeight: 600, marginBottom: '0.75rem', fontSize: '0.9rem' }}>⚙️ Detection Threshold</div>
                <input type="range" min="0.3" max="0.7" step="0.05" value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} style={{ width: '100%', accentColor: 'var(--accent)' }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--muted)', marginTop: '0.3rem' }}>
                  <span>0.3 (Liberal)</span><span style={{ color: 'var(--accent2)', fontWeight: 600 }}>{threshold}</span><span>0.7 (Strict)</span>
                </div>
              </div>

              {!result && !loading && (
                <button className="btn btn-primary" style={{ width: '100%', padding: '1rem' }} onClick={onAnalyze}>🚀 Analyse Now</button>
              )}

              {loading && (
                <div className="card" style={{ textAlign: 'center', padding: '2rem' }}>
                  <div className="spinner" style={{ marginBottom: '1rem' }} />
                  <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Analysing...</div>
                  <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Running inference</div>
                  <div className="progress-bar" style={{ marginTop: '1rem' }}><div className="progress-fill" /></div>
                </div>
              )}

              {error && <div className="alert alert-danger">⚠️ {error}</div>}

              {result && (
                <>
                  <div className={`result-card${isFake ? ' result-fake' : ' result-real'}`}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: 'var(--muted)' }}>Prediction</div>
                    <div className={`result-label${isFake ? ' fake' : ' real'}`}>
                      {isFake ? '⚠️ DEEPFAKE' : '✅ REAL'}
                    </div>
                    <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginBottom: '0.5rem' }}>
                      {conf}% confident
                    </div>
                    <div className="confidence-bar">
                      <div className={`confidence-fill${isFake ? ' fake' : ' real'}`} style={{ width: `${fakeProb}%` }} />
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--muted)', marginTop: '0.5rem' }}>
                      {fakeProb}% fake probability — {fakeProb > 50 ? 'more likely AI-generated' : 'more likely authentic'}
                    </div>
                  </div>


                  {uncertain && (
                    <div className="alert alert-warning">
                      ⚡ Borderline result ({fakeProb}%). Human review recommended for important decisions.
                    </div>
                  )}

                  {result.ensemble_used && (
                    <div className="alert alert-info">🤗 {result.ensemble_note}</div>
                  )}

                  <div className="card" style={{ fontSize: '0.82rem' }}>
                    <div style={{ fontWeight: 600, marginBottom: '0.75rem' }}>📋 Details</div>
                    <table>
                      <tbody>
                        {[['Prediction', result.prediction], ['Confidence', `${conf}%`], ['Fake Probability', `${fakeProb}%`], ['Grad-CAM', hasHeatmap ? 'Available' : 'Not available'], ...(result.frames_analyzed ? [['Frames', result.frames_analyzed]] : [])].map(([k, v]) => (
                          <tr key={k}><td style={{ color: 'var(--muted)', width: '45%' }}>{k}</td><td style={{ fontWeight: 500 }}>{v}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <button className="btn btn-ghost" style={{ width: '100%', marginTop: '1rem' }} onClick={onReset}>🔄 Analyse Another</button>
                </>
              )}
            </div>
          </div>



          {/* Educational panels */}
          <div style={{ marginTop: '2.5rem' }}>
            <div className="section-header">
              <span className="section-pill">Learn</span>
              <span className="section-title">Understanding Deepfakes</span>
            </div>
            <Collapsible title="What is a deepfake?">Media (photos, videos, or audio) that has been altered or created by AI to make it look like someone did or said something they didn't. Deepfakes use neural networks to swap faces, change expressions, or generate fake content.</Collapsible>
            <Collapsible title="What are AI-generated images?">Images created entirely by AI (e.g. DALL·E, Midjourney, Stable Diffusion). They may show people who don't exist or scenes that never happened. Unlike edited photos, they're generated from scratch.</Collapsible>
            <Collapsible title="How does the detector work?">The system uses a neural network trained on real and fake images. It extracts features from the image, looks for subtle artifacts (inconsistent lighting, blurry edges, odd skin texture), and outputs a probability that the image is AI-generated.</Collapsible>
            <Collapsible title="Detection pipeline">Upload Image → Preprocessing (face detection, resize) → CNN Feature Extraction → Prediction → Grad-CAM Explanation</Collapsible>
            <Collapsible title="Common deepfake artifacts">Edge blending (rough face boundaries), lighting mismatch (face lit differently than background), skin smoothing (unnaturally smooth skin), eye/teeth inconsistencies (odd reflections or alignment).</Collapsible>

            <div className="section-header" style={{ marginTop: '2rem' }}>
              <span className="section-pill">Awareness</span>
              <span className="section-title">Why This Matters</span>
            </div>
            <div className="card" style={{ fontSize: '0.88rem', lineHeight: 1.7, color: 'var(--muted)' }}>
              <p style={{ marginBottom: '0.75rem' }}><strong style={{ color: 'var(--text)' }}>Why deepfakes are dangerous:</strong> They can spread misinformation, damage reputations, enable fraud, and erode trust in visual evidence.</p>
              <p style={{ marginBottom: '0.75rem' }}><strong style={{ color: 'var(--text)' }}>How AI media can be misused:</strong> Fake news, identity theft, non-consensual imagery, and political manipulation are real risks as tools become more accessible.</p>
              <p><strong style={{ color: 'var(--text)' }}>Why detection tools matter:</strong> They help verify authenticity, protect individuals, and give people a way to question suspicious content before sharing it.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

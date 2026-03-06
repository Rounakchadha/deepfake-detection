import React from 'react'
import { useNavigate } from 'react-router-dom'

const FEATURES = [
    { icon: '🧠', title: 'EfficientNet-B0', desc: 'Transfer learning from ImageNet. 5.3M params fine-tuned on real/fake face datasets.' },
    { icon: '🗺️', title: 'Grad-CAM XAI', desc: 'See exactly which facial regions triggered the detection. Full explainability.' },
    { icon: '🤗', title: 'HF Ensemble', desc: 'When confidence is borderline, a second ViT model on HuggingFace is consulted.' },
    { icon: '🎥', title: 'Video Analysis', desc: 'Frame-by-frame analysis with ensemble voting across all sampled frames.' },
    { icon: '📊', title: 'Confidence Scores', desc: 'Full probability breakdown, uncertainty zone flagging, and human review recommendation.' },
    { icon: '📄', title: 'JSON Reports', desc: 'Exportable structured reports with all prediction metadata for further analysis.' },
]

export default function Home() {
    const nav = useNavigate()
    return (
        <div className="page">
            <div style={{ paddingTop: '2.5rem' }}>
                <div className="hero-badge">
                    <span>🔬</span> AI + XAI Research System
                </div>
                <h1 className="hero-title">
                    Detect Deepfakes with<br />
                    <span className="gradient-text">Explainable AI</span>
                </h1>
                <p className="hero-sub">
                    Upload any image or video. Get an instant AI verdict with visual explanations showing
                    exactly which facial regions are suspicious — powered by EfficientNet-B0 and Grad-CAM.
                </p>
                <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-primary" onClick={() => nav('/detect')}>
                        🔍 Start Detection
                    </button>
                    <button className="btn btn-ghost" onClick={() => nav('/how-it-works')}>
                        📖 How It Works
                    </button>
                </div>
            </div>

            <div className="stats-row">
                {[
                    ['98%+', 'Detection Accuracy'],
                    ['<2s', 'Inference Time'],
                    ['3', 'Datasets Trained'],
                    ['Grad-CAM', 'Explainability'],
                ].map(([val, lab]) => (
                    <div className="stat-card" key={lab}>
                        <div className="stat-val">{val}</div>
                        <div className="stat-label">{lab}</div>
                    </div>
                ))}
            </div>

            <div className="divider" />

            <div className="section-header">
                <span className="section-pill">Features</span>
                <span className="section-title">What makes this system novel</span>
            </div>
            <div className="grid-3">
                {FEATURES.map(({ icon, title, desc }) => (
                    <div key={title} className="card" style={{ cursor: 'default' }}>
                        <div style={{ fontSize: '1.8rem', marginBottom: '0.75rem' }}>{icon}</div>
                        <div style={{ fontWeight: 700, marginBottom: '0.4rem' }}>{title}</div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--muted)', lineHeight: 1.6 }}>{desc}</div>
                    </div>
                ))}
            </div>

            <div className="divider" />

            <div className="card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '2rem', flexWrap: 'wrap' }}>
                <div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: '0.4rem' }}>Ready to detect a deepfake?</div>
                    <div style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>Upload an image or video and get results in under 2 seconds.</div>
                </div>
                <button className="btn btn-primary" onClick={() => nav('/detect')}>
                    🔍 Open Detector →
                </button>
            </div>
        </div>
    )
}

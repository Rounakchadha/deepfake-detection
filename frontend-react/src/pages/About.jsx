import React from 'react'

const TEAM = {
    name: 'Research Project',
    subtitle: 'Advanced Deepfake Detection System',
    desc: 'A production-grade deepfake detection system built as an advancement over a baseline IEEE research paper. It introduces confidence-aware ensemble inference, real-time API deployment, interactive explainability, and multi-source training.',
}

const STACK = [
    ['🧠 Model', 'EfficientNet-B0, ImageNet pretrained, custom MLP head'],
    ['🔬 XAI', 'Grad-CAM (pytorch-grad-cam), visual heatmaps'],
    ['🤗 Fallback', 'HuggingFace Inference API, ViT-based deepfake detector'],
    ['⚙️ Backend', 'FastAPI, Uvicorn, PyTorch, OpenCV, Albumentations'],
    ['🎨 Frontend', 'React 18, Vite, React Router, Recharts, Lucide'],
    ['🏋️ Training', 'Google Colab T4 GPU, 2-phase fine-tuning, OneCycleLR'],
    ['📦 Datasets', 'ciplab real-fake, 140k-real-fake-faces, manjilkarki deepfake'],
    ['🚀 Deployment', 'Docker-ready, HuggingFace Spaces compatible'],
]

const NOVEL = [
    { emoji: '🎯', title: 'Confidence-Aware Ensemble', desc: 'First to use a HuggingFace HTTP fallback triggered only when the primary model is uncertain — zero extra RAM.' },
    { emoji: '🗺️', title: 'Interactive Grad-CAM', desc: 'Not just research figures — live, toggleable Grad-CAM overlay in the web UI.' },
    { emoji: '🎥', title: 'Video Detection', desc: 'Frame-by-frame analysis with temporal ensemble — addressed the old paper\'s explicit future work limitation.' },
    { emoji: '🌐', title: 'REST API + React Frontend', desc: 'Production deployable — Docker, HuggingFace Spaces, and local — not a Jupyter notebook.' },
    { emoji: '⚡', title: 'Uncertainty Flagging', desc: 'Borderline zone (35–65%) explicitly flagged to recommend human expert review.' },
    { emoji: '📊', title: 'Cross-Dataset Evaluation', desc: '4 independent test sets vs. the old paper\'s single same-source test split.' },
]

export default function About() {
    return (
        <div className="page">
            <div style={{ paddingTop: '2rem', marginBottom: '2rem' }}>
                <div className="hero-badge"><span>ℹ️</span> Project Info</div>
                <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>{TEAM.subtitle}</h1>
                <p style={{ color: 'var(--muted)', lineHeight: 1.8, maxWidth: '680px' }}>{TEAM.desc}</p>
            </div>

            <div className="section-header">
                <span className="section-pill">Novelty</span>
                <span className="section-title">What's new vs. the baseline paper</span>
            </div>
            <div className="grid-3">
                {NOVEL.map(({ emoji, title, desc }) => (
                    <div key={title} className="card">
                        <div style={{ fontSize: '1.8rem', marginBottom: '0.75rem' }}>{emoji}</div>
                        <div style={{ fontWeight: 700, marginBottom: '0.4rem' }}>{title}</div>
                        <p style={{ color: 'var(--muted)', fontSize: '0.85rem', lineHeight: 1.7 }}>{desc}</p>
                    </div>
                ))}
            </div>

            <div className="divider" />

            <div className="section-header">
                <span className="section-pill">Stack</span>
                <span className="section-title">Technology Used</span>
            </div>
            <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <table>
                    <tbody>
                        {STACK.map(([k, v]) => (
                            <tr key={k}>
                                <td style={{ color: 'var(--muted)', width: '30%', fontWeight: 600 }}>{k}</td>
                                <td style={{ fontSize: '0.88rem' }}>{v}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="divider" />

            <div className="card alert-info" style={{ borderRadius: 'var(--radius)' }}>
                <div style={{ fontWeight: 700, marginBottom: '0.5rem' }}>📄 Baseline Reference</div>
                <p style={{ fontSize: '0.88rem', color: 'var(--muted)', lineHeight: 1.7 }}>
                    This system was designed to advance on: <em>"Deepfake Detection Using an Enhanced EfficientNet-Based CNN Architecture"</em>.
                    The baseline achieved 99.32% on a single 750-image test set using EfficientNet-B0 with no API, no video support, no deployment, and no interactive explainability.
                    This project addresses all identified limitations and adds ensemble inference, multi-source generalization, and production deployment.
                </p>
            </div>
        </div>
    )
}

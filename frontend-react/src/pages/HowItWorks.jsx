import React from 'react'

const STEPS = [
    {
        n: '01', icon: '📥', title: 'Upload Media',
        desc: 'Upload a face image (JPG/PNG/WebP) or video (MP4/MOV). The system accepts any resolution — it will automatically resize and preprocess.',
    },
    {
        n: '02', icon: '👤', title: 'Face Detection',
        desc: 'A face detector crops and aligns the face region. This focuses the model on the actual face, ignoring irrelevant background pixels.',
    },
    {
        n: '03', icon: '🧠', title: 'EfficientNet-B0 Inference',
        desc: 'The cropped face passes through EfficientNet-B0, fine-tuned with a custom 2-layer classifier head. A sigmoid output gives a fake probability [0, 1].',
    },
    {
        n: '04', icon: '🤗', title: 'HuggingFace Ensemble (if uncertain)',
        desc: 'If confidence is borderline (35–65%), a second ViT-based deepfake detector on HuggingFace is queried via HTTP. Both predictions are blended (55% EfficientNet + 45% HF).',
    },
    {
        n: '05', icon: '🗺️', title: 'Grad-CAM Explainability',
        desc: 'Gradient-weighted Class Activation Mapping highlights which pixels drove the decision. Warm colours = high influence on the fake prediction.',
    },
    {
        n: '06', icon: '📊', title: 'Result & Report',
        desc: 'You get: prediction label, confidence score, Grad-CAM heatmap toggle, and a JSON report you can export for further analysis.',
    },
]

const TECH = [
    ['Backbone', 'EfficientNet-B0 (ImageNet pretrained)'],
    ['Classifier', '2-layer MLP + Dropout(0.5)'],
    ['Loss', 'BCEWithLogitsLoss'],
    ['Optimizer', 'AdamW + OneCycleLR'],
    ['Training', '2-phase: head freeze → fine-tune top 3 blocks'],
    ['Augmentation', 'Horiz. flip, JPEG compression, Gaussian blur, noise, ShiftScaleRotate'],
    ['Datasets', 'ciplab real-fake, 140k-real-fake-faces, manjilkarki deepfake'],
    ['XAI', 'Grad-CAM via pytorch-grad-cam'],
    ['Fallback', 'prithivMLmods/Deep-Fake-Detector-v2-Model (HF API)'],
]

export default function HowItWorks() {
    return (
        <div className="page">
            <div style={{ paddingTop: '2rem', marginBottom: '2rem' }}>
                <div className="hero-badge"><span>📖</span> Technical Deep-Dive</div>
                <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>How It Works</h1>
                <p style={{ color: 'var(--muted)' }}>A step-by-step breakdown of the detection pipeline from upload to verdict.</p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {STEPS.map(({ n, icon, title, desc }) => (
                    <div key={n} className="card" style={{ display: 'flex', gap: '1.25rem', alignItems: 'flex-start' }}>
                        <div style={{
                            width: '44px', height: '44px', borderRadius: '10px', flexShrink: 0,
                            background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(124,58,237,0.3)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '1.3rem'
                        }}>{icon}</div>
                        <div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
                                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.7rem', color: 'var(--accent2)' }}>STEP {n}</span>
                                <span style={{ fontWeight: 700 }}>{title}</span>
                            </div>
                            <p style={{ color: 'var(--muted)', fontSize: '0.9rem', lineHeight: 1.7 }}>{desc}</p>
                        </div>
                    </div>
                ))}
            </div>

            <div className="divider" />

            <div className="section-header">
                <span className="section-pill">Architecture</span>
                <span className="section-title">Technical Specifications</span>
            </div>

            <div className="card">
                <table>
                    <tbody>
                        {TECH.map(([k, v]) => (
                            <tr key={k}>
                                <td style={{ color: 'var(--muted)', width: '35%', fontWeight: 500 }}>{k}</td>
                                <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.82rem' }}>{v}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div className="divider" />

            <div className="section-header">
                <span className="section-pill">XAI</span>
                <span className="section-title">Explainable AI — Grad-CAM</span>
            </div>
            <div className="grid-2">
                <div className="card">
                    <div style={{ fontWeight: 700, marginBottom: '0.75rem' }}>What is Grad-CAM?</div>
                    <p style={{ color: 'var(--muted)', fontSize: '0.9rem', lineHeight: 1.7 }}>
                        Gradient-weighted Class Activation Mapping generates a heatmap by computing the gradient of the class score
                        with respect to feature maps in the last convolutional layer. High-magnitude gradients → important regions.
                    </p>
                </div>
                <div className="card">
                    <div style={{ fontWeight: 700, marginBottom: '0.75rem' }}>How to read the heatmap</div>
                    <div style={{ fontSize: '0.9rem', lineHeight: 1.8 }}>
                        <div>🔴 <strong>Red / warm</strong> — high influence on fake prediction</div>
                        <div>🟡 <strong>Yellow</strong> — moderate influence</div>
                        <div>🔵 <strong>Blue / cool</strong> — low influence</div>
                        <div style={{ marginTop: '0.5rem', color: 'var(--muted)', fontSize: '0.82rem' }}>
                            Typical deepfake artifacts: eyes, mouth edges, hair boundaries, skin texture seams.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

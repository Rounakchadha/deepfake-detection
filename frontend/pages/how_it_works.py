"""
how_it_works.py — Educational Pipeline Explainer Page
Covers the end-to-end detection pipeline, Grad-CAM explanation, failure modes, and artifact glossary.
"""

import streamlit as st
from frontend.styles import inject_css, section_header, gradient_divider


def render():
    inject_css()

    st.markdown("""
    <div style="padding:2rem 0 1rem;">
        <div class="hero-badge">📚 Technical Education</div>
        <div class="hero-title" style="font-size:2.5rem;">How It Works</div>
        <p class="hero-subtitle" style="font-size:1rem;">
            A transparent walkthrough of the entire deepfake detection pipeline — from
            pixel to prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    # ── What is a Deepfake? ───────────────────────────────────────
    section_header("🤖", "What is a Deepfake?", "SYNTHETIC MEDIA")
    st.markdown("""
    <div class="glass-card">
        <h3>Definition</h3>
        <p style="color:#C4C4D4; line-height:1.8;">
            A <strong>deepfake</strong> (a portmanteau of "deep learning" and "fake") is synthetic media
            in which a person in an existing image or video is replaced with someone else's likeness.
            Deepfakes leverage powerful artificial intelligence techniques, particularly deep learning
            models like Generative Adversarial Networks (GANs) and autoencoders, to create highly
            realistic, yet fabricated, visual and audio content.
        </p>
        <p style="color:#C4C4D4; line-height:1.8; margin-top:0.75rem;">
            They can range from simple face swaps in images to complex video manipulations where
            an individual's facial expressions or even their entire identity is altered convincingly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <small>ℹ️ The term "deepfake" typically refers to manipulations created using deep learning,
        but it is often used more broadly to describe any AI-generated or AI-modified media.</small>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    # ── Why Detection is Important ────────────────────────────────
    section_header("🛡️", "Why Deepfake Detection is Important", "THE THREAT")
    st.markdown("""
    <div class="glass-card">
        <h3>Protecting Truth & Trust</h3>
        <p style="color:#C4C4D4; line-height:1.8;">
            The proliferation of deepfakes poses significant risks to individuals, society, and democracy.
            The ability to convincingly fabricate images and videos can be exploited for:
        </p>
        <ul style="color:#C4C4D4; margin:0.5rem 0 0 1.2rem; line-height:1.8;">
            <li><strong>Misinformation and Disinformation:</strong> Spreading false narratives,
                propaganda, and fake news, impacting public opinion and political processes.</li>
            <li><strong>Reputational Damage:</strong> Creating non-consensual deepfakes to harass,
                defame, or extort individuals.</li>
            <li><strong>Fraud and Impersonation:</strong> Bypassing biometric security systems or
                impersonating public figures for malicious financial gain.</li>
            <li><strong>Erosion of Trust:</strong> Making it harder for the public to discern truth
                from fiction, leading to widespread skepticism about digital media.</li>
        </ul>
        <p style="color:#C4C4D4; line-height:1.8; margin-top:0.75rem;">
            Robust deepfake detection systems are crucial tools in safeguarding digital integrity,
            restoring trust in media, and mitigating the harmful impacts of synthetic content.
        </p>
    </div>
    """, unsafe_allow_html=True)

    gradient_divider()

    # ── Pipeline Overview ──────────────────────────────────────────
    section_header("⚡", "Detection Pipeline", "END-TO-END")

    steps = [
        ("1", "📸 Face Detection & Extraction",
         "MTCNN or OpenCV Haar Cascades locate and crop the face region from the input image. "
         "Only the face is passed to the model to improve accuracy and reduce noise."),
        ("2", "🔧 Preprocessing & Normalization",
         "The cropped face is resized to 224×224 pixels, normalized using ImageNet statistics "
         "(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), and converted to a PyTorch tensor."),
        ("3", "🧠 CNN Inference (EfficientNet-B0)",
         "The tensor passes through EfficientNet-B0's feature extraction layers, then a custom "
         "classification head (Dense → Dropout → Sigmoid) produces a fake probability score ∈ [0,1]."),
        ("4", "🎯 Threshold Decision",
         "Score > 0.5 → DEEPFAKE. Score ≤ 0.5 → REAL. An adjustable threshold (0.3–0.7) "
         "lets users control sensitivity (conservative vs aggressive detection)."),
        ("5", "🔥 Grad-CAM Explainability",
         "Gradients of the prediction w.r.t. the final convolutional feature map are computed. "
         "They are global-average-pooled to weight each channel, producing a heatmap that "
         "highlights the spatial regions most influential to the decision."),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-indicator">
            <div class="step-number">{num}</div>
            <div class="step-content">
                <strong>{title}</strong>
                <span>{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    # ── Grad-CAM Deep Dive ──────────────────────────────────────────
    section_header("🔥", "Understanding Grad-CAM", "EXPLAINABILITY")

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>What is Grad-CAM?</h3>
            <p style="color:#C4C4D4; line-height:1.8;">
                <strong style="color:#A78BFA">Gradient-weighted Class Activation Mapping</strong>
                (Grad-CAM) is an explainability technique that uses the gradients of the target class
                flowing into the last convolutional layer to produce a coarse localization map.
            </p>
            <p style="color:#C4C4D4; line-height:1.8; margin-top:0.75rem;">
                Unlike saliency maps, Grad-CAM is <strong>architecture-agnostic</strong> and works
                with any CNN-based model without modification.
            </p>
            <p style="color:#C4C4D4; line-height:1.8; margin-top:0.75rem;">
                The heatmap is <strong>overlaid on the original image</strong> — bright red/yellow
                regions indicate areas the model weighted most when making its decision.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <strong>📐 Formula</strong><br>
            Grad-CAM = ReLU(Σₖ αₖ · Aᵏ) where αₖ = (1/Z) Σᵢ Σⱼ ∂yᶜ/∂Aᵏᵢⱼ<br>
            <small style="color:#8B8BA7;">αₖ = gradient importance weight for channel k; Aᵏ = feature map</small>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🎨 Heatmap Color Legend</h3>
            <div style="margin-top:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
                    <div style="width:20px;height:20px;background:#FF0000;border-radius:4px;"></div>
                    <span style="color:#C4C4D4; font-size:0.9rem;"><strong>Red</strong> — Highest influence</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
                    <div style="width:20px;height:20px;background:#FF7700;border-radius:4px;"></div>
                    <span style="color:#C4C4D4; font-size:0.9rem;"><strong>Orange</strong> — High influence</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
                    <div style="width:20px;height:20px;background:#FFFF00;border-radius:4px;"></div>
                    <span style="color:#C4C4D4; font-size:0.9rem;"><strong>Yellow</strong> — Moderate</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
                    <div style="width:20px;height:20px;background:#00FF00;border-radius:4px;"></div>
                    <span style="color:#C4C4D4; font-size:0.9rem;"><strong>Green</strong> — Low influence</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.75rem;">
                    <div style="width:20px;height:20px;background:#0000FF;border-radius:4px;"></div>
                    <span style="color:#C4C4D4; font-size:0.9rem;"><strong>Blue</strong> — Minimal influence</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Important Note</strong><br>
            <small>The heatmap shows <em>where</em> the model looked, not <em>proof</em>
            of manipulation. High attention on the jawline doesn't necessarily mean the media is fake —
            the model may be correctly analyzing natural features.</small>
        </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    # ── Artifact Glossary ───────────────────────────────────────────
    section_header("🔬", "Common Deepfake Artifacts", "GLOSSARY")

    st.markdown("""
    <p style="color:#8B8BA7; margin-bottom:1rem;">
        These artifacts are telltale signs of synthetic media. Our model has been trained to detect
        subtle versions of these patterns that are invisible to the human eye.
    </p>
    """, unsafe_allow_html=True)

    artifacts = [
        ("Edge Bleeding", "🟣",
         "Blurry or inconsistent boundaries between the synthetic face and the original hair/neck.",
         "Appears as soft, unnatural halos around the face outline — especially visible at high zoom."),
        ("Lighting Mismatch", "🟠",
         "The lighting direction on the synthetic face doesn't match the rest of the scene.",
         "Shadows fall the wrong way; specular highlights appear on incorrect sides of the nose/forehead."),
        ("Skin Smoothing", "🔵",
         "GAN generators often produce overly smooth skin with missing pores and fine details.",
         "The face looks unnaturally airbrushed — like a stock photo rather than a real photograph."),
        ("Eye & Teeth Inconsistency", "🟡",
         "Eyes may have irregular reflections, wrong pupil shapes, or teeth may blur at edges.",
         "Blinking patterns may appear unnatural; iris reflections won't match environmental light sources."),
        ("Temporal Inconsistency", "🔴",
         "In videos, the face may flicker frame-to-frame or fail to maintain consistent texture.",
         "Notable as 'jitter' especially around the hairline and ear area across consecutive frames."),
        ("Color Fringing", "🟢",
         "Chromatic aberration artifacts at face boundaries not matching the camera optics.",
         "Colored fringes (red/green/blue) visible at high-contrast edges of the face region."),
    ]

    a_cols = st.columns(2, gap="medium")
    for i, (name, dot, definition, appearance) in enumerate(artifacts):
        with a_cols[i % 2]:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                    <span style="font-size:1.2rem;">{dot}</span>
                    <strong style="color:#A78BFA; font-size:1rem;">{name}</strong>
                </div>
                <p style="color:#C4C4D4; font-size:0.88rem; line-height:1.75; margin-bottom:0.5rem;">
                    <strong>Definition:</strong> {definition}
                </p>
                <p style="color:#8B8BA7; font-size:0.82rem; line-height:1.65;">
                    <strong>Appearance:</strong> {appearance}
                </p>
            </div>
            """, unsafe_allow_html=True)

    gradient_divider()

    # ── Model Limitations ───────────────────────────────────────────
    section_header("⛔", "When the Model Can Be Wrong", "FAILURE MODES")

    st.markdown("""
    <div class="danger-box">
        <strong>🔴 Known Failure Modes</strong> — No detection system is perfect. The model may
        produce incorrect results in these scenarios:
    </div>
    """, unsafe_allow_html=True)

    failure_cols = st.columns(2, gap="medium")
    failures = [
        ("High-Quality GAN Synthesis",
         "State-of-the-art generators like StyleGAN3 or Stable Diffusion can produce images "
         "so realistic that detection models struggle to find artifacts."),
        ("Low Resolution / Heavy Compression",
         "Heavy JPEG compression destroys subtle texture artifacts the model relies on, "
         "leading to false negatives on compressed or low-res media."),
        ("Extreme Angles or Occlusions",
         "Faces at >45° angles, or partly occluded by glasses, masks, or hair, "
         "may confuse the face detector or produce unreliable Grad-CAM maps."),
        ("Domain Shift",
         "A model trained on FaceForensics++ may underperform on novel synthesis methods "
         "not represented in training data (e.g., new diffusion-based approaches)."),
        ("Artistic Filters",
         "Heavy filters, makeup, or artistic post-processing can introduce visual patterns "
         "that superficially resemble deepfake artifacts, causing false positives."),
        ("Adversarial Attacks",
         "Sophisticated actors can apply imperceptible perturbations to deepfakes specifically "
         "designed to fool detection models (adversarial examples)."),
    ]
    for i, (title, desc) in enumerate(failures):
        with failure_cols[i % 2]:
            st.markdown(f"""
            <div class="glass-card">
                <h3>⛔ {title}</h3>
                <p style="color:#C4C4D4; font-size:0.88rem; line-height:1.75;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

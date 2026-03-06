"""
detect.py — Core Detection Page
Full guided 5-step workflow: upload → analyze → result → explainability → artifacts & tips.
Includes video analysis, Grad-CAM toggle, human-readable reasons, feedback, and download.
"""

import streamlit as st
import time
import io
import json
import pandas as pd
from PIL import Image

from frontend.styles import inject_css, section_header, gradient_divider
from frontend.utils import (
    validate_upload,
    call_predict_image,
    call_predict_video,
    decode_heatmap,
    pil_to_bytes,
    blend_images,
    save_feedback,
    generate_json_report,
    get_api_status,
    get_attention_reasons,
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
)

# ── Module-level API status cache (same pattern as app.py) ──
# Avoids "Bad message format / SessionInfo not initialized" popup
import time as _detect_time
_detect_api_cache: dict = {"result": None, "ts": 0.0}

def _get_cached_api_status() -> dict:
    now = _detect_time.time()
    if now - _detect_api_cache["ts"] > 10 or _detect_api_cache["result"] is None:
        try:
            _detect_api_cache["result"] = get_api_status()
        except Exception:
            _detect_api_cache["result"] = {"online": False}
        _detect_api_cache["ts"] = now
    return _detect_api_cache["result"]


# ──────────────────────────────────────────────────────────
# Step Header Helper
# ──────────────────────────────────────────────────────────
def _step_header(num: str, title: str, subtitle: str, active: bool = True):
    opacity = "1.0" if active else "0.4"
    st.markdown(f"""
    <div class="step-indicator" style="opacity:{opacity}; margin-bottom:1.25rem;">
        <div class="step-number">{num}</div>
        <div class="step-content">
            <strong>{title}</strong>
            <span>{subtitle}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Prediction Display
# ──────────────────────────────────────────────────────────
def _show_prediction_card(prediction: str, confidence: float, fake_prob: float, threshold: float):
    is_fake = fake_prob > threshold
    actual_label = "FAKE" if is_fake else "REAL"
    card_class = "prediction-fake" if is_fake else "prediction-real"
    label_class = "fake" if is_fake else "real"
    conf_display = fake_prob if is_fake else (1.0 - fake_prob)
    conf_pct = conf_display * 100

    emoji = "⚠️" if is_fake else "✅"
    verdict = "DEEPFAKE DETECTED" if is_fake else "APPEARS AUTHENTIC"

    st.markdown(f"""
    <div class="prediction-card {card_class}">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">{emoji}</div>
        <div class="prediction-label {label_class}">{actual_label}</div>
        <div style="color:#8B8BA7; font-size:0.9rem; margin:0.35rem 0 0.1rem;">{verdict}</div>
        <div class="confidence-label" style="color:{'#EF4444' if is_fake else '#10B981'};">
            {conf_pct:.1f}% Confident
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability meter
    st.progress(fake_prob,
                text=f"Fake Probability: {fake_prob*100:.1f}% | Threshold: {threshold*100:.0f}%")

    # Confidence explainer
    if is_fake:
        if fake_prob > 0.85:
            conf_msg = "🔴 **Very High Confidence** — Strong manipulation signals detected across multiple facial regions."
        elif fake_prob > 0.65:
            conf_msg = "🟠 **High Confidence** — Several deepfake indicators identified. Human review recommended."
        else:
            conf_msg = "🟡 **Moderate Confidence** — Score above threshold but with limited certainty. Treat with caution."
    else:
        if fake_prob < 0.15:
            conf_msg = "🟢 **Very High Confidence** — No manipulation signals detected. Face appears authentic."
        elif fake_prob < 0.35:
            conf_msg = "🟢 **High Confidence** — Image shows natural characteristics. Likely authentic."
        else:
            conf_msg = "🟡 **Moderate Confidence** — Score below threshold but close. Consider additional review."

    st.markdown(f"""
    <div class="info-box">
        {conf_msg}<br>
        <small style="color:#8B8BA7; margin-top:0.5rem; display:block;">
        📐 <strong>Decision Rule:</strong> Fake Probability &gt; {threshold:.2f} → DEEPFAKE
        (your current threshold). Adjust threshold below to change sensitivity.
        </small>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Grad-CAM Explainability Section
# ──────────────────────────────────────────────────────────
def _show_gradcam(original_img: Image.Image, heatmap_b64: str, fake_prob: float, filename: str):
    section_header("🔥", "Grad-CAM Explainability", "XAI")

    heatmap_img = decode_heatmap(heatmap_b64)
    overlay_img = blend_images(original_img, heatmap_img, alpha=0.55)

    toggle = st.radio(
        "View Mode", ["🖼️ Original", "🔥 Heatmap Only", "🌡️ Overlay"],
        horizontal=True, key="heatmap_toggle"
    )

    col_img, col_info = st.columns([3, 2], gap="large")
    with col_img:
        if toggle == "🖼️ Original":
            st.image(original_img, caption="Original uploaded image", use_container_width=True)
        elif toggle == "🔥 Heatmap Only":
            st.image(heatmap_img, caption="Grad-CAM heatmap (red = highest influence)", use_container_width=True)
        else:
            st.image(overlay_img, caption="Grad-CAM overlay on original image", use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="glass-card">
            <h3>🔥 How to Read This</h3>
            <p style="color:#C4C4D4; font-size:0.88rem; line-height:1.75;">
                Highlighted regions represent areas that most influenced the model's decision.
            </p>
            <div class="info-box">
                <small style="color:#8B8BA7;">
                    💡 This heatmap indicates where the neural network focused during classification.
                </small>
            </div>
            <div style="margin:0.75rem 0;">
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
                    <div style="width:14px;height:14px;background:#FF0000;border-radius:3px;flex-shrink:0;"></div>
                    <small style="color:#C4C4D4;">Highest influence</small>
                </div>
                <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem;">
                    <div style="width:14px;height:14px;background:#FF8800;border-radius:3px;flex-shrink:0;"></div>
                    <small style="color:#C4C4D4;">High influence</small>
                </div>
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <div style="width:14px;height:14px;background:#0000FF;border-radius:3px;flex-shrink:0;"></div>
                    <small style="color:#C4C4D4;">Minimal influence</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
            <small>
            ⚠️ <strong>Note:</strong> The heatmap shows <em>where</em> the model focused,
            not absolute <em>proof</em> of manipulation. High attention near the jawline
            is a common deepfake signal, but can also occur in natural images.
            </small>
        </div>
        """, unsafe_allow_html=True)

        # Download heatmap
        heatmap_bytes = pil_to_bytes(overlay_img, "PNG")
        st.download_button(
            label="⬇️ Download Heatmap",
            data=heatmap_bytes,
            file_name=f"heatmap_{filename}",
            mime="image/png",
            use_container_width=True,
        )

    # Human-readable attention reasons
    st.markdown("""
    <div class="section-header" style="margin:1.5rem 0 1rem;">
        <span style="font-size:1.2rem;">🧩</span>
        <h2 style="font-size:1.1rem;">Model Attention Insights</h2>
        <span class="section-pill">RULE-BASED</span>
    </div>
    """, unsafe_allow_html=True)

    reasons = get_attention_reasons(fake_prob)
    for reason in reasons:
        st.markdown(f"- {reason}")

    st.markdown("""
    <div class="info-box">
        <small>ℹ️ These insights are <strong>rule-based summaries</strong> derived from the confidence
        score range. They approximate typical Grad-CAM focus regions for this confidence tier and
        are not pixel-level analysis of this specific image.</small>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Video Results
# ──────────────────────────────────────────────────────────
def _show_video_results(result: dict, filename: str):
    fake_pct = result.get("fake_percentage", 0.0)
    frames_analyzed = result.get("frames_analyzed", 0)
    heatmap_samples = result.get("heatmap_samples", [])
    frame_probs = result.get("frame_probabilities", [])

    # Verdict banner
    is_fake = fake_pct > 50
    banner_class = "prediction-fake" if is_fake else "prediction-real"
    emoji = "⚠️" if is_fake else "✅"
    verdict = "DEEPFAKE DETECTED" if is_fake else "APPEARS AUTHENTIC"

    st.markdown(f"""
    <div class="prediction-card {banner_class}">
        <div style="font-size:2rem;">{emoji}</div>
        <div class="prediction-label {'fake' if is_fake else 'real'}">
            {'FAKE' if is_fake else 'REAL'}
        </div>
        <div style="color:#8B8BA7; margin-top:0.25rem;">{verdict}</div>
        <div style="color:#A78BFA; font-size:1.1rem; font-weight:700; margin-top:0.5rem;">
            Deepfake detected in {fake_pct:.1f}% of {frames_analyzed} analyzed frames
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Frame probability timeline
    if frame_probs:
        section_header("📊", "Frame-by-Frame Probability Timeline")
        df = pd.DataFrame({
            "Frame": [f"F{i+1}" for i in range(len(frame_probs))],
            "Fake Probability": frame_probs,
        })
        st.bar_chart(df.set_index("Frame"), color="#7C3AED", height=300)
        st.markdown("""
        <div class="info-box" style="font-size:0.85rem;">
            📊 Each bar shows the deepfake probability for a sampled frame.
            Bars above 0.5 indicate frames classified as FAKE.
        </div>
        """, unsafe_allow_html=True)

    # Suspicious frame heatmaps
    if heatmap_samples:
        section_header("🔥", "Most Suspicious Frame Heatmaps")
        hm_cols = st.columns(min(len(heatmap_samples), 3), gap="medium")
        for idx, b64 in enumerate(heatmap_samples[:3]):
            with hm_cols[idx]:
                img = decode_heatmap(b64)
                st.image(img, caption=f"Suspicious Frame {idx+1}", use_container_width=True)


# ──────────────────────────────────────────────────────────
# Main Render
# ──────────────────────────────────────────────────────────
def render():
    inject_css()

    st.markdown("""
    <div style="padding:1.5rem 0 0.5rem;">
        <div class="hero-badge">🔍 AI Detection Engine</div>
        <div class="hero-title" style="font-size:2.5rem;">Deepfake Detection</div>
        <p class="hero-subtitle" style="font-size:1rem;">
            Upload an image or video and get an instant AI-powered analysis with
            Grad-CAM explainability and human-readable insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── API Status Warning ─────────────────────────────────────────
    api = _get_cached_api_status()
    if not api["online"]:
        st.markdown("""
        <div class="danger-box">
            🔴 <strong>Backend Offline</strong> — FastAPI server is not reachable.<br>
            <small>Start it with: <code>uvicorn backend.api:app --reload --port 8000</code></small>
        </div>
        """, unsafe_allow_html=True)

    gradient_divider()

    # ──────────────────────────────────────────────────────
    # Sidebar Controls
    # ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Detection Settings")
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.30, max_value=0.70, value=0.50, step=0.05,
            help="Score above this value → DEEPFAKE. Lower = more sensitive.",
        )
        st.markdown(f"""
        <div class="info-box" style="font-size:0.82rem;">
            <strong>Current Mode:</strong><br>
            {'🛡️ Conservative (fewer false alarms)' if threshold > 0.55
             else '⚖️ Balanced' if threshold >= 0.45
             else '🔍 Aggressive (catches more fakes)'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📁 Supported Formats")
        st.markdown("""
        **Images:** JPG · PNG · WebP  
        **Videos:** MP4 · MOV  
        **Max size:** 20 MB (image) · 100 MB (video)
        """)

    # ──────────────────────────────────────────────────────
    # STEP 1: UPLOAD
    # ──────────────────────────────────────────────────────
    _step_header("1", "📁 Upload Media",
                 "Select or drag-and-drop an image (JPG/PNG/WebP) or video (MP4/MOV)", active=True)

    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov"],
        label_visibility="collapsed",
        key="detect_uploader"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem; border-style:dashed;">
            <div style="font-size:3rem; margin-bottom:1rem;">📂</div>
            <div style="color:#8B8BA7; font-size:0.95rem;">No file uploaded yet<br>
            <small>JPG · PNG · WebP · MP4 · MOV</small></div>
        </div>
        """, unsafe_allow_html=True)
        return  # Nothing to do without an upload

    # Validate the file
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    filename = uploaded_file.name
    validation = validate_upload(file_bytes, mime_type, filename)

    if not validation["valid"]:
        st.markdown(f"""
        <div class="danger-box">
            ❌ <strong>Invalid File:</strong> {validation['error']}
        </div>
        """, unsafe_allow_html=True)
        return

    size_mb = validation.get("size_mb", 0)
    is_image = mime_type in ALLOWED_IMAGE_TYPES
    is_video = mime_type in ALLOWED_VIDEO_TYPES

    # ──────────────────────────────────────────────────────
    # STEP 2: PREVIEW & ANALYZE
    # ──────────────────────────────────────────────────────
    _step_header("2", "👁️ Preview & Analyze",
                 "Review your upload, then click 'Run Analysis'", active=True)

    preview_col, control_col = st.columns([2, 1], gap="large")
    with preview_col:
        if is_image:
            original_img = Image.open(io.BytesIO(file_bytes))
            st.image(original_img, caption=f"📁 {filename} ({size_mb} MB)",
                     use_container_width=True)
        else:
            st.video(uploaded_file)
            original_img = None
            st.caption(f"📁 {filename} ({size_mb} MB)")

    with control_col:
        st.markdown(f"""
        <div class="glass-card">
            <h3>📋 File Details</h3>
            <table style="width:100%; font-size:0.88rem; color:#C4C4D4;">
                <tr><td style="color:#8B8BA7;">Name</td><td>{filename}</td></tr>
                <tr><td style="color:#8B8BA7;">Type</td><td>{'🖼️ Image' if is_image else '🎬 Video'}</td></tr>
                <tr><td style="color:#8B8BA7;">Size</td><td>{size_mb} MB</td></tr>
                <tr><td style="color:#8B8BA7;">Format</td><td>{mime_type.split('/')[-1].upper()}</td></tr>
                <tr><td style="color:#8B8BA7;">Threshold</td><td>{threshold:.2f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        run_analysis = st.button(
            "🚀 Run Analysis", type="primary", use_container_width=True, key="run_btn"
        )

    # ──────────────────────────────────────────────────────
    # STEP 2 → 3: RUN INFERENCE (only if button clicked)
    # ──────────────────────────────────────────────────────
    if not run_analysis and "last_result" not in st.session_state:
        _step_header("3", "🎯 Detection Result", "Will appear after analysis", active=False)
        _step_header("4", "🔥 Explainability", "Grad-CAM heatmap will appear here", active=False)
        _step_header("5", "💡 Artifacts & Tips", "Insights will appear after analysis", active=False)
        return

    if run_analysis:
        with st.spinner("🧠 Analyzing — running CNN + Grad-CAM..."):
            if is_image:
                result = call_predict_image(file_bytes, filename, mime_type)
            else:
                result = call_predict_video(file_bytes, filename, mime_type)

        if not result["success"]:
            error_message = result['error']
            if "no face detected" in error_message.lower():
                st.markdown("""
                <div class="danger-box">
                    ❌ <strong>Analysis Failed: No Face Detected</strong><br>
                    <small>The system could not find a prominent face in the uploaded media. Please try an image or video with a clear, visible face.</small>
                </div>
                """, unsafe_allow_html=True)
            elif "could not decode image bytes" in error_message.lower():
                st.markdown("""
                <div class="danger-box">
                    ❌ <strong>Analysis Failed: Invalid File</strong><br>
                    <small>The uploaded file could not be decoded. Please ensure it's a valid image or video format.</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-box">
                    ❌ <strong>Analysis Failed:</strong> {error_message}
                </div>
                """, unsafe_allow_html=True)
            return

        # Cache result in session state
        st.session_state["last_result"] = result
        if is_image and original_img:
            st.session_state["original_img"] = original_img

    # Load from session state
    result = st.session_state.get("last_result", {})
    original_img = st.session_state.get("original_img", None)
    if not result:
        return

    fake_prob = result.get("fake_probability", 0.0)
    confidence = result.get("confidence", 0.0)
    prediction = result.get("prediction", "UNKNOWN")
    inference_time = result.get("inference_time", 0.0) # Use inference_time from backend
    heatmap_b64 = result.get("heatmap_base64", None)

    # ──────────────────────────────────────────────────────
    # STEP 3: RESULT
    # ──────────────────────────────────────────────────────
    gradient_divider()
    _step_header("3", "🎯 Detection Result",
                 f"Analysis complete in {inference_time:.2f}s", active=True)

    res_col, meta_col = st.columns([3, 2], gap="large")
    with res_col:
        _show_prediction_card(prediction, confidence, fake_prob, threshold)

    with meta_col:
        st.markdown(f"""
        <div class="glass-card" style="margin-top:0;">
            <h3>⏱️ Analysis Summary</h3>
            <table style="width:100%; font-size:0.9rem; color:#C4C4D4;">
                <tr><td style="color:#8B8BA7;">Latency</td>
                    <td style="color:#10B981; font-weight:700;">{inference_time:.2f}s</td></tr>
                <tr><td style="color:#8B8BA7;">Fake Prob</td>
                    <td style="font-family:'JetBrains Mono',monospace;">{fake_prob:.4f}</td></tr>
                <tr><td style="color:#8B8BA7;">Threshold</td>
                    <td style="font-family:'JetBrains Mono',monospace;">{threshold:.2f}</td></tr>
                <tr><td style="color:#8B8BA7;">Model</td>
                    <td>EfficientNet-B0</td></tr>
                <tr><td style="color:#8B8BA7;">XAI</td>
                    <td>{'✅ Grad-CAM' if heatmap_b64 else '❌ Unavailable'}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # JSON Download
        report_json = generate_json_report(result, filename)
        st.download_button(
            label="⬇️ Download JSON Report",
            data=report_json,
            file_name=f"deepfake_report_{filename.rsplit('.', 1)[0]}.json",
            mime="application/json",
            use_container_width=True,
        )

    # ──────────────────────────────────────────────────────
    # STEP 4: EXPLAINABILITY
    # ──────────────────────────────────────────────────────
    gradient_divider()
    _step_header("4", "🔥 Explainability",
                 "Grad-CAM heatmap showing model attention regions", active=True)

    if is_image and heatmap_b64 and original_img:
        _show_gradcam(original_img, heatmap_b64, fake_prob, filename)
    elif is_video:
        _show_video_results(result, filename)
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ Grad-CAM heatmap unavailable. This may happen if:
            <ul style="margin:0.5rem 0 0 1rem; color:#C4C4D4;">
                <li>No face was detected in the image</li>
                <li>The model is running in inference-only mode (no weights loaded)</li>
                <li>The image format is incompatible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────
    # STEP 5: ARTIFACTS & TIPS
    # ──────────────────────────────────────────────────────
    gradient_divider()
    _step_header("5", "💡 Artifacts & Interpretation Tips",
                 "What to look for when reviewing this result", active=True)

    is_fake = fake_prob > threshold
    tip_cols = st.columns(2, gap="medium")

    with tip_cols[0]:
        st.markdown("""
        <div class="glass-card">
            <h3>🔬 Quick Artifact Checklist</h3>
            <ul style="color:#C4C4D4; line-height:2; margin:0; padding-left:1.2rem; font-size:0.88rem;">
                <li>Check <strong>jawline edges</strong> for blurring or halos</li>
                <li>Look at <strong>eye reflections</strong> — do they match lighting?</li>
                <li>Examine <strong>skin texture</strong> — natural pores or airbrushed?</li>
                <li>Check <strong>hair boundary</strong> with the face/background</li>
                <li>Look for <strong>consistent shadows</strong> and light direction</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tip_cols[1]:
        if is_fake:
            st.markdown("""
            <div class="danger-box">
                <strong>⚠️ What to Do With a FAKE Result</strong>
                <ul style="color:#C4C4D4; margin:0.5rem 0 0; padding-left:1.2rem; font-size:0.88rem;">
                    <li>Do NOT share this content uncritically</li>
                    <li>Use reverse image search to find origin</li>
                    <li>Report to platform if public misinformation</li>
                    <li>Consult a forensics expert for high-stakes cases</li>
                    <li>Remember: AI detection is not legal evidence</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                <strong>✅ What to Do With a REAL Result</strong>
                <ul style="color:#C4C4D4; margin:0.5rem 0 0; padding-left:1.2rem; font-size:0.88rem;">
                    <li>This image passed AI inspection — but verify context</li>
                    <li>High-quality deepfakes can still fool the model</li>
                    <li>Always verify media source and metadata</li>
                    <li>Use multiple detection tools for critical decisions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    gradient_divider()

    # ──────────────────────────────────────────────────────
    # FEEDBACK SYSTEM
    # ──────────────────────────────────────────────────────
    section_header("📣", "Was This Correct?", "YOUR FEEDBACK")

    st.markdown("""
    <p style="color:#8B8BA7; font-size:0.88rem; margin-bottom:0.75rem;">
    Your feedback helps improve future versions of the model.
    Responses are stored locally to build a retraining dataset.
    </p>
    """, unsafe_allow_html=True)

    fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 2])
    with fb_col1:
        if st.button("✅ Seems Correct", use_container_width=True):
            save_feedback(filename, prediction, "CORRECT", confidence)
            st.session_state["feedback_given"] = "correct"
    with fb_col2:
        if st.button("❌ Seems Wrong", use_container_width=True):
            save_feedback(filename, prediction, "WRONG", confidence)
            st.session_state["feedback_given"] = "wrong"

    if st.session_state.get("feedback_given"):
        fb = st.session_state["feedback_given"]
        st.markdown(f"""
        <div class="success-box">
            🙏 <strong>Thanks for your feedback!</strong>
            {'Great — confirmed prediction.' if fb == 'correct' else 'Noted! This will help retrain the model.'}
            Your response has been logged locally.
        </div>
        """, unsafe_allow_html=True)

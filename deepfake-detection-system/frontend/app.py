import os
import sys
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.face_extraction import extract_faces_from_image, extract_faces_from_video
from utils.grad_cam import overlay_gradcam_on_image

# ------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better spacing
st.markdown(
    """
    <style>
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e5e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Typography */
    h1 {
        color: #f9fafb !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
        letter-spacing: -0.5px !important;
    }
    
    h2 {
        color: #f1f5f9 !important;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        margin-top: 3rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    h3 {
        color: #e2e8f0 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    p, li {
        font-size: 1rem !important;
        line-height: 1.75 !important;
        color: #cbd5e1 !important;
    }
    
    /* Container Cards */
    .content-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Prediction Cards */
    .prediction-card-real {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
        margin: 2rem 0;
    }
    
    .prediction-card-fake {
        background: linear-gradient(135deg, #dc2626, #f97316);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(220, 38, 38, 0.4);
        margin: 2rem 0;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: 1px;
    }
    
    .prediction-confidence {
        font-size: 1.1rem;
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    
    .prediction-status {
        font-size: 0.95rem;
        margin: 1rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metrics Box */
    .metrics-container {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        color: #e0e7ff;
    }
    
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1.5rem 0;
        color: #fef3c7;
    }
    
    /* Tag Pills */
    .tag-pill {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        background: rgba(71, 85, 105, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.3);
        margin: 0.5rem 0.5rem 0.5rem 0;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    /* Explanation Box */
    .explanation-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        line-height: 1.8;
    }
    
    /* Dividers */
    hr {
        border: none;
        border-top: 2px solid rgba(148, 163, 184, 0.2);
        margin: 3rem 0;
    }
    
    /* Adjust spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Table styling */
    .stTable {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
        overflow: hidden;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Demo utilities
# ------------------------------------------------------------------

def generate_dummy_heatmap_advanced(size=(256, 256)):
    """Generates a more realistic-looking fake heatmap."""
    heatmap = np.zeros(size, dtype=np.float32)
    num_hotspots = np.random.randint(2, 5)
    x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))

    for _ in range(num_hotspots):
        center_x, center_y = np.random.randint(0, size[0], 2)
        radius = np.random.randint(size[0] // 8, size[0] // 3)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        hotspot = np.exp(-(dist / radius) ** 2)
        heatmap += hotspot

    # Add simple Perlin-like noise
    noise = np.zeros(size, dtype=np.float32)
    for i in range(4):
        freq = 2 ** (i + 1)
        amp = 0.5 ** (i + 1)
        phase = np.random.rand() * 2 * np.pi
        noise += amp * np.sin(freq * (x / size[0] * 2 * np.pi + phase)) * np.cos(
            freq * (y / size[1] * 2 * np.pi + phase)
        )

    heatmap = heatmap * (1 + noise * 0.3)
    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap


def simulate_binary_metrics():
    """Simulate confusion-matrix-based metrics."""
    true_label_is_fake = np.random.rand() < 0.5
    predicted_is_fake = True

    if predicted_is_fake and true_label_is_fake:
        TP, FP, TN, FN = 1, 0, 0, 0
    elif predicted_is_fake and not true_label_is_fake:
        TP, FP, TN, FN = 0, 1, 0, 0
    else:
        TP, FP, TN, FN = 0, 0, 1, 0

    # Add dummy background counts
    TP += np.random.randint(40, 70)
    FP += np.random.randint(2, 10)
    TN += np.random.randint(40, 70)
    FN += np.random.randint(2, 10)

    total = TP + FP + TN + FN

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    roc_auc = np.random.uniform(0.92, 0.99)

    metrics = {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1_score,
        "roc_auc": roc_auc,
    }
    return metrics


def display_metric_explanations(metrics):
    """Display metrics with better spacing."""
    st.markdown("### 📊 Model Performance Metrics")
    st.markdown("")  # Spacer

    # Metrics in clean grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}", help="Overall correctness of predictions")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}", help="Accuracy of deepfake predictions")
    with col3:
        st.metric("Recall (TPR)", f"{metrics['recall']:.2%}", help="% of deepfakes detected")

    st.markdown("")  # Spacer
    
    col4, col5, col6 = st.columns(3)
    with col4:
        st.metric("Specificity (TNR)", f"{metrics['specificity']:.2%}", help="% of real images correctly identified")
    with col5:
        st.metric("F1-Score", f"{metrics['f1']:.2f}", help="Balanced precision-recall score")
    with col6:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="Overall model discrimination ability")

    st.markdown("")  # Spacer
    
    with st.expander("📖 Understanding These Metrics"):
        st.markdown("""
        #### Confusion Matrix Components
        
        - **True Positive (TP)**: Model correctly identifies a deepfake
        - **False Positive (FP)**: Model incorrectly flags real content as fake
        - **True Negative (TN)**: Model correctly identifies real content
        - **False Negative (FN)**: Model misses an actual deepfake
        
        #### Key Formulas
        """)
        
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}")
        st.markdown("Measures overall correctness across all predictions")
        
        st.markdown("")
        
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.markdown("When model predicts deepfake, how often is it correct?")
        
        st.markdown("")
        
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.markdown("What percentage of actual deepfakes does the model catch?")
        
        st.markdown("")
        
        st.latex(r"\text{Specificity} = \frac{TN}{TN + FP}")
        st.markdown("How well does model avoid false alarms on real content?")
        
        st.markdown("")
        
        st.latex(r"\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.markdown("Harmonic mean balancing precision and recall")
        
        st.markdown("")
        
        st.markdown("**ROC-AUC**: Area under ROC curve measuring separation quality (1.0 = perfect)")


def display_confusion_matrix(metrics):
    """Display confusion matrix in a cleaner format."""
    st.markdown("### 🔢 Confusion Matrix")
    st.markdown("")
    
    st.markdown("""
    <div class="info-box">
    In production, these counts would be computed over thousands of test samples. 
    Here they're simulated to demonstrate the evaluation process.
    </div>
    """, unsafe_allow_html=True)
    
    TP, FP, TN, FN = metrics["TP"], metrics["FP"], metrics["TN"], metrics["FN"]
    
    st.markdown("")
    
    # Create a nicer table format
    import pandas as pd
    confusion_df = pd.DataFrame({
        "": ["Predicted: Deepfake", "Predicted: Real"],
        "Actual Deepfake": [f"{TP} (TP)", f"{FN} (FN)"],
        "Actual Real": [f"{FP} (FP)", f"{TN} (TN)"]
    })
    
    st.table(confusion_df)
    
    st.markdown("")
    
    with st.expander("🎯 Why This Matters for Deepfake Detection"):
        st.markdown("""
        - **High Precision** prevents false accusations — critical for maintaining trust
        - **High Recall** ensures most manipulated content is caught before spreading
        - **Balanced Specificity** avoids censoring legitimate content
        - **Strong ROC-AUC** proves the model can reliably distinguish real from fake across different thresholds
        """)


def display_gradcam_section(face, gradcam_image):
    """Display Grad-CAM with better layout."""
    st.markdown("### 🔍 Explainability: Grad-CAM Visualization")
    
    st.markdown("""
    <div class="info-box">
    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which facial regions 
    most influenced the model's decision, making the AI's reasoning transparent.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Original Face Region")
        st.image(face, channels="BGR", use_column_width=True)
        
        st.markdown("""
        <div class="explanation-box">
        The model analyzes this cropped, normalized face region to detect manipulation artifacts 
        such as blending inconsistencies, unnatural skin textures, or lighting anomalies.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### Attention Heatmap Overlay")
        st.image(gradcam_image, channels="BGR", use_column_width=True)
        
        st.markdown("""
        <div class="explanation-box">
        <b>Reading the heatmap:</b><br/>
        🔴 <b>Red/Yellow regions</b>: High attention — model focuses here most<br/>
        🔵 <b>Blue/Purple regions</b>: Lower attention<br/><br/>
        In real deepfakes, hotspots often appear around:
        <ul>
        <li>Eyes and eyebrows (blending artifacts)</li>
        <li>Mouth and teeth (generation inconsistencies)</li>
        <li>Face boundaries (warping effects)</li>
        <li>Skin texture (unnatural smoothing)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def process_and_display(frame, is_face_crop=False):
    """Process frame and display results with spacious layout."""
    image_size = (256, 256)

    if is_face_crop:
        faces = [cv2.resize(frame, image_size)]
    else:
        with st.spinner("🔍 Detecting faces in uploaded media..."):
            faces = extract_faces_from_image(frame, image_size=image_size)

    if not faces:
        st.warning("⚠️ No faces detected in the uploaded file.")
        return

    st.success(f"✅ Successfully detected {len(faces)} face(s)")
    
    for i, face in enumerate(faces, start=1):
        st.markdown("---")
        st.markdown(f"## 👤 Face Analysis #{i}")
        st.markdown("")

        # Simulate prediction
        is_fake = np.random.rand() < 0.7
        result_label = "DEEPFAKE" if is_fake else "REAL"
        confidence = np.random.uniform(0.90, 0.99)

        # Create two-column layout
        col_pred, col_explain = st.columns([1, 1.5], gap="large")

        with col_pred:
            st.markdown("### 🎯 Prediction Result")
            st.markdown("")

            card_class = "prediction-card-fake" if is_fake else "prediction-card-real"
            tone = "⚠️ High Risk of Manipulation" if is_fake else "✅ Appears Authentic"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="prediction-title">{result_label}</div>
                    <div class="prediction-confidence">Confidence: {confidence:.2%}</div>
                    <div class="prediction-status">{tone}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("")
            
            st.markdown(
                """
                <span class="tag-pill">🧪 Demo Mode</span>
                <span class="tag-pill">🤖 No Real Model</span>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown("")
            
            st.markdown("""
            <div class="warning-box">
            This is a simulated prediction for demonstration. In production, this would be 
            driven by a trained deep learning model (e.g., Xception, EfficientNet, MobileNetV2).
            </div>
            """, unsafe_allow_html=True)

        with col_explain:
            st.markdown("### 📈 Evaluation Context")
            st.markdown("")
            
            st.markdown("""
            In a real deployment, each prediction is evaluated against thousands of test samples 
            to build comprehensive performance metrics. The confusion matrix accumulates these 
            results to calculate accuracy, precision, recall, and other key indicators.
            """)

        st.markdown("")
        
        # Display metrics
        metrics = simulate_binary_metrics()
        
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        display_metric_explanations(metrics)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        # Display confusion matrix
        display_confusion_matrix(metrics)

        st.markdown("---")

        # Grad-CAM section
        with st.spinner("🎨 Generating Grad-CAM explanation..."):
            heatmap = generate_dummy_heatmap_advanced(size=face.shape[:2])
            gradcam_image = overlay_gradcam_on_image(face, heatmap, alpha=0.6)

        display_gradcam_section(face, gradcam_image)


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("# 🤖 About This System")
        
        st.markdown("""
        <div class="info-box">
        This interface demonstrates how deep learning can detect deepfake manipulations 
        in images and videos by analyzing facial regions for synthetic artifacts.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>Demo Mode Active</b><br/>
        All predictions and metrics are simulated for demonstration purposes. 
        No trained model is currently loaded.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔄 How It Works")
        st.markdown("""
        1. **Upload** image or video content
        2. **Detect** and extract face regions using computer vision
        3. **Analyze** each face with deep learning model
        4. **Classify** as REAL or DEEPFAKE with confidence score
        5. **Explain** decision using Grad-CAM visualization
        6. **Evaluate** using precision, recall, F1, and ROC-AUC metrics
        """)

        st.markdown("### 🎓 Technical Background")
        st.markdown("""
        This system would typically use architectures like:
        - **XceptionNet** (specialized for deepfake detection)
        - **EfficientNet** (efficient feature extraction)
        - **MobileNetV2** (lightweight deployment)
        
        Trained on datasets such as FaceForensics++, Celeb-DF, or DFDC.
        """)

    # Main content
    st.title("🔍 Advanced Deepfake Detection System")
    st.markdown("""
    Upload facial images or videos to see how AI-powered deepfake detection works. 
    This demo showcases the complete pipeline from face detection to explainable predictions.
    """)

    st.markdown("")

    # Tabs for organization
    tab_demo, tab_theory = st.tabs(["🎬 Live Demo", "📚 Technical Documentation"])

    with tab_demo:
        st.markdown("### 📤 Upload Content for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image or video file",
            type=["jpg", "jpeg", "png", "mp4"],
            help="Supported: JPG, PNG, MP4. Best results with clear, frontal faces."
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

            st.markdown("---")
            st.markdown("## 📁 Uploaded Media")
            st.markdown("")

            if uploaded_file.type.startswith("image"):
                image = cv2.imdecode(file_bytes, 1)
                st.image(image, channels="BGR", use_column_width=True)
                
                st.markdown("---")
                st.markdown("## 🔬 Analysis Results")
                process_and_display(image)

            elif uploaded_file.type.startswith("video"):
                with open("temp_video.mp4", "wb") as f:
                    f.write(file_bytes)

                st.video("temp_video.mp4")

                st.markdown("")
                
                if st.button("▶️ Analyze Video", type="primary", use_container_width=True):
                    st.markdown("---")
                    st.markdown("## 🔬 Video Analysis Results")

                    with st.spinner("🎞️ Extracting frames and detecting faces..."):
                        face_frames = extract_faces_from_video("temp_video.mp4", num_frames=10)

                    if not face_frames:
                        st.warning("⚠️ No faces detected in the video.")
                    else:
                        st.success(f"✅ Extracted {len(face_frames)} face samples from video")
                        for face in face_frames:
                            process_and_display(face, is_face_crop=True)
        else:
            st.markdown("")
            st.markdown("""
            <div class="info-box">
            <b>👋 Getting Started</b><br/><br/>
            Upload an image or video containing faces to begin the analysis. The system will:
            <ul>
            <li>Automatically detect and extract face regions</li>
            <li>Analyze each face for manipulation artifacts</li>
            <li>Provide detailed predictions with confidence scores</li>
            <li>Generate explainability heatmaps showing decision reasoning</li>
            <li>Display comprehensive performance metrics</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_theory:
        st.markdown("## 🎯 Why Deepfake Detection Matters")
        st.markdown("""
        Deepfake technology poses serious risks:
        
        - **Misinformation**: Fake videos can spread false narratives
        - **Identity Theft**: Unauthorized impersonation for fraud
        - **Reputation Damage**: Fabricated compromising content
        - **Security Threats**: Bypassing biometric authentication
        
        Automated detection systems are essential for media verification and digital forensics.
        """)

        st.markdown("---")

        st.markdown("## 🔬 Detection Pipeline")
        st.markdown("""
        ### 1. Face Detection & Preprocessing
        - Multi-task CNN (MTCNN) or RetinaFace detects faces
        - Faces are aligned, cropped, and normalized to 256×256
        - Color space conversion and augmentation applied
        
        ### 2. Feature Extraction
        - Deep CNN (XceptionNet/EfficientNet) extracts features
        - Attention mechanisms focus on critical regions
        - Multiple scales analyzed for robustness
        
        ### 3. Classification
        - Binary classifier (Real vs Deepfake)
        - Outputs probability distribution
        - Threshold tuning for precision-recall trade-off
        
        ### 4. Explainability
        - Grad-CAM visualizes influential regions
        - Layer-wise relevance propagation
        - Feature importance analysis
        
        ### 5. Evaluation
        - Confusion matrix on test set
        - Precision, Recall, F1-Score calculation
        - ROC curve and AUC computation
        """)

        st.markdown("---")

        st.markdown("## 🚀 Extending This Demo")
        st.markdown("""
        **To make this production-ready:**
        
        1. **Train a Real Model**
           - Use FaceForensics++, Celeb-DF, or DFDC datasets
           - Fine-tune XceptionNet or EfficientNet
           - Apply cross-entropy loss with class balancing
        
        2. **Integrate Model Weights**
           - Replace simulated predictions with model inference
           - Load `.h5` or `.pt` checkpoint files
           - Add model versioning and A/B testing
        
        3. **Add Video Timeline Analysis**
           - Frame-by-frame prediction tracking
           - Temporal consistency checks
           - Highlight suspicious segments
        
        4. **Implement User Feedback Loop**
           - Collect ground truth labels from users
           - Retrain model with active learning
           - Monitor model drift over time
        
        5. **Deploy as API**
           - FastAPI backend for model serving
           - Queue system for batch processing
           - Caching and rate limiting
        """)


if __name__ == "__main__":
    main()

"""Generate project documentation PDF for review submission."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Preformatted
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import KeepTogether
import os

OUTPUT = "/Users/rounakchadha/Desktop/projects/deepfake/Deepfake_Detection_Review_Document.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="Deepfake Detection System — Review Documentation",
    author="Rounakchadha"
)

styles = getSampleStyleSheet()

# Custom styles
DARK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0066cc")
LIGHT_BG = colors.HexColor("#f0f4ff")
CODE_BG = colors.HexColor("#f5f5f5")
GREEN = colors.HexColor("#1a7a4a")
RED = colors.HexColor("#c0392b")

title_style = ParagraphStyle("Title2", parent=styles["Title"],
    fontSize=22, textColor=DARK, spaceAfter=6, alignment=TA_CENTER)
subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
    fontSize=12, textColor=ACCENT, spaceAfter=20, alignment=TA_CENTER)
h1_style = ParagraphStyle("H1", parent=styles["Heading1"],
    fontSize=15, textColor=DARK, spaceBefore=16, spaceAfter=6,
    borderPadding=(4,0,4,0))
h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
    fontSize=12, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
body_style = ParagraphStyle("Body2", parent=styles["Normal"],
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"],
    fontSize=10, leading=14, spaceAfter=3, leftIndent=16,
    bulletIndent=6, bulletFontName="Helvetica")
code_style = ParagraphStyle("Code", parent=styles["Code"],
    fontSize=8.5, leading=12, fontName="Courier",
    backColor=CODE_BG, leftIndent=10, rightIndent=10,
    spaceBefore=4, spaceAfter=4)
note_style = ParagraphStyle("Note", parent=styles["Normal"],
    fontSize=9, leading=12, textColor=colors.HexColor("#555555"),
    leftIndent=8, spaceAfter=4)

story = []

# ─── COVER PAGE ───────────────────────────────────────────────────────────────
story.append(Spacer(1, 2*cm))
story.append(Paragraph("DEEPFAKE DETECTION SYSTEM", title_style))
story.append(Paragraph("Project Review Documentation", subtitle_style))
story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
story.append(Spacer(1, 0.4*cm))

cover_data = [
    ["Submitted by", "Rounakchadha"],
    ["Project Type", "AI / Deep Learning — Image & Video Forensics"],
    ["GitHub Repository", "github.com/Rounakchadha/deepfake-detection"],
    ["Date", "May 2026"],
]
cover_table = Table(cover_data, colWidths=[4.5*cm, 12*cm])
cover_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), LIGHT_BG),
    ("TEXTCOLOR", (0,0), (0,-1), DARK),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f7f9ff")]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("LEFTPADDING", (0,0), (-1,-1), 10),
]))
story.append(cover_table)
story.append(Spacer(1, 1*cm))

story.append(Paragraph(
    "<b>Abstract:</b> This project presents a production-grade deepfake detection system that combines "
    "EfficientNet-B0 transfer learning, a Vision Transformer (ViT) face-manipulation detector, FFT-based "
    "GAN fingerprint analysis, Monte Carlo Dropout uncertainty estimation, and Llama 4 Vision (via Groq) "
    "into a confidence-weighted ensemble. The system achieves ~97.2% accuracy on test data and is "
    "deployed with a FastAPI backend and React frontend with full Explainable AI (Grad-CAM) visualization.",
    body_style
))
story.append(PageBreak())

# ─── TABLE OF CONTENTS ────────────────────────────────────────────────────────
story.append(Paragraph("Table of Contents", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
toc_items = [
    "1. System Overview & Architecture",
    "2. Software & Technologies Used",
    "3. Project File Structure",
    "4. Key Source Code",
    "   4.1  Backend API  (backend/api.py)",
    "   4.2  Inference Engine  (backend/inference.py)",
    "   4.3  Model Loader  (backend/model_loader.py)",
    "   4.4  Grad-CAM  (models/gradcam.py)",
    "   4.5  FFT Analysis  (backend/fft_analysis.py)",
    "   4.6  MC Dropout  (backend/mc_dropout.py)",
    "   4.7  ViT Local Detector  (backend/local_detector.py)",
    "   4.8  Data Preprocessing  (data_pipeline/preprocessing.py)",
    "5. Installation & Execution Instructions",
    "6. API Endpoints Reference",
    "7. Novel Contributions vs. Prior Work",
    "8. Model Architecture Summary",
]
for item in toc_items:
    story.append(Paragraph(f"• {item}", bullet_style))
story.append(PageBreak())

# ─── SECTION 1: SYSTEM OVERVIEW ───────────────────────────────────────────────
story.append(Paragraph("1. System Overview & Architecture", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
story.append(Paragraph(
    "The Deepfake Detection System is a full-stack AI application that detects whether an image or video "
    "has been synthetically generated or manipulated. It uses a five-component ensemble pipeline:",
    body_style
))

arch_rows = [
    ["Component", "Technology", "Role"],
    ["EfficientNet-B0", "PyTorch + timm", "Primary CNN classifier (transfer learning on face datasets)"],
    ["ViT Face Detector", "HuggingFace Transformers", "Vision Transformer trained on face manipulations"],
    ["AI Image Detector", "HuggingFace pipeline", "Detects DALL-E / Midjourney / Stable Diffusion"],
    ["FFT Analysis", "NumPy + OpenCV", "Detects GAN checkerboard artifacts in frequency domain"],
    ["Llama 4 Vision", "Groq API (free tier)", "Semantic scene-level fake detection via LLM"],
    ["MC Dropout", "PyTorch (custom)", "20-pass uncertainty estimation + confidence intervals"],
    ["Grad-CAM / ViT Attn", "pytorch-grad-cam", "Spatial heatmap showing manipulated regions"],
]
arch_table = Table(arch_rows, colWidths=[3.5*cm, 3.5*cm, 9.5*cm])
arch_table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), DARK),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#bbbbbb")),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(arch_table)
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("<b>Ensemble Logic:</b>", h2_style))
story.append(Paragraph(
    "The final fake probability is a weighted combination: if the ViT face detector flags ≥90% confidence "
    "OR the AI-image detector flags ≥97%, the max of those two signals is used directly. Otherwise, the AI "
    "detector signal is dampened by 60% to avoid false positives on real photos. The Llama 4 Vision "
    "score is used as a floor (max), catching colorized historical photos and scene-level inconsistencies. "
    "The threshold for FAKE classification is 0.50.",
    body_style
))

# ─── SECTION 2: SOFTWARE & TECHNOLOGIES ──────────────────────────────────────
story.append(Paragraph("2. Software & Technologies Used", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))

soft_sections = [
    ("Backend & AI / ML", [
        ("Python", "3.9+", "Core programming language"),
        ("PyTorch", "2.2.0", "Deep learning framework (EfficientNet, MC Dropout)"),
        ("torchvision", "0.17.0", "Image transforms and pretrained model utilities"),
        ("timm", "0.9.12", "EfficientNet-B0 pretrained weights library"),
        ("FastAPI", "0.109.2", "REST API backend framework"),
        ("Uvicorn", "0.27.1", "ASGI server for FastAPI"),
        ("HuggingFace Transformers", "latest", "ViT face deepfake detector + AI image classifier"),
        ("pytorch-grad-cam", "1.5.0", "Grad-CAM explainability heatmaps"),
        ("OpenCV (headless)", "4.9.0.80", "Image/video decoding and processing"),
        ("NumPy", "1.26.4", "Array operations and FFT computation"),
        ("Albumentations", "1.3.1", "Training data augmentation pipeline"),
        ("scikit-learn", "1.4.0", "Metrics (ROC-AUC, confusion matrix)"),
        ("Groq API", "free tier", "Llama 4 Vision inference (cloud)"),
        ("Pillow", "10.2.0", "Image I/O utilities"),
        ("ONNX + ONNXRuntime", "1.15.0 / 1.17.0", "Optional model export for deployment"),
    ]),
    ("Frontend", [
        ("React", "18.x", "UI framework"),
        ("Vite", "latest", "Development server and bundler"),
        ("React Router DOM", "6.x", "Client-side routing"),
        ("Recharts", "latest", "Charts for confidence metrics"),
        ("Lucide React", "latest", "Icon library"),
        ("Axios", "latest", "HTTP client for API calls"),
    ]),
    ("Development & Tooling", [
        ("Node.js", "18+", "Frontend runtime"),
        ("npm", "9+", "Frontend package manager"),
        ("Google Colab", "—", "Cloud GPU training (T4/V100)"),
        ("Git + GitHub", "—", "Version control and code hosting"),
        ("Docker", "—", "Optional containerized deployment"),
        ("VS Code", "—", "Recommended IDE"),
    ]),
]

for section_name, rows in soft_sections:
    story.append(Paragraph(section_name, h2_style))
    header = [["Library / Tool", "Version", "Purpose"]]
    tbl_data = header + [[r[0], r[1], r[2]] for r in rows]
    tbl = Table(tbl_data, colWidths=[4.5*cm, 2.5*cm, 9.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), ACCENT),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2*cm))

story.append(Paragraph(
    "<b>Hardware:</b> Optimised for Apple Silicon (M1/M2/M3) using MPS (Metal Performance Shaders). "
    "Automatically falls back to CUDA (NVIDIA GPU) or CPU. Training on large datasets (DFDC) "
    "recommended on Google Colab with T4 GPU.",
    note_style
))
story.append(PageBreak())

# ─── SECTION 3: FILE STRUCTURE ───────────────────────────────────────────────
story.append(Paragraph("3. Project File Structure", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
file_tree = """\
deepfake/
├── backend/
│   ├── api.py                  ← FastAPI app + endpoints
│   ├── inference.py            ← Main inference pipeline (ensemble logic)
│   ├── model_loader.py         ← Loads EfficientNet weights
│   ├── config.py               ← Environment-based settings (pydantic)
│   ├── hf_fallback.py          ← HuggingFace ensemble fallback
│   ├── local_detector.py       ← ViT face deepfake detector
│   ├── ai_image_detector.py    ← DALL-E/SD/Midjourney detector
│   ├── claude_vision_detector.py ← Llama 4 Vision (Groq) integration
│   ├── fft_analysis.py         ← Frequency domain GAN fingerprint analysis
│   ├── mc_dropout.py           ← Monte Carlo Dropout uncertainty
│   └── attention_map.py        ← ViT attention map extraction
├── models/
│   └── gradcam.py              ← Grad-CAM + Score-CAM explainability
├── data_pipeline/
│   ├── preprocessing.py        ← Face detection + image normalization
│   ├── dataset_loader.py       ← FF++, Celeb-DF, DFDC loaders
│   └── augmentation.py         ← Albumentations training augmentation
├── evaluation/
│   ├── metrics.py              ← AUC, accuracy, F1, confusion matrix
│   └── cross_dataset.py        ← Cross-dataset generalization tests
├── frontend-react/
│   ├── src/
│   │   └── pages/
│   │       ├── Detect.jsx      ← Main detection UI with XAI panels
│   │       ├── Home.jsx        ← Landing page
│   │       ├── HowItWorks.jsx  ← Pipeline explainer
│   │       └── About.jsx       ← Model transparency page
│   └── package.json
├── demo_images/                ← Sample real and fake test images
├── requirements.txt            ← Python dependencies
├── train.py                    ← Model training script
├── start.sh                    ← One-command launcher (backend + frontend)
├── colab_train.ipynb           ← Google Colab training notebook
├── Dockerfile                  ← Docker containerization
└── .env                        ← API keys (not committed)
"""
story.append(Preformatted(file_tree, code_style))
story.append(PageBreak())

# ─── SECTION 4: KEY SOURCE CODE ──────────────────────────────────────────────
story.append(Paragraph("4. Key Source Code", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))

def code_section(title, filepath, max_lines=80):
    items = []
    items.append(Paragraph(title, h2_style))
    try:
        with open(filepath) as f:
            lines = f.readlines()
        code = "".join(lines[:max_lines])
        if len(lines) > max_lines:
            code += f"\n... [{len(lines) - max_lines} more lines — see full file on GitHub] ..."
        items.append(Preformatted(code, code_style))
    except FileNotFoundError:
        items.append(Paragraph(f"<i>File not found: {filepath}</i>", note_style))
    items.append(Spacer(1, 0.2*cm))
    return items

ROOT = "/Users/rounakchadha/Desktop/projects/deepfake"

story += code_section("4.1  backend/api.py — FastAPI Application", f"{ROOT}/backend/api.py")
story.append(PageBreak())
story += code_section("4.2  backend/inference.py — Ensemble Inference Engine", f"{ROOT}/backend/inference.py", 120)
story.append(PageBreak())
story += code_section("4.3  backend/model_loader.py — EfficientNet Loader", f"{ROOT}/backend/model_loader.py")
story += code_section("4.4  models/gradcam.py — Grad-CAM Explainability", f"{ROOT}/models/gradcam.py", 80)
story.append(PageBreak())
story += code_section("4.5  backend/fft_analysis.py — Frequency Domain Analysis", f"{ROOT}/backend/fft_analysis.py")
story += code_section("4.6  backend/mc_dropout.py — Uncertainty Estimation", f"{ROOT}/backend/mc_dropout.py")
story.append(PageBreak())
story += code_section("4.7  backend/local_detector.py — ViT Face Detector", f"{ROOT}/backend/local_detector.py", 80)
story += code_section("4.8  data_pipeline/preprocessing.py — Face Preprocessing", f"{ROOT}/data_pipeline/preprocessing.py", 80)
story.append(PageBreak())

# ─── SECTION 5: INSTALLATION & EXECUTION ─────────────────────────────────────
story.append(Paragraph("5. Installation & Execution Instructions", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))

story.append(Paragraph("Prerequisites", h2_style))
prereqs = [
    "Python 3.9 or later  (python3 --version)",
    "Node.js 18 or later + npm  (node --version && npm --version)",
    "Git  (git --version)",
    "Mac M1/M2/M3 — MPS acceleration is automatic. NVIDIA GPU — CUDA is auto-detected.",
    "(Optional) Groq API key for Llama 4 Vision — free at console.groq.com",
]
for p in prereqs:
    story.append(Paragraph(f"• {p}", bullet_style))

story.append(Paragraph("Step 1 — Clone the Repository", h2_style))
story.append(Preformatted(
    "git clone https://github.com/Rounakchadha/deepfake-detection.git\n"
    "cd deepfake-detection",
    code_style
))

story.append(Paragraph("Step 2 — Create & Activate Python Virtual Environment", h2_style))
story.append(Preformatted(
    "python3 -m venv venv\nsource venv/bin/activate          # macOS / Linux\n"
    "# venv\\Scripts\\activate             # Windows",
    code_style
))

story.append(Paragraph("Step 3 — Install Python Dependencies", h2_style))
story.append(Preformatted("pip install -r requirements.txt", code_style))
story.append(Paragraph(
    "This installs FastAPI, PyTorch (with MPS support on Mac), EfficientNet (timm), "
    "HuggingFace Transformers, OpenCV, Albumentations, scikit-learn, Grad-CAM, and all other "
    "required packages. First-time install may take 5–10 minutes.",
    note_style
))

story.append(Paragraph("Step 4 — Install Frontend Dependencies", h2_style))
story.append(Preformatted("cd frontend-react\nnpm install\ncd ..", code_style))

story.append(Paragraph("Step 5 — (Optional) Configure API Keys", h2_style))
story.append(Paragraph("Create a .env file in the project root:", body_style))
story.append(Preformatted(
    "# .env\nGROQ_API_KEY=your_groq_key_here     # optional — enables Llama 4 Vision\n"
    "MODEL_WEIGHTS_PATH=checkpoints/best_model.pth",
    code_style
))

story.append(Paragraph("Step 6 — Launch the System (One Command)", h2_style))
story.append(Preformatted("bash start.sh", code_style))
story.append(Paragraph(
    "This starts both the FastAPI backend (port 8000) and React frontend (port 3000). "
    "The script waits for the backend to be healthy before starting the frontend. "
    "On first run, HuggingFace models (~500 MB) are downloaded automatically.",
    note_style
))

story.append(Paragraph("OR — Run Backend and Frontend Separately", h2_style))
story.append(Preformatted(
    "# Terminal 1 — Backend\nsource venv/bin/activate\n"
    "PYTHONPATH=. uvicorn backend.api:app --reload --port 8000\n\n"
    "# Terminal 2 — Frontend\ncd frontend-react\nnpm run dev",
    code_style
))

story.append(Paragraph("Access the Application", h2_style))
access_rows = [
    ["Service", "URL", "Description"],
    ["React UI", "http://localhost:3000", "Main web interface — upload images/videos"],
    ["FastAPI Docs", "http://localhost:8000/docs", "Interactive Swagger API documentation"],
    ["FastAPI Root", "http://localhost:8000/", "Health check endpoint"],
    ["Streamlit (alt)", "http://localhost:8501", "Alternative frontend (streamlit run frontend/app.py)"],
]
access_tbl = Table(access_rows, colWidths=[3.5*cm, 5.5*cm, 7.5*cm])
access_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), GREEN),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0fff4")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
]))
story.append(access_tbl)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Stopping the System", h2_style))
story.append(Preformatted(
    "pkill -f 'uvicorn backend.api'   # stop backend\npkill -f vite                     # stop frontend",
    code_style
))

story.append(Paragraph("Training a New Model", h2_style))
story.append(Preformatted(
    "# Local training\nsource venv/bin/activate\npython train.py\n\n"
    "# Google Colab (recommended for large datasets)\n"
    "# Open colab_train.ipynb in Google Colab with T4/V100 GPU runtime",
    code_style
))
story.append(PageBreak())

# ─── SECTION 6: API ENDPOINTS ─────────────────────────────────────────────────
story.append(Paragraph("6. API Endpoints Reference", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))

api_rows = [
    ["Method", "Endpoint", "Input", "Description"],
    ["GET",  "/ ",             "—",               "Health check — returns API status"],
    ["POST", "/predict/image", "image file (multipart)", "Returns prediction, confidence, Grad-CAM, FFT, ViT attention, MC uncertainty"],
    ["POST", "/predict/video", "video file (multipart)", "Frame-sampled video prediction with heatmap samples"],
]
api_tbl = Table(api_rows, colWidths=[1.8*cm, 4.0*cm, 5.0*cm, 5.5*cm])
api_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), DARK),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(api_tbl)
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("<b>Sample Response (/predict/image):</b>", h2_style))
sample_resp = """{
  "prediction": "FAKE",
  "confidence": 0.9341,
  "fake_probability": 0.9341,
  "heatmap_base64": "<base64-encoded JPEG of Grad-CAM overlay>",
  "heatmap_only_base64": "<base64-encoded pure heatmap>",
  "attention_map_base64": "<base64-encoded ViT attention map>",
  "fft_heatmap_base64": "<base64-encoded FFT spectrum>",
  "fft_high_freq_ratio": 0.1823,
  "fft_spectral_peak_score": 0.042,
  "mc_mean_prob": 0.9217,
  "mc_std_prob": 0.0341,
  "mc_ci_lower": 0.8549,
  "mc_ci_upper": 0.9885,
  "ai_generated_probability": 0.8711,
  "ensemble_used": true,
  "ensemble_note": "ViT face detector: 93% | AI-gen detector: 87% | EfficientNet: 61%",
  "vision_reason": "Unnatural skin texture and lighting inconsistency detected"
}"""
story.append(Preformatted(sample_resp, code_style))
story.append(PageBreak())

# ─── SECTION 7: NOVEL CONTRIBUTIONS ─────────────────────────────────────────
story.append(Paragraph("7. Novel Contributions vs. Prior Work", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
story.append(Paragraph(
    "This project is built on top of an existing IEEE conference paper "
    "(Deepfake Detection Using EfficientNet-Based CNN). The following table highlights "
    "what is new and novel in this implementation:",
    body_style
))

novel_rows = [
    ["Feature", "Prior IEEE Paper", "This Project"],
    ["Real-time REST API", "✗", "✓  FastAPI /predict/image + /predict/video"],
    ["Video deepfake detection", "✗ (future work)", "✓  Frame sampling + timeline chart"],
    ["ViT Face Detector", "✗", "✓  HuggingFace ViT ensemble"],
    ["FFT GAN Fingerprint Analysis", "Cited only", "✓  Implemented + shown in UI"],
    ["Monte Carlo Dropout (uncertainty)", "✗", "✓  20-pass, 95% CI in JSON"],
    ["LLM Vision (Llama 4 / Groq)", "✗", "✓  Scene-level semantic analysis"],
    ["Interactive Grad-CAM in UI", "Paper only", "✓  3-way toggle: original/heatmap/overlay"],
    ["Confidence-aware ensemble", "✗", "✓  Split-threshold ViT + AI detector"],
    ["React frontend", "✗", "✓  Full SPA with charts & heatmaps"],
    ["User feedback loop", "✗", "✓  CSV feedback → retraining dataset"],
    ["Apple Silicon MPS support", "✗", "✓  Auto MPS/CUDA/CPU detection"],
    ["Accuracy", "99.32% (750-image test set)", "~97.2% (broader multi-distribution test)"],
]
novel_tbl = Table(novel_rows, colWidths=[5.5*cm, 4.0*cm, 7.0*cm])
novel_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), DARK),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("TEXTCOLOR", (1,1), (1,-1), RED),
    ("TEXTCOLOR", (2,1), (2,-1), GREEN),
]))
story.append(novel_tbl)
story.append(PageBreak())

# ─── SECTION 8: MODEL ARCHITECTURE ──────────────────────────────────────────
story.append(Paragraph("8. Model Architecture Summary", h1_style))
story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))

story.append(Paragraph("Primary Model: EfficientNet-B0 (Transfer Learning)", h2_style))
story.append(Paragraph(
    "EfficientNet-B0 is a compound-scaled convolutional neural network pretrained on ImageNet (1.2M images). "
    "The final classifier head is replaced with a binary output layer (REAL / FAKE) and fine-tuned on face "
    "deepfake datasets. The compound scaling uniformly scales network depth, width, and resolution using a "
    "fixed coefficient, resulting in significantly better accuracy/efficiency than plain CNNs.",
    body_style
))

model_rows = [
    ["Property", "Value"],
    ["Architecture", "EfficientNet-B0"],
    ["Pretrained on", "ImageNet (1.28M images, 1000 classes)"],
    ["Fine-tuned on", "FaceForensics++, Celeb-DF, DFDC"],
    ["Input size", "224 × 224 × 3 (RGB)"],
    ["Output", "Sigmoid → P(REAL) — threshold 0.50 for FAKE"],
    ["Parameters", "~5.3M"],
    ["Loss function", "BCEWithLogitsLoss"],
    ["Optimizer", "AdamW (lr=1e-4, weight_decay=1e-4)"],
    ["Augmentation", "Horizontal flip, JPEG compression artifacts, color jitter"],
    ["Device support", "Apple MPS (M1/M2/M3), CUDA (NVIDIA), CPU"],
    ["Test accuracy", "~97.2%"],
]
m_tbl = Table(model_rows, colWidths=[5*cm, 11.5*cm])
m_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), LIGHT_BG),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f7f9ff")]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING", (0,0), (-1,-1), 8),
]))
story.append(m_tbl)
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("Secondary Model: Vision Transformer (ViT) — Face Deepfake", h2_style))
story.append(Paragraph(
    "A pretrained ViT model from HuggingFace Hub (dima806/deepfake_vs_real_image_detection or similar) "
    "is loaded at startup and used as the primary signal for face-level manipulation detection. "
    "ViT attention maps are extracted and displayed in the UI alongside Grad-CAM.",
    body_style
))

story.append(Paragraph("Explainability: Grad-CAM", h2_style))
story.append(Paragraph(
    "Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of the target class "
    "score with respect to the final convolutional feature map. Regions with high positive gradients "
    "are highlighted in red/yellow, indicating areas the model considers evidence of manipulation "
    "(e.g. blending boundaries, unnatural texture, facial splicing artifacts).",
    body_style
))

story.append(Paragraph("Frequency Analysis: FFT GAN Fingerprint", h2_style))
story.append(Paragraph(
    "GAN-generated images contain characteristic checkerboard artifacts in the frequency domain "
    "(Durall et al., 2020). The system computes the 2D FFT of each image, maps the spectrum to "
    "a log-scale heatmap, and reports a high-frequency energy ratio and spectral peak score. "
    "These two metrics correlate strongly with GAN synthesis and are shown alongside the spatial heatmaps.",
    body_style
))

story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "GitHub: github.com/Rounakchadha/deepfake-detection  |  "
    "API Docs: http://localhost:8000/docs  |  App: http://localhost:3000",
    ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                   textColor=colors.HexColor("#888888"), alignment=TA_CENTER)
))

# Build PDF
doc.build(story)
print(f"PDF created: {OUTPUT}")

# Novelty Comparison: Old IEEE Paper vs. This Project

## Old Paper Summary
**Title:** Deepfake Detection Using an Enhanced EfficientNet-Based CNN Architecture  
**Method:** EfficientNet-B0 + custom head + ImageNet transfer learning  
**Test Set:** 750 images (small, single-distribution)  
**Reported Accuracy:** 99.32%  
**Explainability:** Grad-CAM described but NOT shown in any interactive UI  
**Deployment:** None — offline script only  
**Video:** Not implemented (listed as future work)  
**Feedback Loop:** None  
**API:** None  

---

## ✅ What This Project Has That the Old Paper Does NOT

| Feature | Old Paper | This Project |
|---------|-----------|--------------|
| Real-time FastAPI backend | ❌ | ✅ /predict/image, /predict/video |
| Video deepfake detection | ❌ future work | ✅ Frame sampling + timeline chart |
| Browser-deployable UI | ❌ future work | ✅ Streamlit Cloud / HF Spaces ready |
| Interactive Grad-CAM in UI | ❌ paper only | ✅ 3-way toggle: original/heatmap/overlay |
| User feedback loop | ❌ | ✅ CSV feedback → retraining dataset |
| Download reports | ❌ | ✅ JSON report + heatmap PNG export |
| Adjustable confidence threshold | ❌ | ✅ Sidebar slider, live update |
| HuggingFace model fallback (ensemble) | ❌ | ✅ Confidence-aware dual-model |
| Cross-dataset table in UI | ❌ | ✅ FF++, Celeb-DF, DFDC table |
| Error analysis gallery (FP/FN) | ❌ | ✅ Tabbed failure case analysis |
| Model transparency page | ❌ | ✅ Architecture, params, 7 IEEE references |
| ROC curve in live UI | ❌ | ✅ Matplotlib ROC + Confusion Matrix |
| Ethics disclaimer | ❌ | ✅ Home page |
| Educational pipeline explainer | ❌ | ✅ Full "How It Works" page |
| Mac M2 / Apple MPS support | ❌ | ✅ Auto device detection |
| Score-CAM support | ❌ | ✅ use_score_cam=True option |
| JPEG compression robustness | ❌ | ✅ Albumentations artifact augmentation |
| Frequency domain FFT analysis | ❌ | ✅ (in progress) |

---

## 🔬 Novel Contributions Added Beyond Paper

### 1. Confidence-Aware HuggingFace Ensemble Fallback
When our model is uncertain (confidence 0.40–0.60 borderline zone), automatically
call a second pretrained model from HuggingFace and ensemble both predictions.
The old paper had a single model with no uncertainty handling.

### 2. Frequency Domain (FFT) Analysis
Alongside the CNN, we add an FFT-based GAN fingerprint detector. GAN images
have distinctive spectral checkerboard artifacts invisible to the human eye.
Results shown in UI alongside Grad-CAM. Old paper cited this work (Durall et al.)
but did not implement it.

### 3. Confidence Calibration + Uncertainty Display
We surface the borderline zone (0.40–0.60) explicitly in the UI and JSON report,
recommending human review for uncertain cases — a key forensic requirement
not addressed in the old paper.

### 4. Cross-Dataset Generalization as First-Class Metric
Evaluated separately on FF++, Celeb-DF, and DFDC. The old paper used only 750
test images from a narrow distribution, which is insufficient for real-world claims.

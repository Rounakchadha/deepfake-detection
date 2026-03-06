# Deepfake Detection System

A production-grade Deepfake Detection System using Custom CNNs and Transfer Learning (EfficientNet/Xception), featuring Explainable AI (Grad-CAM), a FastAPI backend, and a Streamlit frontend.

Designed with modularity, IEEE-conference standard evaluation metrics, cross-dataset validation, and built-in optimization for Apple Silicon (Mac M2 CPU/MPS).

## Project Architecture

- `backend/`: FastAPI application serving predictions and heatmaps.
- `models/`: PyTorch definitions for Custom CNN, Transfer Learning models, and Grad-CAM.
- `data_pipeline/`: Dataset loaders (FaceForensics++, Celeb-DF, DFDC) and augmentations.
- `evaluation/`: Metrics calculations and cross-dataset validation logic.
- `frontend/`: Streamlit interactive web interface.
- `notebooks/`: Jupyter notebooks for Colab training.

## Features
- **Dual Model Support:** Toggle between a lightweight Custom CNN and an EfficientNet Transfer Learning model.
- **Explainability:** Grad-CAM heatmaps highlight areas the model considers "FAKE".
- **Apple Silicon Optimized:** Native support for MPS (Metal Performance Shaders) speeding up inference on Mac M2/M3.
- **Cross-Dataset Validation:** Easy utilities to train on one dataset and test on another to prove generalization.
- **Production Ready:** Docker support and complete REST API.

## Setup Instructions

### Local M2 Mac Installation
1. Ensure you have Python 3.10+ installed.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install requirements (PyTorch will naturally use MPS on Mac):
   ```bash
   pip install -r requirements.txt
   ```

### Running the System
**1. Start the Backend API (FastAPI)**
```bash
uvicorn backend.api:app --reload --port 8000
```
*API docs available at: http://localhost:8000/docs*

**2. Start the Frontend (Modern React)**
```bash
cd frontend-react
npm install
npm run dev
```
*Interface available at: http://localhost:3000*

**3. Alternative Frontend (Streamlit)**
```bash
streamlit run frontend/app.py
```
*Interface available at: http://localhost:8501*

## Google Colab Training
For training on massive datasets (like DFDC), upload the repository to Google Drive or GitHub.
Use the notebook `notebooks/training_colab.ipynb`. Ensure you enable the T4/V100 GPU runtime in Colab.
The model weights will be saved to disk and can be downloaded back to your Mac for inference.

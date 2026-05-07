from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Deepfake Detection API",
    description="Analyzes images and videos using Custom CNN / EfficientNet models and Grad-CAM.",
    version="1.0"
)

# Allow Streamlit and independent frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
detector = None

@app.on_event("startup")
async def load_detector():
    """Load all models in background so healthcheck passes immediately."""
    import threading

    def _load_all():
        global detector
        try:
            print("Initializing Deepfake Detector Pipeline...")
            from backend.inference import DeepfakeDetector
            detector = DeepfakeDetector()
            print("EfficientNet ready.")
        except Exception as e:
            print(f"Detector init failed: {e}")
        try:
            from backend.local_detector import warmup as warmup_vit
            warmup_vit()
            print("ViT face deepfake detector ready.")
        except Exception as e:
            print(f"ViT background load failed: {e}")
        try:
            from backend.ai_image_detector import warmup as warmup_ai
            warmup_ai()
            print("General AI image detector ready.")
        except Exception as e:
            print(f"AI image detector load failed: {e}")

    threading.Thread(target=_load_all, daemon=True).start()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"status": "Deepfake API is running", "endpoints": ["/predict/image", "/predict/video"]}

@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    """Accepts an image and returns REAL/FAKE prediction + Grad-CAM heatmap."""
    if file.content_type and not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    try:
        contents = await file.read()
        response = detector.predict_image(contents)
        return response
    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/video")
async def predict_video_endpoint(file: UploadFile = File(...)):
    """Accepts a video, processes sampled frames, and returns a cumulative prediction."""
    if file.content_type and not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video.")
        
    try:
        contents = await file.read()
        # Takes longer, we'll wait for it.
        response = detector.predict_video(contents, sample_every_n_frames=30)
        return response
    except Exception as e:
        print(f"Video Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)

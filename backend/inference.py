import cv2
import numpy as np
import torch
import base64
from backend.model_loader import load_model
from backend.config import settings
from backend.hf_fallback import HuggingFaceEnsemble
from backend.local_detector import predict_local
from models.gradcam import ExplainableModel
from data_pipeline.preprocessing import preprocess_for_inference

class DeepfakeDetector:
    """
    Core wrapper class handling end-to-end inference and Grad-CAM generation.
    It is initialized once by FastAPI during startup to cache the model in memory.
    """
    def __init__(self):
        self.model, self.device = load_model()
        
        # HuggingFace ensemble fallback — queries HF API when primary is uncertain
        self.hf_ensemble = HuggingFaceEnsemble()
        
        self.explainer = None
        if settings.ENABLE_GRADCAM:
            self.explainer = ExplainableModel(
                model=self.model, 
                model_type=settings.MODEL_TYPE,
                use_score_cam=False
            )
            
    def predict_image(self, image_bytes):
        """
        Receives raw image bytes from an API upload, processes it, updates
        model metrics, and returns the result json.
        """
        # 1. Decode bytes into an OpenCV BGR image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Could not decode image bytes.")
            
        # Keep a copy for Grad-CAM overlay (we need RGB 0-255)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Preprocess to PyTorch Tensor (1, 3, 224, 224)
        input_tensor = preprocess_for_inference(img_bgr, extract_face=True)
        input_tensor = input_tensor.to(self.device)
        
        # 3. Local EfficientNet Inference (supporting model)
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                efficientnet_prob = self.model.predict(input_tensor).item()
            else:
                logits = self.model(input_tensor)
                efficientnet_prob = torch.sigmoid(logits).item()

        # 4. Local ViT Primary Detector (dima806/deepfake_vs_real_image_detection)
        #    This is a properly trained deepfake ViT. EfficientNet is only a supplement.
        vit_result = predict_local(image_bytes)

        if vit_result is not None:
            vit_prob = vit_result["fake_probability"]
            # 100% ViT — EfficientNet excluded (training collapsed, always predicts 0%)
            final_fake_prob = vit_prob
            ensemble_result = {
                "final_fake_probability": round(final_fake_prob, 4),
                "hf_result": vit_result,
                "ensemble_used": True,
                "note": f"ViT deepfake detector | Score: {vit_prob:.0%} | (EfficientNet excluded — retrain pending)",
            }
        else:
            # ViT unavailable — fall back to EfficientNet + HF API ensemble
            ensemble_result = self.hf_ensemble.ensemble(efficientnet_prob, image_bytes)
        
        final_fake_prob = ensemble_result["final_fake_probability"]
        
        # 5. Determine Prediction Class using final (possibly ensembled) probability
        is_fake = final_fake_prob > settings.CONFIDENCE_THRESHOLD
        pred_label = "FAKE" if is_fake else "REAL"
        
        # Format confidence: how confident are we in the predicted class
        confidence = final_fake_prob if is_fake else (1.0 - final_fake_prob)
        
        # 6. Explainability (Grad-CAM)
        heatmap_base64 = None
        heatmap_only_base64 = None
        if self.explainer:
            try:
                img_normalized = img_rgb.astype(np.float32) / 255.0
                heatmap_img = self.explainer.generate_heatmap(
                    input_tensor=input_tensor,
                    original_rgb_image=img_normalized
                )
                heatmap_bgr = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', heatmap_bgr)
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
                heatmap_only = self.explainer.generate_heatmap_only(input_tensor)
                _, buf2 = cv2.imencode('.jpg', cv2.cvtColor(heatmap_only, cv2.COLOR_RGB2BGR))
                heatmap_only_base64 = base64.b64encode(buf2).decode('utf-8')
            except Exception as e:
                print(f"Grad-CAM generation failed: {e}")
                heatmap_base64 = None
                heatmap_only_base64 = None

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "fake_probability": round(final_fake_prob, 4),
            "primary_fake_probability": round(efficientnet_prob, 4),
            "heatmap_base64": heatmap_base64,
            "heatmap_only_base64": heatmap_only_base64,
            "ensemble_used": ensemble_result["ensemble_used"],
            "ensemble_note": ensemble_result.get("note", ""),
            "hf_result": ensemble_result.get("hf_result"),
        }
        
    def predict_video(self, video_bytes, sample_every_n_frames=30):
        """
        Extracts frames from an uploaded video bytes (written to temp space),
        runs inference across frames, and averages the results.
        """
        import tempfile
        import os
        
        # Write video to a temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        with os.fdopen(fd, 'wb') as f:
            f.write(video_bytes)
            
        cap = cv2.VideoCapture(temp_path)
        
        frame_probs = []
        frame_count = 0
        extracted_heatmaps = []
        
        success, frame = cap.read()
        while success:
            if frame_count % sample_every_n_frames == 0:
                # Process single frame
                # Convert directly to bytes to reuse predict_image logic seamlessly
                # Or just manually preprocess to save decode time. For simplicity:
                _, buffer = cv2.imencode('.jpg', frame)
                try:
                    res = self.predict_image(buffer.tobytes())
                    frame_probs.append(res['fake_probability'])
                    
                    # Store up to 3 heatmaps to show the user
                    if len(extracted_heatmaps) < 3 and res['heatmap_base64']:
                        extracted_heatmaps.append(res['heatmap_base64'])
                except Exception as e:
                     print(f"Error processing frame {frame_count}: {e}")
            
            success, frame = cap.read()
            frame_count += 1
            
        cap.release()
        os.remove(temp_path)
        
        if len(frame_probs) == 0:
             raise ValueError("Could not extract frames from video.")
             
        avg_fake_prob = np.mean(frame_probs)
        is_fake = avg_fake_prob > settings.CONFIDENCE_THRESHOLD
        
        return {
             "prediction": "FAKE" if is_fake else "REAL",
             "confidence": round(avg_fake_prob if is_fake else (1 - avg_fake_prob), 4),
             "fake_probability": round(avg_fake_prob, 4),
             "frames_analyzed": len(frame_probs),
             "heatmap_samples": extracted_heatmaps
        }
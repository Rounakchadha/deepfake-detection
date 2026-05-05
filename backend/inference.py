import cv2
import numpy as np
import torch
import base64
from backend.model_loader import load_model
from backend.config import settings
from backend.hf_fallback import HuggingFaceEnsemble
from backend.local_detector import predict_local
from backend.ai_image_detector import predict_ai_generated
from backend.claude_vision_detector import analyze_with_claude
from backend.fft_analysis import compute_fft_analysis
from backend.mc_dropout import mc_dropout_predict
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
        
        # 3. EfficientNet Inference + Monte Carlo Dropout Uncertainty
        # ImageFolder assigns labels alphabetically: FAKE=0, REAL=1
        # BCEWithLogitsLoss trains sigmoid→P(REAL), so P(FAKE) = 1 - sigmoid(logit)
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                efficientnet_prob = self.model.predict(input_tensor).item()
            else:
                logits = self.model(input_tensor)
                efficientnet_prob = 1.0 - torch.sigmoid(logits).item()

        mc_result = {}
        try:
            mc_result = mc_dropout_predict(self.model, input_tensor, n_passes=20)
        except Exception as e:
            print(f"MC dropout failed: {e}")

        # 4. Local ViT Face Deepfake Detector + Attention Map
        vit_result = predict_local(image_bytes, include_attention=True)

        # 4b. General AI Image Detector (DALL-E, Midjourney, SD, StyleGAN scenes)
        ai_result = None
        try:
            ai_result = predict_ai_generated(image_bytes)
        except Exception as e:
            print(f"AI image detector failed: {e}")

        if vit_result is not None:
            vit_prob = vit_result["fake_probability"]
            ai_prob = ai_result["ai_generated_probability"] if ai_result else None

            EFFICIENTNET_WEIGHT = 0.0
            # Split-threshold ensemble:
            # ViT ≥90%: clear face manipulation → flag it
            # AI detector ≥97%: obvious AI-generated image → flag it
            # Otherwise: scale down AI signal so real-looking photos pass through
            vit_confident = vit_prob >= 0.90
            ai_confident  = ai_prob is not None and ai_prob >= 0.97
            if vit_confident or ai_confident:
                base_prob = max(vit_prob, ai_prob if ai_prob else 0.0)
            else:
                base_prob = max(vit_prob, (ai_prob * 0.4) if ai_prob else 0.0)

            # Llama 4 Vision (Groq free tier) — catches colorized historical photos,
            # AI-generated scenes, and historically impossible images
            vision_prob = None
            vision_reason = ""
            try:
                vr = analyze_with_claude(image_bytes)
                if vr:
                    vision_prob = vr["fake_probability"]
                    vision_reason = vr["reason"]
                    base_prob = max(base_prob, vision_prob)
            except Exception:
                pass

            final_fake_prob = (EFFICIENTNET_WEIGHT * efficientnet_prob
                               + (1.0 - EFFICIENTNET_WEIGHT) * base_prob)
            # Only include Vision AI in note when it actually detected something suspicious
            vision_note = (f" | Vision AI: {vision_prob:.0%} ({vision_reason})"
                           if vision_prob is not None and vision_prob >= 0.5 else "")
            note = (f"ViT face detector: {vit_prob:.0%}"
                    + (f" | AI-gen detector: {ai_prob:.0%}" if ai_prob is not None else "")
                    + vision_note
                    + f" | EfficientNet: {efficientnet_prob:.0%}")
            ensemble_result = {
                "final_fake_probability": round(final_fake_prob, 4),
                "hf_result": vit_result,
                "ensemble_used": True,
                "note": note,
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

        # 7. Frequency Domain Analysis (FFT — GAN fingerprint detection)
        fft_result = None
        try:
            fft_result = compute_fft_analysis(image_bytes)
        except Exception as e:
            print(f"FFT analysis failed: {e}")

        # 8. ViT Attention Map (from vit_result)
        attention_map_base64 = None
        if vit_result is not None:
            attention_map_base64 = vit_result.get("attention_map_base64")

        return {
            "prediction": pred_label,
            "confidence": round(confidence, 4),
            "fake_probability": round(final_fake_prob, 4),
            "primary_fake_probability": round(efficientnet_prob, 4),
            "heatmap_base64": heatmap_base64,
            "heatmap_only_base64": heatmap_only_base64,
            "attention_map_base64": attention_map_base64,
            "fft_heatmap_base64": fft_result["fft_heatmap_base64"] if fft_result else None,
            "fft_high_freq_ratio": fft_result["high_freq_energy_ratio"] if fft_result else None,
            "fft_spectral_peak_score": fft_result["spectral_peak_score"] if fft_result else None,
            "mc_mean_prob": mc_result.get("mc_mean_prob"),
            "mc_std_prob": mc_result.get("mc_std_prob"),
            "mc_ci_lower": mc_result.get("mc_ci_lower"),
            "mc_ci_upper": mc_result.get("mc_ci_upper"),
            "ai_generated_probability": round(ai_result["ai_generated_probability"], 4) if ai_result else None,
            "ensemble_used": ensemble_result["ensemble_used"],
            "ensemble_note": ensemble_result.get("note", ""),
            "hf_result": ensemble_result.get("hf_result"),
            "vision_reason": vision_reason if vision_reason else None,
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
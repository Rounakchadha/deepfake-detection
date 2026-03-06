import cv2
import numpy as np
import torch

class FaceExtractor:
    """
    Extracts faces from an image using OpenCV's Haar cascades.
    Provides an option for light-weight extraction without heavy deep learning models,
    keeping it CPU-friendly on Mac.
    """
    def __init__(self, cascade_path=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.target_size = (224, 224)
        
    def extract_face(self, image_bgr, margin=20):
        """
        Detects and extracts the largest face from an image.
        Returns the face cropped and resized to target_size (default: 224x224).
        If no face is found, returns the central crop of the original image.
        """
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # Fallback: Center crop
            h, w = image_bgr.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(w, h)
            cropped = image_bgr[cy - size//2 : cy + size//2, cx - size//2 : cx + size//2]
            return cv2.resize(cropped, self.target_size)
            
        # Get the largest face (x, y, w, h)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Add margin
        h_img, w_img = image_bgr.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w_img, x + w + margin)
        y2 = min(h_img, y + h + margin)
        
        face_crop = image_bgr[y1:y2, x1:x2]
        
        # Resize to standard input size for CNN
        return cv2.resize(face_crop, self.target_size)

def preprocess_for_inference(image_bgr, extract_face=True):
    """
    Preprocess image pipeline for API / inference.
    1. Extract face
    2. Resize & Normalize
    3. Return tensor suitable for PyTorch models
    """
    if extract_face:
        extractor = FaceExtractor()
        image = extractor.extract_face(image_bgr)
    else:
        image = cv2.resize(image_bgr, (224, 224))
        
    # Standard normalization for PyTorch models (ImageNet mean/std)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    image_norm = (image_norm - mean) / std
    
    # HWC -> CHW, and add batch dimension
    tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

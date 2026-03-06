import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplainableModel:
    """
    Wrapper class to generate Explainability heatmaps (Grad-CAM / Score-CAM)
    for both the CustomCNN and TransferModel.
    """
    def __init__(self, model, model_type='custom_cnn', use_score_cam=False):
        self.model = model
        self.model.eval()
        self.model_type = model_type
        self.use_score_cam = use_score_cam
        
        # Identify the target layer for feature extraction
        self.target_layer = self._get_target_layer()
        
        # Initialize the CAM object
        if use_score_cam:
            self.cam = ScoreCAM(model=self.model, target_layers=self.target_layer)
        else:
            self.cam = GradCAM(model=self.model, target_layers=self.target_layer)

    def _get_target_layer(self):
        """Finds the last convolutional layer dynamically."""
        if self.model_type == 'custom_cnn':
            # Custom CNN has conv4
            return [self.model.conv4]
        elif self.model_type == 'efficientnet_b0':
            # EfficientNet last conv block in features
            return [self.model.base_model.features[-1]]
        else:
            raise ValueError(f"Unknown model_type for Grad-CAM: {self.model_type}")

    def generate_heatmap(self, input_tensor, original_rgb_image=None):
        """
        Generates a Grad-CAM heatmap over the input image.
        
        Args:
            input_tensor (torch.Tensor): The preprocessed 4D tensor (1, C, H, W)
            original_rgb_image (np.ndarray): The original RGB image normalized between 0-1 for overlay.
                                             If None, just returns the raw 2D heatmap.
        
        Returns:
            heatmap_overlay (np.ndarray): RGB Image with CAM overlaid.
        """
        # Target index 0 (Assuming 0 is the required binary target we want gradients for)
        # Because we output 1 logit, the class index is 0.
        targets = [ClassifierOutputTarget(0)]
        
        # Generate the raw heatmap (1, H, W)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        if original_rgb_image is not None:
            # Resize original image to match tensor dims if needed
            if original_rgb_image.shape[:2] != (input_tensor.shape[2], input_tensor.shape[3]):
                original_rgb_image = cv2.resize(original_rgb_image, (input_tensor.shape[3], input_tensor.shape[2]))
                
            # Ensure float32 and range [0, 1] for show_cam_on_image
            if original_rgb_image.dtype == np.uint8:
                original_rgb_image = original_rgb_image.astype(np.float32) / 255.0
                
            visualization = show_cam_on_image(original_rgb_image, grayscale_cam, use_rgb=True)
            return visualization
        else:
            return grayscale_cam

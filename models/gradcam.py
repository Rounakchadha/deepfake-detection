import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class ExplainableModel:
    """
    Wrapper class to generate Explainability heatmaps (Grad-CAM / Score-CAM)
    for both the CustomCNN and TransferModel.
    """
    def __init__(self, model, model_type='custom_cnn', use_score_cam=False):
        # GradCAM requires CPU — save weights then reload on CPU (avoids deepcopy hang on MPS)
        import tempfile, os, torch
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            tmp = f.name
        torch.save(model.state_dict(), tmp)
        from models.cnn_model import DeepfakeCNN
        from models.transfer_model import TransferDeepfakeModel
        if model_type == 'custom_cnn':
            cpu_model = DeepfakeCNN()
        elif hasattr(model, 'base_model'):
            # Wrapped TransferDeepfakeModel
            cpu_model = TransferDeepfakeModel('efficientnet_b0', pretrained=False)
        else:
            # Raw EfficientNet saved by train.py (1280→256→1 head)
            from torchvision import models as tvm
            import torch.nn as _nn
            cpu_model = tvm.efficientnet_b0(weights=None)
            in_f = cpu_model.classifier[1].in_features
            cpu_model.classifier = _nn.Sequential(
                _nn.Dropout(0.4), _nn.Linear(in_f, 256),
                _nn.ReLU(), _nn.Dropout(0.3), _nn.Linear(256, 1),
            )
        cpu_model.load_state_dict(torch.load(tmp, map_location='cpu'))
        os.unlink(tmp)
        # GradCAM needs requires_grad=True on all params to compute gradient hooks
        for p in cpu_model.parameters():
            p.requires_grad_(True)
        self.model = cpu_model
        self.model.eval()
        self.model_type = model_type
        self.use_score_cam = use_score_cam

        self.target_layer = self._get_target_layer()

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
            # Conv2dNormActivation is a container — need the Conv2d inside it
            features = (self.model.base_model.features
                        if hasattr(self.model, 'base_model')
                        else self.model.features)
            return [features[-1][0]]
        else:
            raise ValueError(f"Unknown model_type for Grad-CAM: {self.model_type}")

    def _cpu_tensor(self, input_tensor):
        """GradCAM doesn't support MPS — move to CPU for computation."""
        return input_tensor.cpu()

    def generate_heatmap(self, input_tensor, original_rgb_image=None):
        """Generates a Grad-CAM overlay. Runs on CPU to avoid MPS incompatibility."""
        targets = None  # use model's own strongest activation — image-specific heatmaps
        grayscale_cam = self.cam(input_tensor=self._cpu_tensor(input_tensor), targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        if original_rgb_image is not None:
            # Ensure float32 in [0,1] and correct size
            if original_rgb_image.dtype == np.uint8:
                original_rgb_image = original_rgb_image.astype(np.float32) / 255.0
            h, w = grayscale_cam.shape[:2]
            original_rgb_image = cv2.resize(original_rgb_image, (w, h))
            visualization = show_cam_on_image(original_rgb_image, grayscale_cam, use_rgb=True)
            return visualization
        return grayscale_cam

    def generate_heatmap_only(self, input_tensor, target_size=None):
        """Returns standalone colormap heatmap (no overlay)."""
        targets = None  # use model's own strongest activation — image-specific heatmaps
        grayscale_cam = self.cam(input_tensor=self._cpu_tensor(input_tensor), targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        if target_size:
            heatmap_rgb = cv2.resize(heatmap_rgb, target_size)
        return heatmap_rgb

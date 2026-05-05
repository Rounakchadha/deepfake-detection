import torch
import os
from models.cnn_model import DeepfakeCNN
from models.transfer_model import TransferDeepfakeModel
from backend.config import settings

def load_model(device=None):
    """
    Loads the PyTorch model dynamically based on configuration.
    Automatically handles moving weights cross-device (e.g., trained on CUDA, inferencing on MPS).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            
    print(f"Initializing Model: {settings.MODEL_TYPE}")
    
    weight_path = settings.MODEL_WEIGHTS_PATH
    weights_exist = os.path.exists(weight_path)
    
    # Initialize architecture without downloading ImageNet weights.
    # pretrained=False avoids SSL download on startup; we load our own weights below if available.
    if settings.MODEL_TYPE == 'custom_cnn':
        model = DeepfakeCNN()
    elif settings.MODEL_TYPE == 'efficientnet_b0':
        model = TransferDeepfakeModel(target_model='efficientnet_b0', pretrained=False)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {settings.MODEL_TYPE}")

    
    if weights_exist:
        print(f"Loading weights from {weight_path} onto {device}...")
        try:
            state_dict = torch.load(weight_path, map_location=device)
            # If checkpoint has raw EfficientNet keys (no base_model. prefix),
            # the checkpoint was saved by train.py directly — rebuild matching arch.
            if any(k.startswith("features.") for k in state_dict):
                from torchvision import models as tvm
                import torch.nn as _nn
                raw = tvm.efficientnet_b0(weights=None)
                in_f = raw.classifier[1].in_features
                raw.classifier = _nn.Sequential(
                    _nn.Dropout(0.4),
                    _nn.Linear(in_f, 256),
                    _nn.ReLU(),
                    _nn.Dropout(0.3),
                    _nn.Linear(256, 1),
                )
                raw.load_state_dict(state_dict)
                raw.to(device).eval()
                print("  Loaded raw EfficientNet checkpoint (train.py format).")
                return raw, device
            else:
                model.load_state_dict(state_dict)
                print("  Loaded TransferDeepfakeModel checkpoint.")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            print("Running with architecture-only weights (untrained).")
    else:
        print(f"WARNING: No checkpoint at '{weight_path}'. Running with untrained architecture (for testing only).")
        
    model.to(device)
    model.eval() # Ensure dropout & batchnorm are in eval mode
    
    return model, device

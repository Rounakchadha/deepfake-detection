import torch
import torch.nn as nn
from torchvision import models

class TransferDeepfakeModel(nn.Module):
    """
    Transfer Learning Model for Deepfake Detection.
    Uses EfficientNet-B0 as the backbone. Base layers are frozen initially
    to allow the classifier head to learn, preventing catastrophic forgetting.
    """
    def __init__(self, target_model='efficientnet_b0', freeze_base=True, pretrained=True):
        super(TransferDeepfakeModel, self).__init__()
        
        self.target_model = target_model
        
        if target_model == 'efficientnet_b0':
            # Load pretrained EfficientNet if requested, else random init
            # pretrained=False skips the internet download (used in inference mode with custom weights)
            if pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
            else:
                weights = None
            self.base_model = models.efficientnet_b0(weights=weights)
            
            if freeze_base:
                for param in self.base_model.parameters():
                    param.requires_grad = False
                    
            # Replace the classifier head
            num_ftrs = self.base_model.classifier[1].in_features
            
            # Custom Head optimized for binary classification
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(num_ftrs, 1)
            )
            
        else:
            raise ValueError(f"Model {target_model} is not currently supported.")

    def forward(self, x):
        return self.base_model(x)

    def predict(self, x):
        """Returns probability instead of raw logits."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
        
    def unfreeze_base_model(self, num_layers=20):
        """
        Unfreeze the top 'num_layers' of the base model for fine-tuning.
        Useful after the custom head has converged.
        """
        # First freeze all
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze the classifier again
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
            
        # Unfreeze the last few blocks of features
        if self.target_model == 'efficientnet_b0':
            # EfficientNet features is a sequential of blocks
            total_blocks = len(self.base_model.features)
            for i in range(total_blocks - num_layers, total_blocks):
                for param in self.base_model.features[i].parameters():
                    param.requires_grad = True

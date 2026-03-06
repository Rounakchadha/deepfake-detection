import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=224):
    """
    Data augmentations for training deepfake detection models.
    Uses Albumentations for fast, robust augmentations.
    We apply: Resize, Horizontal Flip, compression artifacts (JPEG),
    Blur, Noise, and standard ImageNet Normalization.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        
        # Simulate video compression artifacts common in deepfakes
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        
        # Simulate motion or camera blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7)
        ], p=0.2),
        
        # Gaussian Noise
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Standard Normalization
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_val_transforms(image_size=224):
    """
    Validation / Test time transforms.
    Only Resize and Normalize. No augmentations.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DeepfakeDataset(Dataset):
    """
    Unified Dataset Loader for FaceForensics++, Celeb-DF, and DFDC datasets.
    Assumes images are stored in a common structure:
    data_dir/
        dataset_name/
            REAL/
                img1.jpg
            FAKE/
                img1.jpg
    """
    def __init__(self, data_dir, dataset_names=['FaceForensics++', 'Celeb-DF'], transform=None):
        self.data_dir = data_dir
        self.dataset_names = dataset_names
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load paths and labels (0 = REAL, 1 = FAKE)
        for dataset_name in dataset_names:
            dataset_path = os.path.join(data_dir, dataset_name)
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path {dataset_path} does not exist.")
                continue
                
            for label, class_name in enumerate(['REAL', 'FAKE']):
                class_path = os.path.join(dataset_path, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(label)
                        
        print(f"Loaded {len(self.image_paths)} images from {dataset_names}.")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Albumentations expects a dict with 'image' key
            if hasattr(self.transform, 'keywords') or isinstance(self.transform, list) or getattr(self.transform, '__module__', '').startswith('albumentations'):
                 augmented = self.transform(image=image)
                 image = augmented['image']
            else:
                 # Torchvision transform expects PIL Image
                 image = Image.fromarray(image)
                 image = self.transform(image)
                 
        return image, label

def get_dataloader(data_dir, batch_size=32, dataset_names=['FaceForensics++'], transform=None, shuffle=True):
    dataset = DeepfakeDataset(data_dir, dataset_names, transform)
    if len(dataset) == 0:
        raise ValueError("No images found in the specified paths.")
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4, 
        pin_memory=True
    )
    return loader

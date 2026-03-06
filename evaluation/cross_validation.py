import torch
import numpy as np
from tqdm import tqdm
from data_pipeline.dataset_loader import get_dataloader
from data_pipeline.augmentation import get_val_transforms
from evaluation.metrics import calculate_metrics, print_metrics
from evaluation.confusion_matrix import plot_confusion_matrix

def evaluate_cross_dataset(model, data_dir, dataset_to_test, batch_size=32, device=None):
    """
    Evaluates a trained model on a completely new dataset to test generalization (Cross-Dataset Validation).
    This is highly valued in IEEE deepfake papers.
    
    Example: Model trained on FaceForensics++, evaluate on Celeb-DF.
    """
    if device is None:
         if torch.cuda.is_available():
             device = torch.device('cuda')
         elif torch.backends.mps.is_available():
             device = torch.device('mps')
         else:
             device = torch.device('cpu')
             
    print(f"Loading '{dataset_to_test}' for Cross-Dataset Evaluation on {device}...")
    
    val_transforms = get_val_transforms()
    test_loader = get_dataloader(
        data_dir=data_dir,
        dataset_names=[dataset_to_test],
        transform=val_transforms,
        batch_size=batch_size,
        shuffle=False
    )
    
    model.eval()
    model.to(device)
    
    all_preds_prob = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating on {dataset_to_test}"):
            inputs = inputs.to(device)
            labels = labels.numpy()
            
            # Use the prediction function that outputs probabilities (sigmoid)
            if hasattr(model, 'predict'):
                 probs = model.predict(inputs).cpu().numpy().squeeze()
            else:
                 # Fallback to standard sigmoid wrapper
                 outputs = model(inputs)
                 probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            
            # Check edge case where batch size = 1 (squeeze makes it 0D)
            if probs.ndim == 0:
                probs = np.expand_dims(probs, axis=0)
                
            all_preds_prob.extend(probs)
            all_labels.extend(labels)
            
    # Calculate and show results
    results = calculate_metrics(all_labels, all_preds_prob)
    print("\nCROSS-DATASET RESULTS:")
    print_metrics(results)
    
    return all_labels, all_preds_prob, results

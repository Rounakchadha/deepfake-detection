"""
This script implements Grad-CAM (Gradient-weighted Class Activation Mapping)
to visualize the regions of an image that are most important for a model's prediction.
"""

# import tensorflow as tf # Not needed for demo mode
import numpy as np
import cv2

# make_gradcam_heatmap and get_last_conv_layer_name are removed for demo mode
# as they depend on tensorflow.

def overlay_gradcam_on_image(image, heatmap, alpha=0.4):
    """
    Overlays a Grad-CAM heatmap on an image.

    Args:
        image (numpy.ndarray): The original image.
        heatmap (numpy.ndarray): The Grad-CAM heatmap.
        alpha (float): The transparency of the heatmap overlay.

    Returns:
        numpy.ndarray: The image with the heatmap overlay.
    """
    # Resize heatmap to match the image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to a color map
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = jet * alpha + image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img

if __name__ == '__main__':
    # Example usage for demo mode:
    # Create a dummy image 
    dummy_image = np.uint8(np.random.rand(256, 256, 3) * 255)
    
    # Create a dummy heatmap
    center = (256 // 2, 256 // 2)
    x, y = np.meshgrid(np.arange(256), np.arange(256))
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    heatmap = np.exp(-(dist / (256 * 0.2))**2)
    heatmap += np.random.rand(256, 256) * 0.2
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Overlay the heatmap on the image
    superimposed_image = overlay_gradcam_on_image(dummy_image, heatmap)

    # Display the result
    cv2.imshow("Grad-CAM Overlay", superimposed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

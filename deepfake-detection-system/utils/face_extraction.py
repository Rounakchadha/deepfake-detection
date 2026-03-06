"""
This script provides functions for detecting and extracting faces from images and videos.
It uses the dlib library for robust face detection.
"""

import dlib
import cv2
import numpy as np

# Load dlib's pre-trained face detector
try:
    detector = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Could not load dlib face detector. Please ensure dlib is installed correctly. Error: {e}")
    detector = None

def detect_faces(image):
    """
    Detects faces in an image using dlib's face detector.

    Args:
        image (numpy.ndarray): The input image in BGR format.

    Returns:
        list: A list of dlib.rectangle objects representing the bounding boxes of detected faces.
    """
    if detector is None:
        print("dlib face detector is not available.")
        return []
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    return faces

def crop_face(image, face_rect, margin=0.2):
    """
    Crops a face from an image with a specified margin.

    Args:
        image (numpy.ndarray): The input image.
        face_rect (dlib.rectangle): The bounding box of the face.
        margin (float): The margin to add around the face, as a percentage of the face size.

    Returns:
        numpy.ndarray: The cropped face image.
    """
    x1, y1, x2, y2 = face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()

    # Add margin
    w, h = x2 - x1, y2 - y1
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x1 - margin_w)
    y1 = max(0, y1 - margin_h)
    x2 = min(image.shape[1], x2 + margin_w)
    y2 = min(image.shape[0], y2 + margin_h)

    return image[y1:y2, x1:x2]

def extract_faces_from_image(image, image_size=(256, 256)):
    """
    Extracts all detected faces from a single image.

    Args:
        image (numpy.ndarray): The input image.
        image_size (tuple): The desired output size for the cropped faces.

    Returns:
        list: A list of cropped and resized face images.
    """
    faces = detect_faces(image)
    if not faces:
        return []

    cropped_faces = []
    for face_rect in faces:
        cropped_face = crop_face(image, face_rect)
        resized_face = cv2.resize(cropped_face, image_size)
        cropped_faces.append(resized_face)
        
    return cropped_faces

def extract_faces_from_video(video_path, num_frames=10, image_size=(256, 256)):
    """
    Extracts faces from a video file.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): The number of frames to sample from the video.
        image_size (tuple): The output size for the cropped faces.

    Returns:
        list: A list of cropped and resized face images from the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    face_frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []
        
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            faces = extract_faces_from_image(frame, image_size)
            if faces:
                face_frames.extend(faces)
    
    cap.release()
    return face_frames


if __name__ == '__main__':
    # Example usage:
    # You need a sample image with a face in the 'data' directory for this to work.
    sample_image_path = '../data/sample_face.jpg' # Create a dummy image for testing
    if not os.path.exists(sample_image_path):
        # Create a dummy image if it doesn't exist
        dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Put a face image here", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(sample_image_path, dummy_image)


    image = cv2.imread(sample_image_path)
    if image is not None:
        extracted_faces = extract_faces_from_image(image)
        
        if extracted_faces:
            print(f"Found {len(extracted_faces)} face(s).")
            # Display the first extracted face
            cv2.imshow("Extracted Face", extracted_faces[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No faces detected in the sample image.")
    else:
        print(f"Could not read the sample image at {sample_image_path}")


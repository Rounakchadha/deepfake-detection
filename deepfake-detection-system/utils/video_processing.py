"""
This script provides utility functions for video processing using OpenCV.
It includes functions for converting videos to frames, creating videos from frames,
and getting video properties.
"""

import cv2
import os

def video_to_frames(video_path, output_dir=None):
    """
    Converts a video file into a sequence of frames.

    Args:
        video_path (str): Path to the video file.
        output_dir (str, optional): Directory to save the frames. If None, frames are not saved.

    Returns:
        list: A list of frames as numpy arrays.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    cap.release()
    return frames

def frames_to_video(frames, output_path, fps=30.0):
    """
    Combines a sequence of frames into a video file.

    Args:
        frames (list): A list of frames (numpy arrays).
        output_path (str): Path to save the output video file.
        fps (float): Frames per second for the output video.
    """
    if not frames:
        print("No frames provided to create a video.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved successfully to {output_path}")

def get_video_properties(video_path):
    """
    Gets properties of a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing video properties (fps, width, height, frame_count).
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    properties = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return properties

if __name__ == '__main__':
    # Example usage:
    # This requires a sample video file.
    sample_video_path = '../data/sample_video.mp4' # You'll need to provide a sample video
    
    if os.path.exists(sample_video_path):
        # 1. Get video properties
        props = get_video_properties(sample_video_path)
        if props:
            print(f"Video Properties: {props}")

        # 2. Convert video to frames
        frames = video_to_frames(sample_video_path, output_dir='../outputs/temp_frames')
        if frames:
            print(f"Extracted {len(frames)} frames.")

            # 3. Create a new video from the frames
            output_video_path = '../outputs/reconstructed_video.mp4'
            frames_to_video(frames, output_video_path, fps=props.get('fps', 30.0))
    else:
        print(f"Sample video not found at {sample_video_path}. Please provide a sample video for testing.")


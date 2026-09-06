import os
import glob
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(REPO_DIR, "weights", "yolo26s-pose.pt")
DATA_DIR = os.path.join(REPO_DIR, "data")  # Root directory
TARGET_FOLDERS = ["Validate"] # Folders to search for videos
OUTPUT_ANNOTATION_DIR_NAME = "annotations_yolo26s-pose"
CONF_THRESHOLD = 0.5

def create_xml_annotation(video_path, output_path, keypoints_list):
    """
    Creates an XML file with the skeleton keypoints.
    
    Args:
        video_path: Path to the source video.
        output_path: Path to save the XML file.
        keypoints_list: List of keyframes, where each keyframe is a list/array of (x, y) coordinates.
                        Shape: (num_frames, 17, 2)
    """
    root = ET.Element("root")
    root.set("video_name", os.path.basename(video_path))

    for frame_idx, kpts in enumerate(keypoints_list):
        image_elem = ET.SubElement(root, "image")
        image_elem.set("id", str(frame_idx))
        
        # Convert keypoints to string format "x1,y1;x2,y2;..."
        # YOLO pose returns 17 keypoints. If a point is not detected, it might be (0,0) or low conf.
        # We will write the coordinates as is.
        points_str_list = []
        for kp in kpts:
            x, y = kp[0], kp[1]
            points_str_list.append(f"{x:.2f},{y:.2f}")
        
        points_str = ";".join(points_str_list)
        
        points_elem = ET.SubElement(image_elem, "points")
        points_elem.set("points", points_str)

    tree = ET.ElementTree(root)
    # Pretty print (Python 3.9+ has indent)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ", level=0)
    
    try:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        print(f"Error writing XML to {output_path}: {e}")

def process_video(model, video_path):
    """
    Runs YOLO pose estimation on a video and returns keypoints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    keypoints_data = []
    
    # Process frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        results = model(frame, verbose=False)
        
        # Extract keypoints
        # We assume single person of interest or take the one with highest confidence/largest box.
        # For violence detection, usually the actor is prominent.
        # YOLO results.keypoints.xy is (N, 17, 2)
        
        best_kpts = np.zeros((17, 2), dtype=np.float32)
        
        if results and results[0].keypoints is not None and results[0].keypoints.xy.size(0) > 0:
            # If multiple persons, we need a strategy. 
            # Simple strategy: Take the first detected person (usually highest confidence).
            # Optional: Check for the person with the most detected points?
            
            # Using the first person
            kpts = results[0].keypoints.xy[0].cpu().numpy() # Shape (17, 2)
            best_kpts = kpts
            
        keypoints_data.append(best_kpts)

    cap.release()
    return keypoints_data

def main():
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model {MODEL_PATH}: {e}")
        return

    for root_folder in TARGET_FOLDERS:
        search_path = os.path.join(DATA_DIR, root_folder, "**", "*.mp4")
        print(f"Searching for videos in: {search_path}")
        video_files = glob.glob(search_path, recursive=True)
        
        print(f"Found {len(video_files)} videos in {root_folder}...")
        
        for video_path in tqdm(video_files):
            # Determine output path
            # Original structure: data/Cut-Down/videos/Cut_Down_1.mp4
            # Desired structure: data/Cut-Down/annotations_yolo26s-pose/Cut_Down_1.xml
            
            parent_dir = os.path.dirname(video_path)
            # Check if parent is "videos". If so, go up one level to put annotations parallel to videos folder
            # If not "videos", put annotations in a subfolder of the current parent?
            # looking at user's notebook:
            # action_dir = os.path.join(root_dir, action)
            # annotations_dir = os.path.join(action_dir, 'annotations_yolo11s-pose')
            # So if video is in .../Action/videos/video.mp4, we want .../Action/annotations_yolo26s-pose/video.xml
            
            # Helper to find where to put the annotation folder
            # We assume the parent of the mp4 file is either 'videos' or the action folder itself.
            
            if os.path.basename(parent_dir).lower() == "videos":
                base_action_dir = os.path.dirname(parent_dir)
            else:
                base_action_dir = parent_dir
            
            annotation_dir = os.path.join(base_action_dir, OUTPUT_ANNOTATION_DIR_NAME)
            os.makedirs(annotation_dir, exist_ok=True)
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            xml_output_path = os.path.join(annotation_dir, f"{video_name}.xml")
            
            if os.path.exists(xml_output_path):
                # print(f"Skipping {video_name}, XML already exists.")
                continue
                
            # Process
            keypoints = process_video(model, video_path)
            if keypoints is not None:
                create_xml_annotation(video_path, xml_output_path, keypoints)

    print("Extraction complete.")

if __name__ == "__main__":
    main()

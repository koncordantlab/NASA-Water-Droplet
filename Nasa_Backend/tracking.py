import os
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'Nasa_Backend\\app_root\\weights_DP(6).pt'    # Path to YOLO model
VIDEO_PATH = "Nasa_Backend\\20 seconds.mp4"              # Input video
OUTPUT_PATH = "Nasa_Backend\\output\\output_tracked.mp4"          # Output video

# **Visualization settings**
GREEN_BOX_COLOR = (0, 255, 0)
RED_BOX_COLOR = (0, 0, 255)
THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6

def load_model(model_path: str) -> YOLO:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)

def open_video(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return cap, fps, width, height

def create_video_writer(output_path, fps, width, height):
    """Instantiate a video writer which allows the program to generate new video frames.

    Args:
        output_path (str)
        fps (int)
        width (int)
        height (int)

    Returns:
        VideoWriter
    """
    # Source - https://stackoverflow.com/questions/30103077/what-is-the-codec-for-mp4-videos-in-python-opencv
    # Posted by Gonzalo Garcia, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-21, License - CC BY-SA 4.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def compute_iou(boxA, boxB):
    """Compute (i)ntersection (o)ver (u)nion with respect to bounding boxes.

    Args:
        boxA (array): Box with two corners.
        boxB (array): Box with two corners.

    Returns:
        float: Calculated ration of area intersected over the union of boxes.
    """
    x1 = max(boxA[0], boxB[0]) 
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0

def draw_box(frame, box, obj_id):
    """Draws green box for successful detections

    Args:
        frame (array): Single video frame.
        box (array): Box with two corners.
        obj_id (int): Unique object id
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN_BOX_COLOR, 2)

    label = f"ID {obj_id}"
    cv2.putText(frame, label, (x1, y1 - 5),
                FONT, FONT_SCALE, GREEN_BOX_COLOR, THICKNESS)

def draw_lost_box(frame, box, obj_id):
    """Draws red box around formerly detected objects.

    Args:
        frame (array): Single video frame.
        box (array): Box with corner locations
        obj_id (int): Unique ID for this detection.
    """
    x1, y1, x2, y2 = map(int, box)
    label = "ID" + obj_id + "(lost)"
    cv2.rectangle(frame, (x1, y1), (x2, y2), RED_BOX_COLOR, 2) 
    cv2.putText(frame, label, (x1, y1 - 5),
                FONT, FONT_SCALE, GREEN_BOX_COLOR, THICKNESS)

def track_all_objects(model, video_path, output_path):
    """Multi-object tracking loop creating unique IDs for all detected objects across frames.

    Args:
        model (YOLO): Pre-trained YOLO object detection model.
        video_path (str)
        output_path (str)

    Raises:
        RuntimeError: Video read error.
    """

    #TODO Class Detection
    #TODO Bridge change detection
    #TODO Overlap integration
    #TODO Business logic for detecting parents (drawing IOU circle)

    cap, fps, width, height = open_video(video_path)
    writer = create_video_writer(output_path, fps, width, height)

    tracked_objects = {}  # obj_id -> box
    next_id = 1

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    # Detect objects in first frame
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        tracked_objects[next_id] = xyxy
        next_id += 1
    
    # Draw first frame
    frame_id = 1
    for obj_id, box in tracked_objects.items():
        draw_box(frame, box, obj_id)
    writer.write(frame)
    print("Frame complete", frame_id)
    frame_id += 1

    # Process the rest
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = [b.xyxy[0].cpu().numpy() for b in results.boxes]

        updated_objects = {}

        # Match detections to existing tracked objects
        matched = set()
        for obj_id, prev_box in tracked_objects.items():
            best_iou = 0
            best_idx = None

            for i, det in enumerate(detections):
                iou = compute_iou(prev_box, det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx is not None:
                updated_objects[obj_id] = detections[best_idx]
                matched.add(best_idx)
            else:
                updated_objects[obj_id] = prev_box


        # Any remaining detections become NEW objects
        for i, det in enumerate(detections):
            if i not in matched:
                updated_objects[next_id] = det
                next_id += 1
        
        # Draw all boxes
        for obj_id, box in updated_objects.items():
            prev_box = tracked_objects.get(obj_id)
            if (prev_box is not None) and (np.array_equal(box, prev_box)):
                draw_lost_box(frame, box, obj_id)      # red
            else:
                draw_box(frame, box, obj_id)  # green
        
        tracked_objects = updated_objects
        writer.write(frame)
        print("Frame complete", frame_id)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"Tracking complete. Output saved to: {output_path}")


if __name__ == "__main__":
    print("Loading YOLO model...")
    model = load_model(MODEL_PATH)

    print("Tracking all objects...")
    track_all_objects(model, VIDEO_PATH, OUTPUT_PATH)

    print("Done.")
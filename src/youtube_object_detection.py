import cv2
from yt_dlp import YoutubeDL
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
from video_story_generator import generate_video_story
from transformers import pipeline, VideoMAEFeatureExtractor, VideoMAEForVideoClassification
from PIL import Image

def setup_output_dir():
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def process_youtube_video(url, start_time=(0, 0), duration=60):
    """
    Process YouTube video stream for specified duration
    Args:
        url (str): YouTube video URL
        start_time (tuple): (minutes, seconds) tuple for start position
        duration (int): Duration to process in seconds
    """
    try:
        # Setup
        output_dir = setup_output_dir()
        output_path = output_dir / "processed_video.mp4"
        
        # Ensure start_time is a tuple
        if isinstance(start_time, (int, float)):
            start_time = (int(start_time), 0)
        start_minutes, start_seconds = start_time
        
        # Calculate start position in seconds
        start_position = start_minutes * 60 + start_seconds
        
        # Load YOLO model - switched to YOLOv8s for better accuracy
        print("Loading YOLO model (YOLOv8s)...")
        model = YOLO('yolov8s.pt')
        
        # Get YouTube video URL using yt-dlp
        print(f"Accessing YouTube video: {url}")
        print(f"Starting at: {start_minutes} minutes {start_seconds} seconds")
        
        with YoutubeDL({
            'format': 'best[ext=mp4]',
            'quiet': True
        }) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info['url']
        
        # Open video stream
        cap = cv2.VideoCapture(video_url)
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_MSEC, start_position * 1000)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Calculate total frames to process
        total_frames = int(fps * duration)
        
        # Setup progress bar
        progress_bar = tqdm(
            total=total_frames,
            desc="Processing video",
            unit="frames",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}]"
        )
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        print("\nProcessing video stream...")
        
        # Add scene understanding model
        scene_classifier = pipeline("image-classification", 
                                 model="microsoft/resnet-50", 
                                 top_k=5)
        
        # Initialize lists to store detections and scene info
        all_detections = []
        scene_info = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # YOLO detection for objects
            results = model(frame, conf=0.25)
            result = results[0]  # Get first result
            
            # Store detection data we need
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    detections.append(class_name)
            
            all_detections.extend(detections)
            
            # Scene understanding
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            scene_results = scene_classifier(pil_image)
            
            # Update scene info
            for pred in scene_results:
                if pred['label'] not in scene_info:
                    scene_info[pred['label']] = []
                scene_info[pred['label']].append(pred['score'])
            
            # Draw detections and write frame
            annotated_frame = process_detections(frame, result)
            out.write(annotated_frame)
            
            frame_count += 1
            progress_bar.update(1)
            
            if frame_count >= total_frames:
                break
        
        # Count total detections
        from collections import Counter
        detection_counts = dict(Counter(all_detections))
        
        # Average scene scores
        averaged_scene_info = {
            label: sum(scores) / len(scores) 
            for label, scores in scene_info.items()
        }
        
        # Cleanup
        progress_bar.close()
        cap.release()
        out.release()
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        
        # Generate story using accumulated detections
        print("\nGenerating story from video analysis...")
        generate_video_story(detection_counts, averaged_scene_info, 
                           duration, url, (start_minutes, start_seconds))
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        print(f"Error details: {e.__class__.__name__}")
        import traceback
        print(f"Error traceback: {traceback.format_exc()}")
        return None

def process_detections(frame, result):
    """Process and visualize YOLO detections"""
    if result.boxes is not None:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[class_id]
            
            # Define color based on class
            color = get_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with class name and confidence
            label = f'{class_name} {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1-label_height-10), (x1+label_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def get_color(class_id):
    """Generate consistent color for class ID"""
    np.random.seed(class_id)
    return tuple(map(int, np.random.randint(0, 255, 3)))

def setup_action_recognition():
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    return feature_extractor, model

if __name__ == "__main__":
    # Get input from user
    url = input("Enter YouTube URL: ")
    
    # Get start time
    start_minutes = int(input("Enter start time (minutes): ") or "0")
    start_seconds = int(input("Enter start time (seconds): ") or "0")
    
    # Get duration
    duration = int(input("Enter duration in seconds (default 60): ") or "60")
    
    processed_video = process_youtube_video(
        url, 
        start_time=(start_minutes, start_seconds),
        duration=duration
    )
    
    if processed_video:
        print(f"\nVideo processing completed successfully!")
        print(f"Output video: {processed_video}")
    else:
        print("\nVideo processing failed!") 
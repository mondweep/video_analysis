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

def setup_models():
    """Setup all required models for comprehensive video analysis"""
    models = {
        # Use YOLOv8x with optimized settings
        'object_detector': YOLO('yolov8x.pt'),
        
        # Add a backup detector for cross-validation
        'backup_detector': YOLO('yolov8l.pt'),
        
        # Scene understanding
        'scene_classifier': pipeline("image-classification", 
                                   model="microsoft/resnet-50",
                                   top_k=5)
    }
    
    # Optimize detection settings
    models['object_detector'].conf = 0.35  # Increased confidence threshold
    models['object_detector'].iou = 0.45   # Better box overlap threshold
    models['backup_detector'].conf = 0.35
    
    return models

def process_frame_comprehensive(frame, models, frame_buffer):
    """Process a single frame with all models"""
    results = {
        'objects': [],
        'scenes': [],
        'actions': [],
        'attributes': []
    }
    
    try:
        # 1. Primary Object Detection
        yolo_results = models['object_detector'](frame, conf=0.35)
        backup_results = models['backup_detector'](frame, conf=0.35)
        
        # Combine detections from both models
        detected_objects = set()
        
        # Process primary detector results
        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])
            class_name = yolo_results[0].names[class_id]
            conf = float(box.conf[0])
            
            # Only add high-confidence detections
            if conf > 0.35:
                detected_objects.add(class_name)
        
        # Cross-validate with backup detector
        for box in backup_results[0].boxes:
            class_id = int(box.cls[0])
            class_name = backup_results[0].names[class_id]
            conf = float(box.conf[0])
            
            if conf > 0.35:
                detected_objects.add(class_name)
        
        results['objects'].extend(list(detected_objects))
        
        # 2. Scene Classification
        scene_results = models['scene_classifier'](Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        results['scenes'] = [pred['label'] for pred in scene_results]
        
    except Exception as e:
        print(f"Error in frame processing: {str(e)}")
    
    return results

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
        
        # Ensure start_time is properly formatted
        if isinstance(start_time, (int, float)):
            start_minutes = int(start_time)
            start_seconds = 0
        else:
            start_minutes, start_seconds = start_time
            
        # Calculate start position in seconds
        start_position = start_minutes * 60 + start_seconds
        
        # Initialize all models
        print("Loading models...")
        models = setup_models()
        
        # Initialize frame buffer for action recognition
        frame_buffer = []
        
        # Initialize results storage
        comprehensive_results = {
            'objects': [],
            'scenes': {},
            'actions': {},
            'temporal_info': []
        }
        
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with all models
            frame_results = process_frame_comprehensive(frame, models, frame_buffer)
            
            # Get YOLO results directly for visualization
            yolo_results = models['object_detector'](frame, conf=0.3)
            
            # Draw detections with bounding boxes
            annotated_frame = process_detections(frame, yolo_results)
            out.write(annotated_frame)
            
            # Store results
            comprehensive_results['objects'].extend(frame_results['objects'])
            for key in ['scenes', 'actions']:
                for item in frame_results[key]:
                    comprehensive_results[key][item] = comprehensive_results[key].get(item, 0) + 1
            
            # Store temporal information
            if frame_results['actions']:
                comprehensive_results['temporal_info'].append({
                    'timestamp': frame_count / fps,
                    'actions': frame_results['actions']
                })
            
            frame_count += 1
            progress_bar.update(1)
            
            if frame_count >= total_frames:
                break
        
        # Cleanup
        progress_bar.close()
        cap.release()
        out.release()
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        
        # Generate enhanced story using comprehensive results
        print("\nGenerating enhanced story from comprehensive video analysis...")
        generate_video_story(
            objects=comprehensive_results['objects'],
            scenes=comprehensive_results['scenes'],
            actions=comprehensive_results['actions'],
            temporal_info=comprehensive_results['temporal_info'],
            duration=duration,
            video_url=url,
            start_time=(start_minutes, start_seconds)  # Pass as tuple
        )
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        print(f"Error details: {e.__class__.__name__}")
        import traceback
        print(f"Error traceback: {traceback.format_exc()}")
        return None

def process_detections(frame, yolo_results):
    """Process and visualize detections with bounding boxes"""
    # Draw bounding boxes and labels for each detection
    for i, box in enumerate(yolo_results[0].boxes):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class name and confidence
        class_id = int(box.cls[0])
        class_name = yolo_results[0].names[class_id]
        conf = float(box.conf[0])
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label with confidence
        label = f'{class_name} {conf:.2f}'
        cv2.putText(frame, 
                   label, 
                   (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5,  # Font scale
                   (0, 255, 0),  # Color (BGR)
                   2)  # Thickness
    
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
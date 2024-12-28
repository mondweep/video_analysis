import cv2
from ultralytics import YOLO
from pathlib import Path
import subprocess
import shutil
import librosa
import soundfile as sf
import numpy as np

def isolate_guitar_audio(audio_path, output_path):
    """
    Enhanced guitar isolation with better speech removal
    """
    print("Isolating guitar audio...")
    try:
        # Load audio with error handling
        print("Loading audio file...")
        y, sr = librosa.load(audio_path, sr=None)
        
        # Initial check and cleanup
        if not np.isfinite(y).all():
            print("Warning: Initial audio contains invalid values. Cleaning...")
            y = np.nan_to_num(y)
            y = np.clip(y, -1.0, 1.0)
        
        # Apply STFT to original audio
        print("Applying STFT...")
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D))
        
        # Filter frequencies
        print("Filtering frequencies...")
        freqs = librosa.fft_frequencies(sr=sr)
        guitar_mask = (freqs >= 70) & (freqs <= 1200)  # Guitar range
        voice_mask = (freqs >= 85) & (freqs <= 255)    # Voice range
        
        # Create and apply masks
        mask = guitar_mask & ~voice_mask
        mask = np.expand_dims(mask, axis=1)  # Match STFT shape
        
        # Apply mask to STFT
        D_filtered = D * mask
        
        # Inverse STFT
        print("Reconstructing audio...")
        filtered_audio = librosa.istft(D_filtered)
        
        # Normalize and cleanup
        filtered_audio = librosa.util.normalize(filtered_audio)
        filtered_audio = np.clip(filtered_audio, -1.0, 1.0)
        
        # Save filtered audio
        print("Saving processed audio...")
        sf.write(output_path, filtered_audio, sr)
        print("Guitar isolation complete!")
        return output_path
        
    except Exception as e:
        print(f"Error in audio processing: {e}")
        return None

def track_and_separate_persons(video_path):
    """
    Track and separate persons in video
    """
    print(f"Processing video: {video_path}")
    
    # Initialize YOLO model with correct tracking configuration
    model = YOLO('yolov8n.pt')
    
    output_dir = Path("output/persons")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Get video properties
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    # Define chunk size (e.g., 30 seconds)
    chunk_duration = 30  # seconds
    chunk_frames = int(chunk_duration * fps)
    
    # Calculate chunk ranges
    chunk_ranges = []
    for start_frame in range(0, total_frames, chunk_frames):
        end_frame = min(start_frame + chunk_frames, total_frames)
        chunk_ranges.append((start_frame, end_frame))
    
    # Track persons in video with higher confidence threshold
    print("\nTracking persons (confidence threshold: 75%)...")
    person_tracks = {}
    
    try:
        # Process each chunk
        for chunk_idx, (start_frame, end_frame) in enumerate(chunk_ranges, 1):
            print(f"\nProcessing chunk {chunk_idx}/{len(chunk_ranges)}")
            
            # Create temporary file for this chunk
            temp_input = temp_dir / f"temp_input_{chunk_idx-1}.mp4"
            
            # Extract chunk using ffmpeg
            extract_command = [
                'ffmpeg', '-i', str(video_path),
                '-ss', str(start_frame),
                '-t', str(end_frame - start_frame),
                '-c:v', 'libx264', '-c:a', 'aac',
                str(temp_input)
            ]
            subprocess.run(extract_command)
            
            try:
                # Use track method with correct configuration
                results = model.track(
                    source=str(temp_input),
                    tracker="botsort.yaml",  # Use the built-in tracker configuration
                    save=False,
                    stream=True
                )
                
                # Process tracking results
                if results:
                    for frame_idx, r in enumerate(results):
                        if hasattr(r, 'boxes') and hasattr(r.boxes, 'id') and r.boxes.id is not None:
                            boxes = r.boxes
                            for box_idx in range(len(boxes)):
                                track_id = int(boxes.id[box_idx])
                                conf = float(boxes.conf[box_idx])
                                
                                if conf >= 0.75:  # High confidence threshold
                                    if track_id not in person_tracks:
                                        person_tracks[track_id] = []
                                    
                                    person_tracks[track_id].append({
                                        'frame': frame_idx + start_frame,
                                        'time': (frame_idx + start_frame) / fps,
                                        'bbox': boxes.xywh[box_idx].tolist(),
                                        'confidence': conf
                                    })
            
            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {str(e)}")
                continue

        # After all chunks are processed, handle the tracks
        if person_tracks:
            print(f"\nFound {len(person_tracks)} persons to process")
            
            # Process each person's tracks
            for person_id, tracks in person_tracks.items():
                if not tracks:
                    continue
                    
                # ... rest of the person processing code remains the same ...
                
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    video_path = os.getenv('VIDEO_FILE')
    if not video_path:
        raise ValueError("VIDEO_FILE not found in .env file")
    
    track_and_separate_persons(video_path)

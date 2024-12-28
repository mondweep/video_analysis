"""
Video Processing Requirements Checklist:
1. Video Processing:
    ☐ Process in 30-second segments
    ☐ Batch size of 32 frames
    ☐ YOLO detection confidence > 75%
    ☐ Even dimensions for h264 compatibility
    ☐ Generate individual videos per tracked person
    ☐ Display processing progress every 10 seconds
    ☐ Process only specified time range from .env file
        ☐ START_TIME in minutes
        ☐ END_TIME in minutes (optional)

2. Audio Processing:
    ☐ Guitar frequency isolation (82Hz-1175Hz)
    ☐ Preserve processed audio in output
    ☐ Handle audio errors gracefully
    ☐ Include processed audio in per-person videos

3. Performance & Resources:
    ☐ Memory-efficient segment processing
    ☐ Temporary file management
    ☐ Progress tracking with:
        ☐ Frames processed/total
        ☐ Processing speed (FPS)
        ☐ ETA calculation
        ☐ 10-second update interval
        ☐ Clear segment progress (current/total)
        ☐ Cumulative progress across segments
    ☐ Error handling and recovery

4. Output Requirements:
    ☐ Maintain original video quality
    ☐ Synchronized audio-video
    ☐ Multiple person tracking support
    ☐ Clean temporary files on completion
    ☐ Separate video file per detected person

Version: 1.0
Last Updated: [date]
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import time
from datetime import datetime, timedelta
import moviepy.editor as mp
import librosa
import soundfile as sf
import numpy as np
from scipy import signal
import math
import subprocess
import shutil
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file

def isolate_guitar_audio(input_video, output_audio):
    """Extract audio with better quality preservation"""
    try:
        print("Processing audio with ffmpeg...")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(input_video),
            '-af', 'volume=1.5,highpass=f=50:width_type=q:width=0.5,lowpass=f=12000:width_type=q:width=0.5',  # Wider frequency range
            '-acodec', 'pcm_s16le',
            '-ar', '48000',  # Higher sample rate
            str(output_audio)
        ], check=True)
        return True
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        try:
            print("Attempting to save original audio...")
            subprocess.run([
                'ffmpeg', '-y',
                '-i', str(input_video),
                '-acodec', 'pcm_s16le',
                str(output_audio)
            ], check=True)
            return True
        except Exception as e2:
            print(f"Failed to save original audio: {str(e2)}")
        return False

def process_audio_channel(y, sr):
    """Process a single channel of audio"""
    # Preserve more of the original signal
    y_harmonic = librosa.effects.harmonic(y, margin=3.0)
    
    # Gentler filtering
    y_filtered = y_harmonic.copy()
    
    # Apply very mild noise reduction
    S = librosa.stft(y_filtered)
    S_filtered = librosa.decompose.nn_filter(
        np.abs(S),
        aggregate=np.median,
        metric='cosine',
        width=3
    )
    y_filtered = librosa.istft(S_filtered * np.exp(1.j * np.angle(S)))
    
    # Mix with original with more weight to original
    y_mixed = 0.8 * y + 0.2 * y_filtered
    
    # Normalize while preserving dynamics
    y_mixed = librosa.util.normalize(y_mixed) * 0.95
    
    return y_mixed

def process_video_chunk(video_clip, start_frame, end_frame, total_frames, start_time):
    """Helper function to process a chunk of video frames"""
    processed_frames = 0
    last_update = time.time()
    
    def process_frame(get_frame_func, t):
        nonlocal processed_frames, last_update
        try:
            frame = get_frame_func(t)
            processed_frames += 1
            
            # Update progress every 5 seconds
            current_time = time.time()
            if current_time - last_update >= 5:
                elapsed_time = current_time - start_time
                overall_progress = ((start_frame + processed_frames) / total_frames) * 100
                fps_processing = processed_frames / (current_time - last_update)
                
                print("\033[K", end="\r")  # Clear line
                print(f"Progress: [{start_frame + processed_frames}/{total_frames}] {overall_progress:.1f}%")
                print(f"Processing speed: {fps_processing:.2f} FPS")
                print(f"Elapsed time: {str(timedelta(seconds=int(elapsed_time)))}")
                print("\033[3A", end="")  # Move cursor up 3 lines
                
                last_update = current_time
            
            return frame
            
        except Exception as e:
            print(f"\nError processing frame: {e}")
            return get_frame_func(t)
    
    return video_clip.fl(process_frame)

def process_video(input_video, output_video, start_time, end_time, output_dir, time_range):
    """Process video with improved progress tracking"""
    SEGMENT_DURATION = 30
    BATCH_SIZE = 32
    total_processed_frames = 0
    
    try:
        # Create base temp directory
        base_temp_dir = Path("output/temp").absolute()
        base_temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create segments directory
        segments_dir = base_temp_dir / "segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Process audio first
        print("Processing audio...")
        temp_audio = base_temp_dir / "processed_audio.wav"
        if not isolate_guitar_audio(input_video, temp_audio):
            print("Warning: Audio processing failed, will use original audio...")
            # Extract original audio as fallback
            try:
                video = mp.VideoFileClip(input_video)
                if video.audio is not None:
                    video.audio.write_audiofile(str(temp_audio))
                video.close()
            except Exception as e:
                print(f"Error extracting original audio: {str(e)}")
        
        # Initialize YOLO model
        print("\nLoading YOLO model...")
        model = YOLO('yolov8n.pt')
        
        # Process video in segments
        video = cv2.VideoCapture(input_video)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate segment information
        frames_per_segment = int(SEGMENT_DURATION * fps)
        start_frame = int(start_time * 60 * fps)  # Convert minutes to frames
        end_frame = int(end_time * 60 * fps) if end_time else int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = end_frame - start_frame
        
        print(f"\nVideo Processing Parameters:")
        print(f"FPS: {fps}")
        print(f"Start time: {start_time} minutes ({start_frame} frames)")
        print(f"End time: {end_time} minutes ({end_frame} frames)")
        print(f"Total duration: {(end_frame - start_frame) / fps / 60:.2f} minutes")
        
        # Calculate segments
        segments = []
        current_frame = start_frame
        while current_frame < end_frame:
            segment_end = min(current_frame + frames_per_segment, end_frame)
            segments.append((current_frame, segment_end))
            current_frame = segment_end
        
        print(f"\nDebug: Planning {len(segments)} segments:")
        for i, (seg_start, seg_end) in enumerate(segments, 1):
            print(f"Segment {i}: frames {seg_start}-{seg_end} ({seg_start/fps/60:.2f}-{seg_end/fps/60:.2f} minutes)")
        
        # Calculate total frames to process
        total_frames = end_frame - start_frame
        print(f"\nTotal frames to process: {total_frames}")
        
        # Process each segment
        for i, (segment_start, segment_end) in enumerate(segments, 1):
            print(f"\nProcessing segment {i}/{len(segments)}")
            print(f"Frame range: {segment_start}-{segment_end}")
            
            segment_path = output_dir / "temp" / "segments" / f"segment_{segment_start}_{segment_end}.mp4"
            processed_frames = process_segment(video, segment_start, segment_end, segment_path, 
                                            BATCH_SIZE, model, i, len(segments))
            
            total_processed_frames += processed_frames
            
            # Calculate and display overall progress
            overall_progress = (total_processed_frames / total_frames) * 100
            print(f"\nOverall Progress: {overall_progress:.1f}%")
            print(f"Frames processed: {total_processed_frames}/{total_frames}")
            print(f"Segments completed: {i}/{len(segments)}")
            
            # Estimated time remaining
            if i > 0:
                avg_time_per_segment = time.time() - start_time / i
                eta = avg_time_per_segment * (len(segments) - i)
                print(f"Estimated time remaining: {str(timedelta(seconds=int(eta)))}")
        
        # Verify segments exist
        print("\nVerifying processed segments...")
        valid_segments = []
        for segment_start, segment_end in segments:
            segment_path = output_dir / "temp" / "segments" / f"segment_{segment_start}_{segment_end}.mp4"
            if segment_path.exists():
                valid_segments.append(segment_path)
                print(f"Verified: {segment_path}")
            else:
                print(f"Missing: {segment_path}")
        
        if not valid_segments:
            raise RuntimeError("No valid segments found to combine")
        
        # Create segments list file
        segments_list = base_temp_dir / "segments.txt"
        print(f"\nCreating segments list at: {segments_list}")
        with open(segments_list, "w") as f:
            for segment in valid_segments:
                f.write(f"file '{segment.absolute()}'\n")
        
        # Combine segments with audio
        print("\nCombining segments with audio...")
        combine_segments_with_audio(
            segments_list,
            temp_audio,
            output_video,
            start_time,
            end_time - start_time if end_time else None
        )
        
        # Clean up
        print("\nCleaning up temporary files...")
        shutil.rmtree(base_temp_dir)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

def process_segment(video, segment_start, segment_end, output_path, batch_size, model, current_segment, total_segments):
    """Process a single segment with improved progress tracking"""
    processed_frames = 0
    frames_batch = []
    last_update = time.time()
    start_time = time.time()
    segment_total_frames = segment_end - segment_start
    
    try:
        # Ensure we're at the correct starting position
        video.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
        current_frame = segment_start
        
        # Initialize video writer
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        frames_processed_since_last_update = 0
        
        while current_frame < segment_end:
            # Read frame
            ret, frame = video.read()
            if not ret:
                print(f"\nFailed to read frame at position {current_frame}")
                break
                
            frames_batch.append(frame)
            processed_frames += 1
            frames_processed_since_last_update += 1
            
            # Process batch when full or at end
            if len(frames_batch) >= batch_size or current_frame == segment_end - 1:
                # Run detection with improved parameters
                results = model.track(
                    frames_batch,
                    persist=True,
                    conf=0.4,        # Lower confidence threshold to detect more objects
                    iou=0.5,         # Lower IOU threshold to detect more objects
                    classes=None,    # Remove classes filter to detect all objects
                    line_width=2,    # Slightly thicker lines
                    show_boxes=True,
                    show_conf=True,  # Show confidence scores
                    show_labels=True,
                    tracker="botsort.yaml",  # Use botsort tracker instead of bytetrack
                    verbose=False
                )
                
                # Process each frame in batch
                for frame, result in zip(frames_batch, results):
                    annotated_frame = process_detections(frame, result)
                    out.write(annotated_frame)
                
                frames_batch = []
                
                # Update progress more frequently
                if time.time() - last_update >= 5:  # Update every 5 seconds
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = frames_processed_since_last_update / (current_time - last_update)
                    segment_progress = (processed_frames / segment_total_frames) * 100
                    
                    # Calculate ETA more accurately
                    frames_remaining = segment_total_frames - processed_frames
                    segments_remaining = total_segments - current_segment
                    
                    if fps > 0:
                        time_per_frame = 1 / fps
                        eta_seconds = frames_remaining * time_per_frame + \
                                    (segments_remaining * segment_total_frames * time_per_frame)
                        eta = str(timedelta(seconds=int(eta_seconds)))
                    else:
                        eta = "calculating..."
                    
                    print(f"\nSegment {current_segment}/{total_segments}")
                    print(f"Progress: {processed_frames}/{segment_total_frames} frames ({segment_progress:.1f}%)")
                    print(f"Processing Speed: {fps:.1f} FPS")
                    print(f"Time Elapsed: {str(timedelta(seconds=int(elapsed)))}")
                    print(f"Estimated time remaining: {eta}")
                    
                    last_update = current_time
                    frames_processed_since_last_update = 0  # Reset counter
            
            current_frame += 1
            
        out.release()
        return processed_frames
        
    except Exception as e:
        print(f"\nError processing segment: {str(e)}")
        if out is not None:
            out.release()
        raise

def combine_segments_with_audio(segments_file, audio_path, output_path, start_time, duration):
    """Combine video segments with original audio"""
    try:
        # First combine video segments
        temp_video = str(Path(output_path).with_suffix('.temp.mp4'))
        subprocess.run([
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(segments_file),
            '-c', 'copy',
            str(temp_video)
        ], check=True)

        # Extract the exact audio segment we need from original video
        temp_audio = str(Path(output_path).parent / 'temp_audio.wav')
        start_seconds = start_time * 60  # Convert minutes to seconds
        duration_seconds = duration * 60  # Convert minutes to seconds
        
        subprocess.run([
            'ffmpeg', '-y',
            '-i', str(audio_path),
            '-ss', str(start_seconds),
            '-t', str(duration_seconds),
            '-acodec', 'pcm_s16le',  # Use uncompressed audio
            str(temp_audio)
        ], check=True)

        # Combine video with extracted audio
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', temp_audio,
            '-c:v', 'copy',      # Copy video stream
            '-c:a', 'aac',       # Convert audio to AAC
            '-b:a', '320k',      # High audio bitrate
            str(output_path)
        ], check=True)

        # Clean up temporary files
        Path(temp_video).unlink(missing_ok=True)
        Path(temp_audio).unlink(missing_ok=True)
        
        return True
    except Exception as e:
        print(f"Error combining segments with audio: {e}")
        return False

def save_frames_for_person(frames, person_id, chunk_idx, fps):
    # Create frames directory if it doesn't exist
    frames_dir = Path("output/temp/frames") / f"person_{person_id}"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Create frames list file
    frames_list_path = Path("output/temp") / f"frames_person_{person_id}.txt"
    with open(frames_list_path, "w") as f:
        for i, frame in enumerate(frames):
            frame_path = frames_dir / f"chunk_{chunk_idx}_frame_{i}.jpg"
            cv2.imwrite(str(frame_path), frame)
            f.write(f"file '{frame_path.absolute()}'\n")
            f.write(f"duration {1/fps}\n")
    
    return frames_list_path

def create_person_video(person_id, frames_dir, audio_path, output_path=None):
    """
    Create a video for a specific person with audio
    
    Args:
        person_id (int): ID of the person
        frames_dir (Path): Directory containing the frames
        audio_path (Path): Path to the audio file
        output_path (Path, optional): Final output path. If None, saves to temp directory
    """
    # Create temporary video without audio
    temp_video = Path("output/temp") / f"person_{person_id}_no_audio.mp4"
    
    # Create frames list file
    frames_list_path = Path("output/temp") / f"frames_person_{person_id}.txt"
    with open(frames_list_path, "w") as f:
        for frame_path in sorted(frames_dir.glob(f"person_{person_id}_frame_*.jpg")):
            f.write(f"file '{frame_path.absolute()}'\n")
            f.write(f"duration {1/25}\n")  # Assuming 25 fps, adjust if needed
    
    # Create video without audio
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(frames_list_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "25",
        str(temp_video)
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating video for person {person_id}: {e}")
        return None

    # Add audio to the video
    final_output = output_path if output_path else Path("output/temp") / f"person_{person_id}_final.mp4"
    add_audio_cmd = [
        'ffmpeg', '-y',
        '-i', str(temp_video),
        '-i', str(audio_path),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        str(final_output)
    ]
    try:
        subprocess.run(add_audio_cmd, check=True)
        return final_output
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio for person {person_id}: {e}")
        return None

def process_frame(frame, person_id, output_path):
    """Process and save a single frame for a person"""
    if frame is None:
        return
        
    # Ensure dimensions are even numbers
    height, width = frame.shape[:2]
    new_width = width - (width % 2)  # Make width even
    new_height = height - (height % 2)  # Make height even
    
    if new_width != width or new_height != height:
        frame = cv2.resize(frame, (new_width, new_height))
        
    # Save the frame
    cv2.imwrite(output_path, frame)

def process_detections(frame, result):
    """Process YOLO detections with improved object visualization"""
    # Define colors for different object classes
    colors = {
        'person': (0, 255, 0),    # Green
        'chair': (255, 0, 0),     # Red
        'door': (0, 0, 255),      # Blue
        'whiteboard': (255, 255, 0),  # Yellow
        'tv': (128, 0, 128),      # Purple
        'book': (165, 42, 42),    # Brown
        'bottle': (0, 128, 255),  # Light blue
    }
    
    if result.boxes is not None:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[class_id]
            
            # Get tracking ID if available
            track_id = None
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
            
            # Only show detections with confidence > 0.4
            if conf > 0.4:
                # Get color for this class (default to white if not in colors dict)
                color = colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with class name, ID, and confidence
                label = f'{class_name}'
                if track_id is not None and class_name == 'person':
                    label += f' #{track_id}'
                label += f' {conf:.2f}'
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-label_height-10), (x1+label_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection)

def merge_person_videos(person_videos, output_dir):
    """Merge videos of the same person based on spatial overlap"""
    merged_videos = {}
    
    # Group videos by person based on bounding box overlap
    for video_path in person_videos:
        person_id = int(video_path.stem.split('_')[1])
        if person_id not in merged_videos:
            merged_videos[person_id] = []
        merged_videos[person_id].append(video_path)
    
    # Merge videos for each person
    for person_id, videos in merged_videos.items():
        if len(videos) > 1:
            # Create concat file
            concat_file = output_dir / f"person_{person_id}_concat.txt"
            with open(concat_file, "w") as f:
                for video in sorted(videos):
                    f.write(f"file '{video.absolute()}'\n")
            
            # Merge videos
            output_path = output_dir / f"person_{person_id}_final.mp4"
            subprocess.run([
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ], check=True)
            
            # Clean up individual videos
            for video in videos:
                video.unlink()
            concat_file.unlink()

def generate_person_video(frames, output_path, fps=25.0):
    """Generate video for a person with improved error handling"""
    try:
        if not frames:
            print(f"No frames to process for {output_path}")
            return False
            
        # Create temporary path for initial video
        temp_path = str(Path(output_path).with_suffix('.temp.mp4'))
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Failed to create video writer for {output_path}")
            return False
            
        # Write frames
        for frame in frames:
            if frame is not None and frame.shape[:2] == (height, width):
                out.write(frame)
            else:
                print(f"Skipping invalid frame for {output_path}")
                
        out.release()
        
        # Move temp file to final location
        if Path(temp_path).exists():
            shutil.move(temp_path, output_path)
            return True
        else:
            print(f"Temp file not created: {temp_path}")
            return False
            
    except Exception as e:
        print(f"Error generating video for {output_path}: {str(e)}")
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        return False

def cleanup_temp_files(temp_dir):
    """Clean up old temporary files from previous runs"""
    try:
        # Clean up old files from previous runs
        if temp_dir.exists():
            for item in temp_dir.glob('*'):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            print("\nCleaned up old temporary files")
    except Exception as e:
        print(f"\nWarning: Failed to clean up some temporary files: {str(e)}")

def main(video_file):
    try:
        # Setup paths
        temp_dir = Path("output/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old temp files at start instead of end
        cleanup_temp_files(temp_dir)
        
        # Create output directory relative to the script location (src/)
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get time range from .env file
        start_time = float(os.getenv('START_TIME', 0))  # in minutes
        end_time = float(os.getenv('END_TIME', 0)) or None  # in minutes
        
        # Generate output filename based on input filename and time range
        input_filename = Path(video_file).stem
        time_range = f"{int(start_time)}-{int(end_time) if end_time else 'end'}"
        output_video = str(output_dir / f"{input_filename}_{time_range}_processed.mp4")
        
        print(f"Processing video: {video_file}")
        print(f"Time range: {start_time} to {end_time if end_time else 'end'} minutes")
        print(f"Output will be saved to: {output_video}")
        
        # Get input video fps at the start of the function
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Add a small delay after combining segments
        def process_and_verify():
            process_video(video_file, output_video, start_time, end_time, output_dir, time_range)
            time.sleep(2)  # Add a small delay to ensure file is written
            if not Path(output_video).exists():
                raise RuntimeError(f"Failed to create output file: {output_video}")
            
        process_and_verify() 
        
        print("\nProcessing complete! Temporary files are preserved in the output/temp directory")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    video_file = os.getenv('VIDEO_FILE')
    if not video_file:
        raise ValueError("VIDEO_FILE environment variable not set")
    main(video_file) 
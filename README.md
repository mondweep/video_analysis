# Video Analysis with YOLO and Gemini AI

A Python-based video analysis tool that combines object detection, scene understanding, and AI-generated storytelling.

## YouTube Video Analysis (youtube_object_detection.py & video_story_generator.py)

### Key Features
- **YouTube Video Processing**: Download and analyze YouTube videos from specified timestamps
- **Object Detection**: Uses YOLOv8s for real-time object detection and tracking
- **Scene Understanding**: Implements Microsoft ResNet-50 for scene classification
- **AI Storytelling**: Generates creative stories and poems based on video content using Google's Gemini AI
- **Visual Output**: Creates annotated videos with bounding boxes and labels
- **Progress Tracking**: Real-time progress bars and status updates

### Story Generation Features
- Family-friendly content generation
- Scene-based storytelling
- Custom safety filters for content appropriateness
- Automatic story and poem generation
- Output saved as timestamped text files

## Local Video Processing (local_video_tracking.py)

### Key Features
- **Segment Processing**: Handles videos in 30-second segments
- **Batch Processing**: Processes 32 frames at a time
- **High-Quality Detection**: YOLO detection with 75%+ confidence
- **Audio Processing**:
  - Guitar frequency isolation (82Hz-1175Hz)
  - Audio preservation in output
  - Error handling for audio processing
  
### Advanced Features
- **Person Tracking**:
  - Individual video generation per tracked person
  - Spatial overlap detection
  - Merged videos for continuous tracking
- **Resource Management**:
  - Memory-efficient processing
  - Temporary file management
  - Automatic cleanup
  
### Progress Monitoring
- Frames processed/total
- Processing speed (FPS)
- ETA calculations
- 10-second update intervals
- Segment progress tracking
- Cumulative progress across segments

## Installation

1. Clone the repository:

git clone https://github.com/mondweep/video_analysis.git


2. Install dependencies:

pip install -r requirements.txt

3. Create a `.env` file with your API keys:

GEMINI_API_KEY=your_key_here

## Usage

### YouTube Video Analysis

python youtube_object_detection.py

### Local Video Processing

python local_video_tracking.py

## Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- FFmpeg installed on system
- Internet connection for YouTube processing

## Output
- Processed videos in MP4 format
- AI-generated stories in text files
- Individual person tracking videos
- Progress logs and statistics

## License
MIT License

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

import google.generativeai as genai
from pathlib import Path
import os
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

load_dotenv()

def setup_gemini():
    """Initialize Gemini API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

def analyze_detections(all_detections):
    """Analyze YOLO detections to create a summary"""
    objects_detected = []
    
    # Process detections from all frames
    for frame_data in all_detections:
        if frame_data['boxes'] is not None:
            for box in frame_data['boxes']:
                class_id = int(box.cls[0])
                class_name = frame_data['names'][class_id]
                objects_detected.append(class_name)
    
    # Count occurrences of each object
    object_counts = Counter(objects_detected)
    return dict(object_counts)

def generate_video_story(detection_counts, scene_info, duration, video_url, start_time):
    """Generate a story and poem based on video content"""
    try:
        model = setup_gemini()
        
        # Ensure start_time is properly formatted
        if isinstance(start_time, (int, float)):
            start_minutes = int(start_time)
            start_seconds = 0
        else:
            start_minutes, start_seconds = map(int, start_time)
        
        # Create scene description
        scene_description = []
        for obj, count in detection_counts.items():
            scene_description.append(f"- {obj}: {count} occurrences")
        
        scene_description.append("\nScene classification:")
        for scene, score in scene_info.items():
            if score > 0.1:
                scene_description.append(f"- {scene}: {score:.2f} confidence")
        
        scene_text = "\n".join(scene_description)
        prompt = f"""
        Create a family-friendly story and poem about this scene:
        
        Video duration: {duration} seconds
        Time marker: {start_minutes}:{start_seconds:02d}
        
        Observed elements:
        {scene_text}
        
        Please write:
        1. A brief, G-rated story about what's happening (100 words)
        2. A simple, child-friendly poem (4 lines)
        
        Keep the tone light and appropriate for all ages.
        """
        
        # Updated safety settings according to the documentation
        safety_settings = [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        print(f"\nScene Text: {scene_text}")
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        # Check if response is blocked
        if not response.candidates:
            raise ValueError("Response was blocked by safety filters. Trying with more conservative prompt...")
        
        # Save and display the story
        output_dir = Path("output") / "stories"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"video_story_{timestamp}.txt"
        
        with open(output_file, 'w') as f:
            if hasattr(response, 'text'):
                f.write(response.text)
            else:
                f.write(str(response.candidates[0].content.parts[0].text))
        
        print(f"\nStory and poem generated and saved to: {output_file}")
        print("\nGenerated content:")
        print("-" * 50)
        if hasattr(response, 'text'):
            print(response.text)
        else:
            print(response.candidates[0].content.parts[0].text)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error generating story: {str(e)}")
        print(f"Error details: {e.__class__.__name__}")
        import traceback
        print(f"Error traceback: {traceback.format_exc()}")

def setup_scene_analyzer():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

def analyze_scene(frame, processor, model):
    inputs = processor(frame, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    scene_description = processor.decode(output[0], skip_special_tokens=True)
    return scene_description 
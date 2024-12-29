import google.generativeai as genai
from pathlib import Path
import os
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import openai

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

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

def generate_video_story(objects, scenes, actions, temporal_info, duration, video_url, start_time):
    """Generate a story from video analysis results"""
    try:
        # Process detection data
        object_counts = dict(Counter(objects))
        
        # Get most common elements
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_scenes = sorted(scenes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Setup Gemini
        model = setup_gemini()
        
        # Create prompts
        story_prompt = f"""
        Create an engaging 200-word story based on these elements from a video:
        - Main objects seen: {', '.join(f'{obj} ({count} times)' for obj, count in top_objects)}
        - Scene settings: {', '.join(f'{scene}' for scene, _ in top_scenes)}
        
        Requirements:
        1. Story should be exactly 200 words
        2. Incorporate the most frequently seen objects naturally
        3. Use the scene settings as the environment
        4. Make it creative and engaging
        """
        
        poem_prompt = f"""
        Create a short, vivid poem about this scene:
        - Key elements: {', '.join(f'{obj}' for obj, _ in top_objects[:5])}
        - Setting: {', '.join(f'{scene}' for scene, _ in top_scenes[:2])}
        
        Requirements:
        1. Keep it under 8 lines
        2. Make it evocative and atmospheric
        3. Include at least 3 of the key elements
        4. Reference the setting
        """
        
        # Generate content using Gemini
        print("\nGenerating story...")
        story_response = model.generate_content(story_prompt)
        
        print("Generating poem...")
        poem_response = model.generate_content(poem_prompt)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output/stories")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"video_story_{timestamp}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"Video Analysis Results\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Source: {video_url}\n")
            f.write(f"Time: {start_time[0]}:{start_time[1]}, Duration: {duration}s\n\n")
            
            # Write detected elements
            f.write("Scene Elements Detected:\n")
            f.write(f"{'='*50}\n\n")
            f.write("Objects:\n")
            f.write(f"{'-'*20}\n")
            f.write('\n'.join(f"- {obj} ({count} times)" for obj, count in top_objects))
            f.write("\n\nScenes:\n")
            f.write(f"{'-'*20}\n")
            f.write('\n'.join(f"- {scene}" for scene, _ in top_scenes))
            
            # Write generated story
            f.write("\n\nGenerated Story:\n")
            f.write(f"{'='*50}\n")
            f.write(story_response.text)
            
            # Write generated poem
            f.write("\n\nPoetic Interpretation:\n")
            f.write(f"{'='*50}\n")
            f.write(poem_response.text)
        
        print(f"\nGenerated content saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating story: {str(e)}")
        import traceback
        print(f"Error traceback: {traceback.format_exc()}")

def get_system_prompt(style):
    """Return appropriate system prompt based on style"""
    prompts = {
        'story': "You are a creative writer who turns video scenes into engaging narratives.",
        'poem': "You are a poet who creates vivid imagery from visual scenes.",
        'analysis': "You are an analytical observer who provides detailed scene descriptions."
    }
    return prompts.get(style, prompts['story'])

def create_story_prompt(elements):
    """Create detailed story prompt from scene elements"""
    return f"""
Create an engaging story based on this video scene:
- Main objects observed: {', '.join(elements['objects'])}
- Scene setting: {', '.join(elements['scenes'])}
- Actions occurring: {', '.join(elements['actions'])}
- Environmental elements: {', '.join(elements['environment'])}
- Weather/Atmosphere: {', '.join(elements['weather'])}
- Sequence of events: {', '.join(elements['sequence'])}

Make the story vivid and engaging, incorporating the natural elements, actions, and atmosphere.
"""

def create_poem_prompt(elements):
    """Create poetry prompt from scene elements"""
    return f"""
Create a descriptive poem capturing this scene:
- Visual elements: {', '.join(elements['objects'])}
- Setting: {', '.join(elements['scenes'])}
- Movement: {', '.join(elements['actions'])}
- Nature: {', '.join(elements['environment'])}
- Atmosphere: {', '.join(elements['weather'])}

Focus on creating vivid imagery and emotional resonance.
"""

def create_analysis_prompt(elements):
    """Create analytical description prompt"""
    return f"""
Provide a detailed analysis of this video scene:
- Primary elements: {', '.join(elements['objects'])}
- Setting classification: {', '.join(elements['scenes'])}
- Observed actions: {', '.join(elements['actions'])}
- Environmental context: {', '.join(elements['environment'])}
- Atmospheric conditions: {', '.join(elements['weather'])}
- Event progression: {', '.join(elements['sequence'])}

Focus on the relationships between elements and their significance.
"""

def save_generated_content(stories, elements, video_url, start_time, duration):
    """Save all generated content to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/stories")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"video_story_{timestamp}.txt"
    
    with open(output_path, 'w') as f:
        f.write(f"Video Analysis Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Source: {video_url}\n")
        f.write(f"Time: {start_time[0]}:{start_time[1]}, Duration: {duration}s\n\n")
        
        # Write scene elements
        f.write("Scene Elements Detected:\n")
        f.write(f"{'='*50}\n")
        for category, items in elements.items():
            f.write(f"\n{category.title()}:\n")
            f.write(f"{'-'*20}\n")
            f.write('\n'.join(f"- {item}" for item in items))
            f.write('\n')
        
        # Write generated content
        for style, content in stories.items():
            f.write(f"\n\n{style.title()}:\n")
            f.write(f"{'='*50}\n")
            f.write(content)
            f.write('\n')
        
    print(f"\nGenerated content saved to: {output_path}")

def setup_scene_analyzer():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model

def analyze_scene(frame, processor, model):
    inputs = processor(frame, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    scene_description = processor.decode(output[0], skip_special_tokens=True)
    return scene_description 
import boto3
import json
from datetime import datetime

def process_analysis_results(job_ids):
    """
    Process results from all Rekognition analysis jobs
    """
    rekognition = boto3.client('rekognition')
    
    try:
        results = {
            'timestamp': datetime.now().isoformat(),
            'guitar_analysis': {}
        }
        
        # 1. Process Label Detection
        label_response = rekognition.get_label_detection(JobId=job_ids['labelDetection'])
        results['guitar_analysis']['labels'] = [
            {
                'label': label['Label']['Name'],
                'confidence': label['Label']['Confidence'],
                'timestamp': label['Timestamp']
            }
            for label in label_response['Labels']
            if label['Label']['Confidence'] > 80
        ]
        
        # 2. Process Person Tracking
        person_response = rekognition.get_person_tracking(JobId=job_ids['personTracking'])
        results['guitar_analysis']['movements'] = [
            {
                'timestamp': person['Timestamp'],
                'position': person['Person']['BoundingBox'],
                'confidence': person['Person']['Confidence']
            }
            for person in person_response['Persons']
        ]
        
        # 3. Process Segments
        segment_response = rekognition.get_segment_detection(JobId=job_ids['segmentDetection'])
        results['guitar_analysis']['segments'] = [
            {
                'type': segment['Type'],
                'start_timestamp': segment['StartTimestampMillis'],
                'end_timestamp': segment['EndTimestampMillis'],
                'duration': segment['DurationMillis']
            }
            for segment in segment_response.get('Segments', [])
        ]
        
        # 4. Process Face Detection
        face_response = rekognition.get_face_detection(JobId=job_ids['faceDetection'])
        results['guitar_analysis']['face_analysis'] = [
            {
                'timestamp': face['Timestamp'],
                'pose': face['Face']['Pose'],
                'emotions': face['Face'].get('Emotions', [])
            }
            for face in face_response['Faces']
        ]
        
        # 5. Generate insights
        results['insights'] = generate_insights(results['guitar_analysis'])
        
        return results
        
    except Exception as e:
        print(f"Error processing results: {str(e)}")
        return None

def generate_insights(analysis):
    """
    Generate meaningful insights from the analysis
    """
    insights = {
        'practice_segments': [],
        'technique_observations': [],
        'engagement_levels': []
    }
    
    # Analyze practice segments
    for segment in analysis['segments']:
        duration_seconds = segment['duration'] / 1000
        if duration_seconds > 30:  # Significant practice segment
            insights['practice_segments'].append({
                'start_time': segment['start_timestamp'] / 1000,
                'duration': duration_seconds,
                'type': segment['type']
            })
    
    # Analyze hand movements and guitar positioning
    for movement in analysis['movements']:
        if movement['confidence'] > 90:
            insights['technique_observations'].append({
                'timestamp': movement['timestamp'],
                'position_details': movement['position']
            })
    
    # Analyze engagement through face detection
    for face_data in analysis['face_analysis']:
        if face_data['emotions']:
            primary_emotion = max(face_data['emotions'], key=lambda x: x['Confidence'])
            insights['engagement_levels'].append({
                'timestamp': face_data['timestamp'],
                'emotion': primary_emotion['Type'],
                'confidence': primary_emotion['Confidence']
            })
    
    return insights 
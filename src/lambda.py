import boto3
import json
import time

def lambda_handler(event, context):
    rekognition = boto3.client('rekognition', region_name='us-east-1')
    s3 = boto3.client('s3', region_name='us-east-1')
    
    # Get video details from S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    video = event['Records'][0]['s3']['object']['key']
    
    try:
        # Start both label detection and person tracking
        label_response = rekognition.start_label_detection(
            Video={'S3Object': {'Bucket': bucket, 'Name': video}},
            MinConfidence=80
        )
        
        person_response = rekognition.start_person_tracking(
            Video={'S3Object': {'Bucket': bucket, 'Name': video}}
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Video analysis started',
                'jobs': {
                    'labelDetection': label_response['JobId'],
                    'personTracking': person_response['JobId']
                }
            })
        }
        
    except Exception as e:
        print(f"Full error details: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error starting video analysis',
                'error': str(e)
            })
        }

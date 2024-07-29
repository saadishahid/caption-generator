from django.test import TestCase
from django.conf import settings
import boto3
from botocore.exceptions import ClientError
import os

class RekognitionTestCase(TestCase):
    def setUp(self):
        # This method is called before each test
        self.rekognition_client = boto3.client('rekognition',
                                               aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                               aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                               region_name=settings.AWS_REGION)

    def test_rekognition_connection(self):
        try:
            # Just testing if we can call a method without errors
            response = self.rekognition_client.list_collections()
            self.assertTrue('CollectionIds' in response)
            print("Successfully connected to AWS Rekognition")
        except ClientError as e:
            self.fail(f"Failed to connect to AWS Rekognition: {str(e)}")

    def test_rekognition_label_detection(self):
        # Path to a test image in your static files
        image_path = os.path.join(settings.BASE_DIR, 'api', 'tests', 'test_photo.png')
        
        try:
            with open(image_path, 'rb') as image_file:
                response = self.rekognition_client.detect_labels(Image={'Bytes': image_file.read()})
            
            self.assertTrue('Labels' in response)
            self.assertTrue(len(response['Labels']) > 0)
            print(f"Detected {len(response['Labels'])} labels in the test image")
            for label in response['Labels'][:10]:  # Print first 10 labels
                print(f"Label: {label['Name']}, Confidence: {label['Confidence']:.2f}%")
        except ClientError as e:
            self.fail(f"Failed to detect labels: {str(e)}")
        except FileNotFoundError:
            self.fail(f"Test image not found at {image_path}")
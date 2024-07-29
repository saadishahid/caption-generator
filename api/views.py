from django.core.files.base import ContentFile
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Image, Caption
from .serializers import ImageSerializer
from django.conf import settings
import boto3
from botocore.exceptions import ClientError
import random

class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.select_related('user').all()
    serializer_class = ImageSerializer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rekognition_client = boto3.client('rekognition',
                                               aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                               aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                               region_name=settings.AWS_REGION)

    @action(detail=False, methods=['post'])
    def upload_image(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image_bytes = image_file.read()
            labels, is_nsfw, nsfw_score = self.process_image_with_rekognition(image_bytes)
        except Exception as e:
            return Response({"error": f"Error processing image: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Create an Image instance with metadata
        image = Image(
            user=request.user,
            is_nsfw=is_nsfw,
            nsfw_score=nsfw_score,
            status='REJECTED' if is_nsfw else 'ACCEPTED'
        )

        if not is_nsfw:
            image.image.save(image_file.name, ContentFile(image_bytes), save=False)
        
        image.save()
        
        if is_nsfw:
            return Response({"error": "NSFW image detected and rejected"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Generate caption
        caption_text = self.generate_caption(labels)
        if caption_text:
            Caption.objects.create(image=image, text=caption_text)
        
        serializer = self.get_serializer(image)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def process_image_with_rekognition(self, image_bytes):
        try:
            # Detect moderation labels
            moderation_response = self.rekognition_client.detect_moderation_labels(Image={'Bytes': image_bytes})
            moderation_labels = moderation_response.get('ModerationLabels', [])
            is_nsfw = any(label['ParentName'] == 'Explicit Nudity' or label['ParentName'] == 'Violence' for label in moderation_labels)
            nsfw_score = max([label['Confidence'] for label in moderation_labels]) if moderation_labels else 0

            # Detect general labels
            label_response = self.rekognition_client.detect_labels(Image={'Bytes': image_bytes})
            labels = [label['Name'] for label in label_response['Labels']]
            
            return labels, is_nsfw, nsfw_score
        except ClientError as e:
            print(f"Error processing image with Rekognition: {e}")
            raise

    def generate_caption(self, labels):
        templates = [
            "This image showcases {}.",
            "A scene featuring {}.",
            "An intriguing composition of {}.",
            "A captivating image highlighting {}."
        ]
        
        main_subject = random.choice(labels) if labels else 'something interesting'
        return random.choice(templates).format(main_subject)

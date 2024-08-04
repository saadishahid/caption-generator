import logging
from django.core.files.base import ContentFile
from django.core.cache import cache
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Image, Caption
from .serializers import ImageSerializer
from django.conf import settings
import boto3
from botocore.exceptions import ClientError
from .caption_generator import CustomCaptionGenerator

logger = logging.getLogger(__name__)

class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.select_related('user').all()
    serializer_class = ImageSerializer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rekognition_client = boto3.client('rekognition',
                                               aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                               aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                               region_name=settings.AWS_REGION)
        self.caption_generator = CustomCaptionGenerator()

    @action(detail=False, methods=['post'], url_path='upload_image', url_name='upload_image')
    def upload_image(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image_bytes = image_file.read()
            labels, is_nsfw, nsfw_score = self.process_image_with_rekognition(image_bytes)
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return Response({"error": f"Error processing image: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

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

        try:
            caption_text = self.caption_generator.generate_improved_caption(labels)
            caption = Caption.objects.create(image=image, text=caption_text)

            # Train the model with the new image's labels and generated caption
            self.caption_generator.update_model(labels, caption_text)

        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return Response({"error": f"Error generating caption: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        serializer = self.get_serializer(image)
        image_data = serializer.data
        image_data['caption'] = {'text': caption.text}  # Add the caption to the response data

        return Response(image_data, status=status.HTTP_201_CREATED)

    def process_image_with_rekognition(self, image_bytes):
        cache_key = f"rekognition_{hash(image_bytes)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        try:
            moderation_response = self.rekognition_client.detect_moderation_labels(Image={'Bytes': image_bytes})
            moderation_labels = moderation_response.get('ModerationLabels', [])
            is_nsfw = any(label['ParentName'] in ['Explicit Nudity', 'Violence'] for label in moderation_labels)
            nsfw_score = max([label['Confidence'] for label in moderation_labels]) if moderation_labels else 0

            label_response = self.rekognition_client.detect_labels(Image={'Bytes': image_bytes})
            labels = [label['Name'] for label in label_response['Labels']]
            
            result = (labels, is_nsfw, nsfw_score)
            cache.set(cache_key, result, timeout=3600)  # Cache for 1 hour
            return result
        except ClientError as e:
            logger.error(f"Error processing image with Rekognition: {e}")
            raise

    @action(detail=True, methods=['post'], url_path='update_caption', url_name='update_caption')
    def update_caption(self, request, pk=None):
        image = self.get_object()
        new_caption = request.data.get('caption')
        if not new_caption:
            return Response({"error": "No caption provided"}, status=400)

        caption = Caption.objects.get(image=image)
        caption.text = new_caption
        caption.save()

        # Update the model with the new caption
        self.caption_generator.update_model(image.labels, new_caption)

        return Response({"success": "Caption updated successfully"})

    @action(detail=False, methods=['post'], url_path='train', url_name='train')
    def train(self, request):
        labels_list = request.data.get('labels_list', [])
        captions = request.data.get('captions', [])
        
        if not labels_list or not captions:
            return Response({"error": "Labels and captions must be provided"}, status=status.HTTP_400_BAD_REQUEST)

        self.caption_generator.explicit_train(labels_list, captions)
        
        return Response({"success": "Model trained successfully"})

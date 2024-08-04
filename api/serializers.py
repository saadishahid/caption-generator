from rest_framework import serializers
from .models import Image, Caption

class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = ['id', 'image', 'uploaded_at', 'is_nsfw', 'nsfw_score', 'labels', 'status', 'user']  # Ensure all fields are included
        read_only_fields = ['is_nsfw', 'nsfw_score', 'uploaded_at', 'labels', 'status', 'user']

class CaptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Caption
        fields = ['id', 'image', 'text', 'generated_at']
        read_only_fields = ['generated_at']

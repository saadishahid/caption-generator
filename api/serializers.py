from rest_framework import serializers
from .models import Image, Caption

class ImageSerializer(serializers.ModelSerializer):
       class Meta:
           model = Image
           fields = ['id', 'image', 'uploaded_at', 'is_nsfw', 'nsfw_score']
           read_only_fields = ['is_nsfw', 'nsfw_score']

class CaptionSerializer(serializers.ModelSerializer):
       class Meta:
           model = Caption
           fields = ['id', 'image', 'text', 'generated_at']
   
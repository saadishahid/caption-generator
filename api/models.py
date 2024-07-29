from django.db import models
from django.contrib.auth.models import User


class Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_nsfw = models.BooleanField(default=False)
    nsfw_score = models.FloatField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('ACCEPTED', 'Accepted'),
        ('REJECTED', 'Rejected'),
    ], default='ACCEPTED')
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'uploaded_at']),
        ]

class Caption(models.Model):
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    text = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)
   
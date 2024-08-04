from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class Image(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/')
    is_nsfw = models.BooleanField(default=False)
    nsfw_score = models.FloatField(default=0)
    status = models.CharField(max_length=10, default='PENDING')
    labels = models.JSONField(default=list)
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Use auto_now_add

    def __str__(self):
        return f"Image {self.id} by {self.user.username}"

class Caption(models.Model):
    image = models.OneToOneField(Image, on_delete=models.CASCADE)
    text = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)  # Use auto_now_add

    def __str__(self):
        return f"Caption for Image {self.image.id}"

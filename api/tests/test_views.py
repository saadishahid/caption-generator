import unittest
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient, APITestCase
from rest_framework import status
from django.contrib.auth.models import User
from api.models import Image, Caption
from api.views import ImageViewSet

class ImageUploadTestCase(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)

    def test_upload_image(self):
        url = reverse('image-upload_image')  # Check the resolved URL
        print(f"Resolved URL for upload_image: {url}")
        with open('api/tests/test_image.jpg', 'rb') as image:
            response = self.client.post(url, {'image': image}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('id', response.data)
        self.assertEqual(Image.objects.count(), 1)
        self.assertEqual(Caption.objects.count(), 1)

    def test_upload_nsfw_image(self):
        url = reverse('image-upload_image')  # Check the resolved URL
        print(f"Resolved URL for upload_image: {url}")
        with open('api/tests/test_image.jpg', 'rb') as image:
            # Mock Rekognition to detect NSFW content
            ImageViewSet.process_image_with_rekognition = lambda self, image_bytes: ([], True, 95.0)
            response = self.client.post(url, {'image': image}, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(Image.objects.count(), 0)
        self.assertEqual(Caption.objects.count(), 0)

    def test_update_caption(self):
        upload_url = reverse('image-upload_image')  # Check the resolved URL for upload_image
        print(f"Resolved URL for upload_image: {upload_url}")
        with open('api/tests/test_image.jpg', 'rb') as image:
            response = self.client.post(upload_url, {'image': image}, format='multipart')
        image_id = response.data['id']
        update_url = reverse('image-update_caption', kwargs={'pk': image_id})  # Check the resolved URL for update_caption
        print(f"Resolved URL for update_caption: {update_url}")
        new_caption = "A beautiful garden."
        response = self.client.post(update_url, {'caption': new_caption}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        caption = Caption.objects.get(image_id=image_id)
        self.assertEqual(caption.text, new_caption)

if __name__ == '__main__':
    unittest.main()

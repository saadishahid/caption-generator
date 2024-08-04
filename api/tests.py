from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from unittest.mock import patch, MagicMock
from .models import Image, Caption
from .views import ImageViewSet
from .caption_generator import CustomCaptionGenerator
import numpy as np

class CustomCaptionGeneratorTestCase(TestCase):
    def setUp(self):
        self.generator = CustomCaptionGenerator()

    def test_generate_caption_no_labels(self):
        caption = self.generator.generate_caption([])
        self.assertEqual(caption, "An interesting image.")

    def test_generate_caption_with_labels(self):
        labels = ["dog", "cat", "house"]
        caption = self.generator.generate_caption(labels)
        self.assertIn("dog, cat, house", caption)

    def test_train_and_generate_improved_caption(self):
        labels_list = [["dog", "cat"], ["house", "tree"]]
        captions = ["A dog and a cat", "A house near a tree"]
        self.generator.train(labels_list, captions)

        new_labels = ["dog", "house"]
        improved_caption = self.generator.generate_improved_caption(new_labels)
        self.assertIn(improved_caption, captions)

    @patch('numpy.random.choice')
    def test_generate_improved_caption_probability(self, mock_choice):
        labels_list = [["dog", "cat"], ["house", "tree"]]
        captions = ["A dog and a cat", "A house near a tree"]
        self.generator.train(labels_list, captions)

        new_labels = ["dog", "house"]
        mock_choice.return_value = 0  # Always choose the first caption

        improved_caption = self.generator.generate_improved_caption(new_labels)
        self.assertEqual(improved_caption, "A dog and a cat")

        mock_choice.assert_called_once()

    def test_update_model(self):
        initial_labels = ["dog", "cat"]
        initial_caption = "A dog and a cat"
        self.generator.update_model(initial_labels, initial_caption)

        new_labels = ["dog", "house"]
        new_caption = "A dog in front of a house"
        self.generator.update_model(new_labels, new_caption)

        improved_caption = self.generator.generate_improved_caption(new_labels)
        self.assertIn(improved_caption, [initial_caption, new_caption])

class ImageViewSetTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.client.force_authenticate(user=self.user)
        self.viewset = ImageViewSet()
        self.viewset.caption_generator = MagicMock()

    @patch('api.views.ImageViewSet.process_image_with_rekognition')
    def test_upload_image_success(self, mock_process_image):
        mock_process_image.return_value = (['person', 'dog'], False, 0.1)
        self.viewset.caption_generator.generate_improved_caption.return_value = "A person with a dog"

        image_content = b'fake image content'
        image = SimpleUploadedFile("test_image.jpg", image_content, content_type="image/jpeg")

        response = self.client.post('/api/images/upload_image/', {'image': image}, format='multipart')

        self.assertEqual(response.status_code, 201)
        self.assertIn('id', response.data)
        self.assertIn('image', response.data)
        self.assertFalse(response.data['is_nsfw'])
        self.assertEqual(response.data['nsfw_score'], 0.1)

        image = Image.objects.get(id=response.data['id'])
        self.assertEqual(image.user, self.user)
        self.assertEqual(image.status, 'ACCEPTED')

        caption = Caption.objects.get(image=image)
        self.assertEqual(caption.text, "A person with a dog")

    @patch('api.views.ImageViewSet.process_image_with_rekognition')
    def test_upload_nsfw_image(self, mock_process_image):
        mock_process_image.return_value = (['explicit content'], True, 0.9)

        image_content = b'fake nsfw image content'
        image = SimpleUploadedFile("nsfw_image.jpg", image_content, content_type="image/jpeg")

        response = self.client.post('/api/images/upload_image/', {'image': image}, format='multipart')

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
        self.assertEqual(response.data['error'], "NSFW image detected and rejected")

        self.assertEqual(Image.objects.count(), 1)
        image = Image.objects.first()
        self.assertEqual(image.status, 'REJECTED')
        self.assertTrue(image.is_nsfw)
        self.assertEqual(image.nsfw_score, 0.9)

    def test_upload_image_no_file(self):
        response = self.client.post('/api/images/upload_image/', {}, format='multipart')

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
        self.assertEqual(response.data['error'], "No image provided")

    @patch('api.views.ImageViewSet.process_image_with_rekognition')
    def test_update_caption(self, mock_process_image):
        # First, upload an image
        mock_process_image.return_value = (['person', 'dog'], False, 0.1)
        self.viewset.caption_generator.generate_improved_caption.return_value = "A person with a dog"

        image_content = b'fake image content'
        image = SimpleUploadedFile("test_image.jpg", image_content, content_type="image/jpeg")

        response = self.client.post('/api/images/upload_image/', {'image': image}, format='multipart')
        image_id = response.data['id']

        # Now, update the caption
        new_caption = "A man walking his pet dog"
        response = self.client.post(f'/api/images/{image_id}/update_caption/', {'caption': new_caption})

        self.assertEqual(response.status_code, 200)
        self.assertIn('success', response.data)

        caption = Caption.objects.get(image__id=image_id)
        self.assertEqual(caption.text, new_caption)

        self.viewset.caption_generator.update_model.assert_called_once_with(['person', 'dog'], new_caption)

    @patch('api.views.ImageViewSet.get_object')
    def test_update_caption_no_caption_provided(self, mock_get_object):
        mock_image = MagicMock()
        mock_get_object.return_value = mock_image

        response = self.client.post('/api/images/1/update_caption/', {})

        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.data)
        self.assertEqual(response.data['error'], "No caption provided")

class ImageViewSetIntegrationTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='12345')
        self.client.force_authenticate(user=self.user)

    @patch('boto3.client')
    def test_full_upload_and_caption_flow(self, mock_boto3_client):
        # Mock AWS Rekognition response
        mock_rekognition = MagicMock()
        mock_rekognition.detect_moderation_labels.return_value = {
            'ModerationLabels': []
        }
        mock_rekognition.detect_labels.return_value = {
            'Labels': [{'Name': 'Dog'}, {'Name': 'Park'}]
        }
        mock_boto3_client.return_value = mock_rekognition

        # Upload image
        image_content = b'fake image content'
        image = SimpleUploadedFile("test_image.jpg", image_content, content_type="image/jpeg")
        response = self.client.post('/api/images/upload_image/', {'image': image}, format='multipart')

        self.assertEqual(response.status_code, 201)
        image_id = response.data['id']

        # Verify image and caption were created
        image = Image.objects.get(id=image_id)
        self.assertEqual(image.user, self.user)
        self.assertEqual(image.status, 'ACCEPTED')

        caption = Caption.objects.get(image=image)
        self.assertIn("Dog", caption.text)
        self.assertIn("Park", caption.text)

        # Update caption
        new_caption = "A happy dog playing in the park"
        response = self.client.post(f'/api/images/{image_id}/update_caption/', {'caption': new_caption})

        self.assertEqual(response.status_code, 200)

        # Verify caption was updated
        caption.refresh_from_db()
        self.assertEqual(caption.text, new_caption)

        # Upload another image to test improved captioning
        image2 = SimpleUploadedFile("test_image2.jpg", b'another fake image content', content_type="image/jpeg")
        response = self.client.post('/api/images/upload_image/', {'image': image2}, format='multipart')

        self.assertEqual(response.status_code, 201)
        image2_id = response.data['id']

        caption2 = Caption.objects.get(image__id=image2_id)
        self.assertIn("Dog", caption2.text)
        self.assertIn("Park", caption2.text)
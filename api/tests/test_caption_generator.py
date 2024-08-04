from django.test import TestCase
from api.caption_generator import CustomCaptionGenerator
import numpy as np
from unittest.mock import patch

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

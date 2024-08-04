import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class CustomCaptionGenerator:
    def __init__(self):
        """Initializes the caption generator with a TF-IDF vectorizer and a Naive Bayes classifier."""
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.templates = [
            "This image shows {}.",
            "A scene featuring {}.",
            "An image containing {}.",
            "A view of {}."
        ]
        self.trained = False

    def generate_caption(self, labels):
        """
        Generates a basic caption using the provided labels.
        
        :param labels: List of labels describing the image.
        :return: Generated caption string.
        """
        if not labels:
            return "An interesting image."
        
        main_subjects = ', '.join(labels[:3])  # Use top 3 labels
        return random.choice(self.templates).format(main_subjects)

    def train(self, labels_list, captions):
        """
        Trains the model using provided labels and corresponding captions.
        
        :param labels_list: List of lists of labels.
        :param captions: List of captions corresponding to the labels.
        """
        label_strings = [' '.join(labels) for labels in labels_list]
        X = self.vectorizer.fit_transform(label_strings)
        self.classifier.fit(X, captions)
        self.trained = True

    def generate_improved_caption(self, labels):
        """
        Generates an improved caption using the trained model.
        
        :param labels: List of labels describing the image.
        :return: Improved caption string.
        """
        if not self.trained:
            return self.generate_caption(labels)
        
        label_string = ' '.join(labels)
        X = self.vectorizer.transform([label_string])
        
        # Get probability distribution over captions
        probs = self.classifier.predict_proba(X)[0]
        
        # Sample a caption based on the probability distribution
        caption_idx = np.random.choice(len(self.classifier.classes_), p=probs)
        return self.classifier.classes_[caption_idx]

    def update_model(self, labels, caption):
        """
        Updates the trained model with new data.
        
        :param labels: List of labels describing the image.
        :param caption: The new caption to be added to the model.
        """
        if not self.trained:
            self.train([labels], [caption])
        else:
            label_string = ' '.join(labels)
            X = self.vectorizer.transform([label_string])
            self.classifier.partial_fit(X, [caption], classes=self.classifier.classes_)

    def explicit_train(self, labels_list, captions):
        """
        Explicitly trains the model using provided labels and corresponding captions.
        
        :param labels_list: List of lists of labels.
        :param captions: List of captions corresponding to the labels.
        """
        self.train(labels_list, captions)

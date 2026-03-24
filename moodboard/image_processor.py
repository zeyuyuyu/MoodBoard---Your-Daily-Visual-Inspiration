import os
import tensorflow as tf
import numpy as np
import json

class ImageProcessor:
    def __init__(self, model_path='moodboard/image_classification_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['nature', 'cityscape', 'portrait', 'abstract', 'food']

    def classify_image(self, image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return predicted_class, confidence

    def tag_images(self, directory):
        tags = {}
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(directory, filename)
                predicted_class, confidence = self.classify_image(image_path)
                tags[filename] = {'class': predicted_class, 'confidence': float(confidence)}
        return tags

if __name__ == '__main__':
    processor = ImageProcessor()
    tags = processor.tag_images('moodboard/images')
    with open('moodboard/image_tags.json', 'w') as f:
        json.dump(tags, f, indent=4)
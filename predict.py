import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model_path = 'model/cat_dog_classifier.h5'
if not os.path.exists(model_path):
    print("❌ Model not found! Train the model using main.py first.")
    exit()

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Dataset path
dataset_path = 'dataset'
categories = ['cats', 'dogs']

# Ask how many images to predict
num_images = int(input("Enter how many images you want to predict: "))

for i in range(num_images):
    # Pick a random image
    category = random.choice(categories)
    folder_path = os.path.join(dataset_path, category)
    random_image = random.choice(os.listdir(folder_path))
    img_path = os.path.join(folder_path, random_image)

    print(f"\n🎯 Selected image {i+1}: {img_path}")

    # Preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]
    label = "🐱 Cat" if prediction < 0.5 else "🐶 Dog"
    print(f"Prediction: {label}")

    # Show the image
    plt.imshow(image.load_img(img_path))
    plt.title(f"Prediction: {label}")
    plt.axis('off')
    plt.show()
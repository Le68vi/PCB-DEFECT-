import tensorflow as tf
import numpy as np
import os
import cv2
import random

# Path to your train folder
train_dir = r"C:\Users\Dell\Desktop\man\zaalima\PR1\dataset\train"

def representative_data_gen():
    image_paths = []

    # Collect images from both classes
    for class_name in ["pass", "defect"]:
        class_folder = os.path.join(train_dir, class_name)
        for filename in os.listdir(class_folder):
            image_paths.append(os.path.join(class_folder, filename))

    # Randomly pick 200 images for calibration
    sample_paths = random.sample(image_paths, min(200, len(image_paths)))

    for path in sample_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

# Convert to INT8 TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("pcb_resnet_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

with open("pcb_resnet_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ INT8 TFLite model created successfully.")
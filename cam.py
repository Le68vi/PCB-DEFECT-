import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load model
model = load_model("pcb_resnet50_model.h5")

# ==============================
# Image Preprocessing Function
# ==============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# ==============================
# Grad-CAM Function
# ==============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# ==============================
# Overlay Heatmap
# ==============================
def overlay_heatmap(img_path, heatmap, alpha=0.4):

    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return superimposed_img


# ==============================
# PROCESS VALIDATION FOLDER
# ==============================

input_folder = "dataset/val"   # CHANGE THIS IF NEEDED
output_folder = "gradcam_results"

os.makedirs(output_folder, exist_ok=True)

# Loop through defect and pass folders
for class_name in os.listdir(input_folder):

    class_path = os.path.join(input_folder, class_name)

    if not os.path.isdir(class_path):
        continue

    # Create same class folder in output
    output_class_path = os.path.join(output_folder, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        try:
            img_array = preprocess_image(img_path)

            # For ResNet50 last conv layer:
            heatmap = make_gradcam_heatmap(
                img_array,
                model,
                "conv5_block3_out"   # last conv layer of ResNet50
            )

            result = overlay_heatmap(img_path, heatmap)

            cv2.imwrite(os.path.join(output_class_path, img_name), result)

            print(f"Processed: {img_name}")

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

print("Grad-CAM generation completed!")
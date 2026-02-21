import tensorflow as tf

# Load your trained model (.keras or .h5)
model = tf.keras.models.load_model("pcbresnet50_model.keras")
# OR
# model = tf.keras.models.load_model("pcb_resnet50_model.h5")

# Save as TensorFlow SavedModel (folder format)
model.save("pcb_resnet_savedmodel", save_format="tf")

print("SavedModel created successfully.")
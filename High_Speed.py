import tensorflow as tf
import numpy as np
import cv2
import time

# ==============================
# PERFORMANCE SETTINGS
# ==============================
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

# ==============================
# LOAD TFLITE MODEL
# ==============================
interpreter = tf.lite.Interpreter(
    model_path="pcb_resnet_int8.tflite",
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==============================
# CLASS LABELS
# Adjust based on your training
# ==============================
class_names = ["Defect", "Pass"]   # index 0 = Defect, 1 = Pass

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# INFERENCE FUNCTION
# ==============================
def infer(frame):
    input_data = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()
    interpreter.invoke()
    end = time.time()

    output = interpreter.get_tensor(output_details[0]['index'])

    inference_time = end - start
    fps = 1 / inference_time if inference_time > 0 else 0

    # Get predicted class
    predicted_index = np.argmax(output)
    confidence = float(np.max(output))

    label = class_names[predicted_index]

    return label, confidence, inference_time, fps


# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows optimization

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Starting PCB Inspection System...")
print("Press 'q' to quit.\n")

# ==============================
# REAL-TIME LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    label, confidence, inf_time, fps = infer(frame)

    # Print result in terminal
    print(f"Result: {label} | Confidence: {confidence:.2f} | FPS: {fps:.2f}")

    # Choose color based on result
    if label == "Pass":
        color = (0, 255, 0)  # Green
    else:
        color = (0, 0, 255)  # Red

    # Display result on frame
    cv2.putText(frame,
                f"{label} ({confidence:.2f})",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    # Show live feed
    cv2.imshow("PCB Inspection - High Speed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
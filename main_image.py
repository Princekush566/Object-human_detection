

import numpy as np
import cv2

# Paths
image_path = 'res/image/img1.jpg'
video_path = ""
prototxt_path = 'Models/MobileNetSSD_deploy.prototxt'
model_path = 'Models/MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

# Class labels and colors
classes = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "male", "female"]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    if not ret:
        break

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0, 0, i, 2]

        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(image, prediction_text, (upper_left_x,
                        upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    # Display the output
    cv2.imshow("Detected Object", image)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()


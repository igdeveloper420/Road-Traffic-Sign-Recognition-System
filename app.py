import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("traffic_sign_model.h5")
classes = ['Left', 'Right', 'Stop','20KMPH','50KMPH','100KMPH','NO Entry']  # Added '20KMPH' class
img_size = 64  # Match training resolution

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_size, img_size))
    img_input = img / 255.0
    img_input = img_input.reshape(1, img_size, img_size, 3)

    prediction = model.predict(img_input)
    class_id = np.argmax(prediction)
    label = classes[class_id]
    confidence = prediction[0][class_id] * 100

    print(f"Prediction Probabilities: {prediction}")
    print(f"Detected: {label} ({confidence:.2f}%)")

    cv2.putText(frame, f"{label} ({confidence:.1f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

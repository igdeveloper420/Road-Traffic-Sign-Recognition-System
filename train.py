import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

data = []
labels = []
classes = ['Left', 'Right', 'Stop', '20KMPH','50KMPH','100KMPH','NO ENTRY','ONEWAY','RIGHT TURN PROHIBITED','LEFT TURN PROHIBITED','OVERTAKING PROHIBITED',
           'HAND CART PROHIBITED','CYCLE PROHIBITED','TRUCK PROHIBITED','ROUND ABOUT','SPEED BREAKER',
           'COMPULSORY KEEP LEFT','DANGEROUS DIP','PEDESTRIAN PROHIBITED'] 
img_size = 64  # Increased image size

# Load images
for i, label in enumerate(classes):
    folder = os.path.join("dataset", label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(i)

data = np.array(data) / 255.0
labels = to_categorical(labels, num_classes=19)  # Changed to 19 for 19 classes

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Improved CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),  # Helps prevent overfitting
    Dense(19, activation='softmax')  # Changed to 19 for 19 classes MEANS OUTPUT FOR 19 CLASSES
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))  # More epochs

model.save("traffic_sign_model.h5")
print("✅ Model trained and saved as traffic_sign_model.h5")

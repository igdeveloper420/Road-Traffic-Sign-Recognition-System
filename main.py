from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model and define classes
model = load_model("traffic_sign_model.h5")
classes = ['Left', 'Right', 'Stop', '20KMPH','50KMPH','100KMPH','NO ENTRY','ONEWAY','RIGHT TURN PROHIBITED','LEFT TURN PROHIBITED','OVERTAKING PROHIBITED',
           'HAND CART PROHIBITED','CYCLE PROHIBITED','TRUCK PROHIBITED','ROUND ABOUT','SPEED BREAKER',
           'COMPULSORY KEEP LEFT','DANGEROUS DIP','PEDESTRIAN PROHIBITED'] 
img_size = 64

def predict_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError("Failed to read image")
    resized = cv2.resize(image, (img_size, img_size))
    input_data = resized / 255.0
    input_data = input_data.reshape(1, img_size, img_size, 3)

    prediction = model.predict(input_data)
    class_id = np.argmax(prediction)
    label = classes[class_id]
    confidence = float(prediction[0][class_id]) * 100

    cv2.putText(image, f"{label} ({confidence:.1f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    result_filename = "result.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, image)

    return label, confidence, result_filename

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            label, confidence, result_image = predict_image(filepath)
            return render_template('index.html',
                                   label=label,
                                   confidence=round(confidence, 2),
                                   image_file=result_image)
    return render_template('index.html')

@app.route('/webcam', methods=['POST'])
def webcam_capture():
    try:
        data_url = request.form['webcam_img']
        encoded_data = data_url.split(',')[1]
        binary_data = base64.b64decode(encoded_data)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam.jpg')

        with open(filepath, 'wb') as f:
            f.write(binary_data)

        label, confidence, result_img = predict_image(filepath)

        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'image_path': result_img
        })

    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to read video")

    label_counts = {}
    frame_count = 0
    selected_frame = None
    selected_label = ""
    selected_confidence = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (img_size, img_size))
        input_data = resized / 255.0
        input_data = input_data.reshape(1, img_size, img_size, 3)

        prediction = model.predict(input_data)
        class_id = np.argmax(prediction)
        label = classes[class_id]
        confidence = float(prediction[0][class_id]) * 100

        if frame_count == 10:  # Pick the 10th frame as preview
            selected_frame = frame.copy()
            selected_label = label
            selected_confidence = confidence

        label_counts[label] = label_counts.get(label, 0) + 1
        frame_count += 1

    cap.release()

    # Save preview image
    if selected_frame is not None:
        cv2.putText(selected_frame, f"{selected_label} ({selected_confidence:.1f}%)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        preview_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video_result.jpg')
        cv2.imwrite(preview_path, selected_frame)
    else:
        raise ValueError("No valid frame found in video")

    most_common_label = max(label_counts, key=label_counts.get)
    avg_confidence = (label_counts[most_common_label] / frame_count) * 100

    return most_common_label, avg_confidence, 'video_result.jpg'


@app.route('/video_upload', methods=['POST'])
def video_upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected video'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        label, confidence, result_video = predict_video(filepath)
        return render_template('index.html',
                       label=label,
                       confidence=round(confidence, 2),
                       image_file=result_video)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

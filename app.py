from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import requests
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import base64
from io import BytesIO

app = Flask(__name__)
dataset_path_train = r'C:\Users\meghb\Desktop\ColourDetection\images\train\\'
dataset_path_val = r'C:\Users\meghb\Desktop\ColourDetection\images\validation\\'
def load_data(path):
    data = []
    labels = []
    
    for color in ["blue", "green", "red"]:
        color_path = path + color + "/"
        _, _, files = next(os.walk(color_path))
        file_count = len(files)
        for i in range(1, file_count):
            img = cv2.imread(color_path + f"1 ({i}).png")
            img = cv2.resize(img, (50, 50))  # Resize for consistency
            
            data.append(img.flatten())            
            labels.append(color)
    print("Number of images:", len(data))
    print("Number of labels:", len(labels))
    return np.array(data), np.array(labels)

data_train, label_train = load_data(dataset_path_train)
data_val, label_val = load_data(dataset_path_val)
print(data_train)
X_train, X_test, y_train, y_test = train_test_split(data_train, label_train, test_size=0.1, random_state=42)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred)
print("Training Accuracy:", train_accuracy)
print("model trained")
def color_recognition(frame):
    color_sample = np.array([[frame[240, 320]]], dtype=np.uint8)
    color_name = knn_model.predict(color_sample.flatten().reshape(1, -1))[0]
    cv2.putText(frame, f"Color: {color_name}", (320, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        color_recognition(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/predict_color', methods=['POST'])
def predict_color():
    print("here")
    # Read the image sent from the client
    print(request.form)
    image = request.form['imageSrc']
    print(image)
    #print('here?')
    #print("Image URL:", image)
    imgSrc1 = image[1:]
    imgSrc = imgSrc1.replace("/", "\\")
    #src = r"static\ui_images\red1.png"
    with open(imgSrc, 'rb') as file:
        image_data = file.read()

    #response = requests.get(image)
    #image_data = response.content
    nparr = np.frombuffer(image_data, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image to match the training data
    img_resized = cv2.resize(img, (50, 50))

    # Flatten the image data
    img_flattened = img_resized.flatten()
    print(knn_model)
    # Predict the color
    color_name = knn_model.predict([img_flattened])[0]

    # Return the predicted color as JSON
    print(color_name)
    return jsonify({'color': color_name})
        
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



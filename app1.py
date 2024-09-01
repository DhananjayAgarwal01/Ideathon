from flask import Flask, render_template, Response
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import time

app = Flask(__name__)

# Load emotion labels
emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}

# Load model
json_file = open('emotion_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Manually register the Sequential class if needed
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential

get_custom_objects().update({"Sequential": Sequential})

classifier = model_from_json(loaded_model_json)
classifier.load_weights("emotion_model1.h5")

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def generate_frames():
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    duration = 5  # seconds
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - start_time

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)

                label_position = (x, y)
                cv2.putText(frame, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame_count += 1
        if elapsed_time >= duration or frame_count >= 30:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, jsonify, request, send_file
import io
from process_image.proc import image_proc_bp
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
# Initialize OpenCV and NumPy
@app.route('/detect-face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return {"status": "error", "message": "No image uploaded"}, 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    _, buffer = cv2.imencode('.jpg', img)
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')


CORS(app)

app.register_blueprint(image_proc_bp, url_prefix='/api')
@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'wrong endpoint, hit the /process-image endpoint with a post request'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



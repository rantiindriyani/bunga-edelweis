from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
from PIL import Image
import base64
import yaml
from io import BytesIO
import os
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# ================================
# LOAD DATA
# ================================
with open('D:/edelwiss/data.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)

class_names = yaml_data.get('names', [])
class_info = {}

# ================================
# LOAD MODEL
# ================================
detector = YOLO('D:/edelwiss/best.pt')


# ================================
# ROUTES
# ================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection')
def detection_page():
    return render_template('detection.html')


@app.route('/recommendation')
def recommendation_page():
    return render_template('recommendation.html')


@app.route('/upload-detection')
def upload_detection():
    return render_template('upload-detection.html')


@app.route('/webcam')
def webcam_detection():
    return render_template('webcam.html')


# ================================
# WEBCAM STREAM
# ================================
last_detection = []


def gen_frames():
    global last_detection
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = detector.predict(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    class_id = int(boxes.cls[i].item())
                    class_name = class_names[class_id]

                    detections.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class_name': class_name,
                        'components': [],
                        'benefits': [],
                        'recipes': {}
                    })

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        last_detection = detections

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webcam_data')
def webcam_data():
    return jsonify({
        'status': 'success',
        'bounding_boxes': last_detection
    })


# ================================
# DETEKSI VIA CAMERA
# ================================
@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        image_str = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_str)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        results = detector.predict(image)
        bounding_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    class_id = int(boxes.cls[i].item())
                    class_name = class_names[class_id]

                    bounding_boxes.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class_name': class_name,
                        'components': [],
                        'benefits': [],
                        'recipes': {}
                    })

        return jsonify({
            'status': 'success',
            'bounding_boxes': bounding_boxes
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ================================
# RECOMMENDATION
# ================================
@app.route("/api/recommend")
def recommend():
    query = request.args.get("q", "").strip().lower()

    data = [
        {
            "leaf_name": "Mekar",
            "components": ["Kelopak terbuka"],
            "benefits": ["Kondisi stabil"],
            "gambar": "static/assets/Mekar-1-_brightness_0-8_jpg.rf.557c69dfae7267960053d767175cce68.jpg",      
            "recipes": {}
        },
        {
            "leaf_name": "Penyemaian",
            "components": ["Benih"],
            "benefits": ["Tahap awal pertumbuhan"],
            "gambar": "static/assets/Penyemaian002_rotate_15_jpg.rf.f2002a1b887cb03baf24311c52f6ec7c.jpg",  
            "recipes": {}
        },
        {
            "leaf_name": "Sangat Mekar",
            "components": ["Kelopak penuh"],
            "benefits": ["Kondisi terbaik"],
            "gambar": "static/assets/Sangat_Mekar-15-_saturation_0-8_jpg.rf.09ea13cce96b4bef217951beb0804b09.jpg", 
            "recipes": {}
        }
    ]

    # Kalau query kosong, tampilkan semua
    if not query:
        return jsonify({"status": "success", "results": data})

    # Exact match saja
    results = [
        item for item in data
        if item["leaf_name"].lower() == query
    ]

    # Fallback: startswith kalau exact tidak ketemu
    if not results:
        results = [
            item for item in data
            if item["leaf_name"].lower().startswith(query)
        ]

    # Fallback: contains kalau masih kosong
    if not results:
        results = [
            item for item in data
            if query in item["leaf_name"].lower()
        ]

    return jsonify({
        "status": "success",
        "results": results
    })


# ================================
# DETEKSI VIA UPLOAD
# ================================
@app.route('/upload-detection', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        file = request.files['file']
        image = Image.open(file.stream).convert("RGB")

        results = detector.predict(image)
        bounding_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = [int(x) for x in box]
                    class_id = int(boxes.cls[i].item())
                    class_name = class_names[class_id]

                    bounding_boxes.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'class_name': class_name,
                        'components': [],
                        'benefits': [],
                        'recipes': {}
                    })

        return jsonify({
            'status': 'success',
            'bounding_boxes': bounding_boxes
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ================================
# RUN APP
# ================================
if __name__ == "__main__":
    app.run(debug=True)
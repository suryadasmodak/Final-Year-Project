from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
import numpy as np
import tenseal as ts
from datetime import datetime
import os
import threading
import time

from facenet_feature import extract_feature_vector
from cancelable_transform import generate_transform_key, cancelable_transform
from encrypt_store import (
    load_context, create_ckks_context,
    save_context, encrypt_template, get_database
)
from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD = 0.9
REAL_COUNTER_REQUIRED = 5
CAMERA_INDEX = 1
BASE_DIR = r"c:\Users\SURYA DAS MODAK\OneDrive\Desktop\FY project\Final-Year-Project"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "l_version_1_300.pt")
CONTEXT_PATH = os.path.join(BASE_DIR, "ckks_context.pkl")
KEYS_DIR = os.path.join(BASE_DIR, "keys")
CAPTURE_PATH = os.path.join(BASE_DIR, "captured_face.jpg")

# ─── Flask Setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "biometric_secret_2024"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ─── Global State ─────────────────────────────────────────────────────────────
yolo_model = None
context = None
cap = None
real_counter = 0
scanning = False
processing = False
current_embedding = None
system_ready = False
latest_frame = None
frame_lock = threading.Lock()

# ─── Load Models ──────────────────────────────────────────────────────────────
def load_models():
    global yolo_model, context, system_ready
    try:
        print("🔧 Loading CKKS context...")
        if os.path.exists(CONTEXT_PATH):
            context = load_context(CONTEXT_PATH)
        else:
            context = create_ckks_context()
            save_context(context, CONTEXT_PATH)

        print("🤖 Loading YOLO model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        yolo_model.to('cuda')

        system_ready = True
        print("✅ All models loaded!")
        socketio.emit('system_status', {
            'ready': True,
            'message': '✅ System Ready'
        })
    except Exception as e:
        print(f"❌ Error: {e}")
        socketio.emit('system_status', {
            'ready': False,
            'message': f'❌ Error: {e}'
        })

# ─── Auto Process Face ────────────────────────────────────────────────────────
def auto_process_face():
    """Automatically process face after capture"""
    global current_embedding, processing

    try:
        processing = True
        socketio.emit('processing_start', {
            'message': '🧠 Processing face automatically...'
        })

        # Extract FaceNet embedding
        embedding = extract_feature_vector(CAPTURE_PATH)
        current_embedding = np.array(embedding, dtype=np.float64)

        # Try authentication
        db = get_database()
        collection = db["enrolled_users"]
        users = list(collection.find({}))

        if len(users) == 0:
            socketio.emit('process_result', {
                'status': 'unknown',
                'message': 'No users enrolled yet.',
                'score': -1
            })
            processing = False
            return

        best_match = None
        best_score = -1

        for user in users:
            user_id = user["user_id"]
            W = generate_transform_key(user_id, keys_dir=KEYS_DIR)
            protected_query = cancelable_transform(current_embedding, W)
            enc_query = ts.ckks_vector(context, protected_query.tolist())
            enc_stored = ts.ckks_vector_from(
                context, user["encrypted_template"])
            dot_product = enc_query.dot(enc_stored)
            score = float(dot_product.decrypt()[0])

            if score > best_score:
                best_score = score
                best_match = user

        matched = best_score >= SIMILARITY_THRESHOLD

        # Log access
        db["access_logs"].insert_one({
            "name": best_match['name'] if matched else "Unknown",
            "user_id": best_match['user_id'] if matched else "N/A",
            "score": round(best_score, 4),
            "status": "GRANTED" if matched else "DENIED",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        if matched:
            socketio.emit('process_result', {
                'status': 'granted',
                'name': best_match['name'],
                'user_id': best_match['user_id'],
                'score': round(best_score, 4)
            })
        else:
            socketio.emit('process_result', {
                'status': 'unknown',
                'score': round(best_score, 4),
                'message': 'Face not recognized. Enroll below.'
            })

    except Exception as e:
        socketio.emit('process_result', {
            'status': 'error',
            'message': str(e)
        })
    finally:
        processing = False

# ─── Camera Thread ────────────────────────────────────────────────────────────
def camera_thread():
    global cap, real_counter, scanning, latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    classNames = ["fake", "real"]

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        display_frame = frame.copy()

        if scanning and yolo_model and not processing:
            try:
                results = yolo_model(frame, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        label = classNames[cls]

                        if label == "real" and conf > CONFIDENCE_THRESHOLD:
                            real_counter += 1
                        else:
                            real_counter = 0

                        socketio.emit('progress', {
                            'counter': real_counter,
                            'required': REAL_COUNTER_REQUIRED,
                            'percent': min(
                                (real_counter / REAL_COUNTER_REQUIRED) * 100,
                                100)
                        })

                        # ✅ Auto capture + auto process
                        if real_counter >= REAL_COUNTER_REQUIRED:
                            face = frame[y1:y2, x1:x2]
                            if face.size > 0:
                                cv2.imwrite(CAPTURE_PATH, face)
                                scanning = False
                                real_counter = 0
                                socketio.emit('face_captured', {
                                    'message': '📸 Face captured!'
                                })
                                # Auto process immediately
                                threading.Thread(
                                    target=auto_process_face,
                                    daemon=True
                                ).start()

                        color = (0, 255, 0) if label == "real" else (0, 0, 255)
                        cv2.rectangle(
                            display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            display_frame,
                            f"{label.upper()} {int(conf*100)}%",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Progress bar on frame
                pw = int((real_counter / REAL_COUNTER_REQUIRED) * 200)
                cv2.rectangle(display_frame, (10, 10), (210, 35),
                              (30, 30, 30), -1)
                cv2.rectangle(display_frame, (10, 10),
                              (10 + pw, 35), (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    f"Real: {real_counter}/{REAL_COUNTER_REQUIRED}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
                cv2.putText(
                    display_frame, "SCANNING...",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            except Exception as e:
                print(f"Frame error: {e}")

        with frame_lock:
            latest_frame = display_frame.copy()

# ─── Video Stream ─────────────────────────────────────────────────────────────
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.033)
                continue
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')
        time.sleep(0.033)

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/captured_face')
def captured_face():
    if os.path.exists(CAPTURE_PATH):
        with open(CAPTURE_PATH, 'rb') as f:
            return Response(f.read(), mimetype='image/jpeg')
    return '', 404

@app.route('/api/start_scan', methods=['POST'])
def start_scan():
    global scanning, real_counter
    if not system_ready:
        return jsonify({'error': 'System not ready yet!'}), 400
    scanning = True
    real_counter = 0
    return jsonify({'success': True})

@app.route('/api/stop_scan', methods=['POST'])
def stop_scan():
    global scanning, real_counter
    scanning = False
    real_counter = 0
    return jsonify({'success': True})

@app.route('/api/enroll', methods=['POST'])
def enroll():
    global current_embedding
    try:
        data = request.json
        name = data.get('name', '').strip()
        user_id = data.get('user_id', '').strip()

        if not name or not user_id:
            return jsonify(
                {'error': 'Name and User ID are required!'}), 400

        if current_embedding is None:
            return jsonify(
                {'error': 'No face captured! Start scan first.'}), 400

        db = get_database()
        collection = db["enrolled_users"]

        if collection.find_one({"user_id": user_id}):
            return jsonify(
                {'error': f"User ID '{user_id}' already exists!"}), 400

        W = generate_transform_key(user_id, keys_dir=KEYS_DIR)
        protected = cancelable_transform(current_embedding, W)
        encrypted_bytes = encrypt_template(protected, context)

        collection.insert_one({
            "user_id": user_id,
            "name": name,
            "encrypted_template": encrypted_bytes,
            "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return jsonify({
            'success': True,
            'message': f'✅ {name} enrolled successfully!'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users')
def get_users():
    db = get_database()
    users = list(db["enrolled_users"].find(
        {}, {"_id": 0, "encrypted_template": 0}))
    return jsonify(users)

@app.route('/api/logs')
def get_logs():
    db = get_database()
    logs = list(db["access_logs"].find(
        {}, {"_id": 0}).sort("timestamp", -1).limit(50))
    return jsonify(logs)

@app.route('/api/stats')
def get_stats():
    db = get_database()
    return jsonify({
        'total_users': db["enrolled_users"].count_documents({}),
        'total_scans': db["access_logs"].count_documents({}),
        'granted': db["access_logs"].count_documents(
            {"status": "GRANTED"}),
        'denied': db["access_logs"].count_documents(
            {"status": "DENIED"})
    })

# ✅ Delete individual user
@app.route('/api/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        db = get_database()
        result = db["enrolled_users"].delete_one({"user_id": user_id})
        if result.deleted_count == 0:
            return jsonify({'error': 'User not found!'}), 404

        # Delete key file too
        key_path = os.path.join(KEYS_DIR, f"{user_id}_key.npy")
        if os.path.exists(key_path):
            os.remove(key_path)

        return jsonify({
            'success': True,
            'message': f'User {user_id} deleted!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_users', methods=['DELETE'])
def clear_users():
    db = get_database()
    db["enrolled_users"].delete_many({})
    return jsonify({'success': True})

@app.route('/api/clear_logs', methods=['DELETE'])
def clear_logs():
    db = get_database()
    db["access_logs"].delete_many({})
    return jsonify({'success': True})

# ─── SocketIO ─────────────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    socketio.emit('system_status', {
        'ready': system_ready,
        'message': '✅ System Ready' if system_ready else '⏳ Loading...'
    })

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    cam_thread = threading.Thread(target=camera_thread, daemon=True)
    cam_thread.start()

    model_thread = threading.Thread(target=load_models, daemon=True)
    model_thread.start()

    print("🌐 Starting at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
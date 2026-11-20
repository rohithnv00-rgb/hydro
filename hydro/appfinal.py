import numpy as np
import pandas as pd
import joblib
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
import hashlib
from datetime import datetime
import os
import serial
import cv2
import threading
import time
from tensorflow.keras.models import load_model

# ---------------------------------------------
# Flask App
# ---------------------------------------------
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# ---------------------------------------------
# Arduino Serial Setup
# ---------------------------------------------
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)   # Change COM port as needed
    print("✅ Arduino connected on COM3")
except Exception as e:
    print("❌ Arduino connection failed:", e)
    arduino = None

# ---------------------------------------------
# Load Leaf Disease Model
# ---------------------------------------------
DISEASE_MODEL_PATH = 'hydrophonic.h5'
DISEASE_CLASSES = ['Healthy', 'K Deficient', 'N Deficient', 'Wilt fungal']
IMG_HEIGHT = 256
IMG_WIDTH = 256

try:
    disease_model = load_model(DISEASE_MODEL_PATH)
    print("✅ Leaf disease model loaded")
except Exception as e:
    print("❌ Error loading leaf disease model:", e)
    disease_model = None

# ---------------------------------------------
# AI Suggestions for each disease class
# ---------------------------------------------
DISEASE_SUGGESTIONS = {
    "Healthy": {
        "status": "success",
        "summary": "Your plant is healthy. Keep up the good work!",
        "actions": [
            "Maintain current watering and nutrient schedule.",
            "Ensure 6–8 hours of indirect sunlight daily.",
            "Inspect leaves weekly for early signs of pests.",
            "Keep soil pH between 6.0 – 7.0."
        ]
    },
    "K Deficient": {
        "status": "warning",
        "summary": "Potassium deficiency detected. Leaves may show yellow/brown edges.",
        "actions": [
            "Apply potassium-rich fertilizer (e.g., Muriate of Potash / K2SO4).",
            "Use wood ash or banana peel compost as organic K source.",
            "Maintain soil pH 6.0 – 6.5 for better K uptake.",
            "Avoid over-watering — it leaches potassium."
        ]
    },
    "N Deficient": {
        "status": "warning",
        "summary": "Nitrogen deficiency detected. Leaves appear pale/yellowish.",
        "actions": [
            "Apply nitrogen-rich fertilizer (Urea / Ammonium Nitrate).",
            "Add organic matter like compost or manure.",
            "Consider planting legumes nearby for natural N-fixation.",
            "Ensure proper drainage to prevent N leaching."
        ]
    },
    "Wilt fungal": {
        "status": "danger",
        "summary": "Fungal wilt detected. This can spread quickly and kill the plant.",
        "actions": [
            "Remove and destroy infected leaves immediately.",
            "Apply fungicide (Carbendazim / Mancozeb) as per label.",
            "Improve air circulation around plants.",
            "Avoid overhead watering — water at the base.",
            "Disinfect tools between plants to stop spread."
        ]
    }
}

# ---------------------------------------------
# Camera / Global Prediction State
# ---------------------------------------------
camera = None
camera_lock = threading.Lock()

latest_prediction = {
    "disease": None,
    "confidence": None,
    "suggestion": None,
    "timestamp": None
}
prediction_lock = threading.Lock()

PREDICT_INTERVAL = 2.0   # seconds between predictions

def open_camera():
    """Open the laptop webcam (0 = default)."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)   # Change to 1, 2 if using external USB cam
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print("✅ Laptop camera opened")
            else:
                print("❌ Could not open laptop camera")
                camera = None

def close_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            print("📷 Camera released")

def get_frame():
    """Thread-safe frame grab."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            return None
        ret, frame = camera.read()
        if not ret:
            return None
        return frame

# ---------------------------------------------
# Background Prediction Worker
# ---------------------------------------------
def prediction_worker():
    """Continuously grab frames, run AI, and update latest_prediction."""
    global latest_prediction
    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.5)
            continue

        if disease_model is None:
            time.sleep(1)
            continue

        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
            img_array = np.expand_dims(img_resized, axis=0).astype('float32')
            # Uncomment if your model expects normalized inputs:
            # img_array = img_array / 255.0

            preds = disease_model.predict(img_array, verbose=0)[0]
            idx = int(np.argmax(preds))
            disease = DISEASE_CLASSES[idx]
            confidence = float(preds[idx]) * 100.0
            suggestion = DISEASE_SUGGESTIONS.get(disease, {})

            with prediction_lock:
                latest_prediction = {
                    "disease": disease,
                    "confidence": confidence,
                    "suggestion": suggestion,
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            print(f"Prediction worker error: {e}")

        time.sleep(PREDICT_INTERVAL)

# ---------------------------------------------
# Database Initialization
# ---------------------------------------------
def init_db():
    conn = sqlite3.connect('agriculture.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ph REAL NOT NULL,
            humidity REAL NOT NULL,
            temperature REAL NOT NULL,
            moisture REAL NOT NULL,
            ldr_status REAL NOT NULL,
            pump_status TEXT NOT NULL,
            disease TEXT,
            confidence REAL
        )
    ''')

    cursor.execute("PRAGMA table_info(sensor_logs)")
    cols = [c[1] for c in cursor.fetchall()]
    if 'disease' not in cols:
        cursor.execute("ALTER TABLE sensor_logs ADD COLUMN disease TEXT")
    if 'confidence' not in cols:
        cursor.execute("ALTER TABLE sensor_logs ADD COLUMN confidence REAL")

    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect('agriculture.db')
    conn.row_factory = sqlite3.Row
    return conn

# ---------------------------------------------
# Load ML Model (pump prediction)
# ---------------------------------------------
try:
    model = joblib.load("randomforest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    imputer = joblib.load("imputer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    if hasattr(model, "feature_names_in_"):
        trained_features = list(model.feature_names_in_)
    else:
        trained_features = ["pH", "humidity", "temperature", "moisture", "ldr_status"]

    print("✅ ML pump model loaded")

except Exception as e:
    print("❌ Error loading ML pump model:", e)
    trained_features = ["pH", "humidity", "temperature", "moisture", "ldr_status"]

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def read_sensor_data():
    try:
        with open('sensor_data.txt', 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1].strip()
                values = last_line.split('\t')
                if len(values) == 5:
                    return {
                        'pH': float(values[0]),
                        'humidity': float(values[1]),
                        'temperature': float(values[2]),
                        'moisture': float(values[3]),
                        'ldr_status': float(values[4])
                    }
    except Exception as e:
        print(f"Error reading sensor data: {e}")
    return None

def predict_pump_status(sensor_data):
    try:
        df = pd.DataFrame([sensor_data])
        for f in trained_features:
            if f not in df.columns:
                df[f] = np.nan
        df = df[trained_features]
        X = imputer.transform(df)
        X_scaled = scaler.transform(X)
        pred_encoded = model.predict(X_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        return pred_label
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

def log_sensor_data(sensor_data, pump_status, disease=None, confidence=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO sensor_logs
           (ph, humidity, temperature, moisture, ldr_status, pump_status, disease, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (
            sensor_data['pH'], sensor_data['humidity'],
            sensor_data['temperature'], sensor_data['moisture'],
            sensor_data['ldr_status'], pump_status,
            disease, confidence
        )
    )
    conn.commit()
    conn.close()

def send_to_arduino(pump_status):
    if arduino:
        try:
            if str(pump_status) == "1" or str(pump_status).lower() == "on":
                arduino.write(b'1')
                print("➡️ Sent '1' to Arduino (PUMP ON)")
            else:
                arduino.write(b'0')
                print("➡️ Sent '0' to Arduino (PUMP OFF)")
        except Exception as e:
            print("❌ Serial Write Error:", e)

# ---------------------------------------------
# MJPEG Video Generator
# ---------------------------------------------
def gen_frames():
    """Yield JPEG-encoded frames for the browser MJPEG stream."""
    while True:
        frame = get_frame()
        if frame is None:
            # Send a placeholder black frame while camera not ready
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Camera not available", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                (username, hash_password(password), email)
            )
            conn.commit()
            flash("Registration successful!", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or Email already exists!", "error")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE username = ? AND password = ?',
            (username, hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials!", "error")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Start camera + prediction worker on first dashboard visit
    open_camera()
    if not any(t.name == 'prediction_worker' for t in threading.enumerate()):
        t = threading.Thread(target=prediction_worker, name='prediction_worker', daemon=True)
        t.start()
        print("🧵 Prediction worker started")

    sensor_data = read_sensor_data()
    pump_status = "Unknown"
    if sensor_data:
        pump_status = predict_pump_status(sensor_data)

    with prediction_lock:
        pred_snapshot = dict(latest_prediction)

    return render_template(
        'dashboard.html',
        sensor_data=sensor_data,
        pump_status=pump_status,
        disease=pred_snapshot['disease'],
        disease_confidence=pred_snapshot['confidence'],
        suggestion=pred_snapshot['suggestion'],
        username=session['username']
    )

@app.route('/video_feed')
def video_feed():
    """Live MJPEG stream of the laptop camera."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    open_camera()
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/prediction')
def api_prediction():
    """Return latest AI prediction (polled by dashboard JS)."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    with prediction_lock:
        pred = dict(latest_prediction)
    return jsonify(pred)

@app.route('/api/sensor-data')
def api_sensor_data():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    sensor_data = read_sensor_data()
    pump_status = "Unknown"

    if sensor_data:
        pump_status = predict_pump_status(sensor_data)
        send_to_arduino(pump_status)

    with prediction_lock:
        pred = dict(latest_prediction)

    if sensor_data:
        log_sensor_data(sensor_data, pump_status, pred['disease'], pred['confidence'])

    return jsonify({
        'sensor_data': sensor_data,
        'pump_status': pump_status,
        'disease': pred['disease'],
        'disease_confidence': pred['confidence'],
        'suggestion': pred['suggestion'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/camera/stop')
def camera_stop():
    """Optional: release the camera."""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    close_camera()
    flash("Camera stopped.", "info")
    return redirect(url_for('dashboard'))

@app.route('/logs')
def logs():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    date_filter = request.args.get('date', '')

    conn = get_db_connection()
    cursor = conn.cursor()

    if date_filter:
        cursor.execute(
            'SELECT * FROM sensor_logs WHERE DATE(timestamp) = ? ORDER BY timestamp DESC',
            (date_filter,)
        )
    else:
        cursor.execute('SELECT * FROM sensor_logs ORDER BY timestamp DESC LIMIT 100')

    logs = cursor.fetchall()
    conn.close()

    return render_template('logs.html', logs=logs, date_filter=date_filter)

@app.route('/logout')
def logout():
    close_camera()
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for('index'))

# ---------------------------------------------
# Run Server
# ---------------------------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=False, host='0.0.0.0', threaded=True)

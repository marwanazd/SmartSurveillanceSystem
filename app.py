from datetime import datetime, timedelta
import logging
import os
import cv2
import threading
from flask import Flask, Response, request, jsonify, render_template, send_from_directory
from sqlalchemy import Column, Integer, String, Float, create_engine
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from werkzeug.utils import secure_filename

from utils.logging_config import setup_logging
from modules.camera import CameraManager
from modules.face_recognition_model import FaceRecognitionModel
from modules.attendance_system import AttendanceTracker
from modules.tracker_model import Tracker
from modules.object_detection_model import ObjectDetectionModel

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Define upload folder and allowed file extensions
UPLOAD_FOLDER = 'image/faces_dataset'
FACE_RECOGNITION_MODEL = 'instance/face_recognition.db'

class Drawing(db.Model):
    __tablename__ = 'drawing'
    id = Column(Integer, primary_key=True)
    type = Column(String(50))
    a = Column(Float)  # x1
    b = Column(Float)  # y1
    c = Column(Float)  # x2
    d = Column(Float)  # y2

    def __repr__(self):
        return f'<Drawing {self.id}>'
    

# Create the database tables
with app.app_context():
    db.create_all()

# Create a CameraManager object to handle the video source.
video = CameraManager(source=0, rotate=False)

def satrt_video():
    video.start()

# Initialize the face recognition model
face_recognizer = FaceRecognitionModel()

# Initialize the attendance tracker
attendance_tracker = AttendanceTracker("instance/attendance.db")
if os.path.exists(FACE_RECOGNITION_MODEL):
    #print("Face recognition database found. Loading model...")
    face_recognizer.load(FACE_RECOGNITION_MODEL)

# Initialize the object detection model
detection = ObjectDetectionModel(
    model_names=['fireModel', 'knive', 'best'],
    db_name='instance/detection'
    )

# Initialize the tracker model
tracker = Tracker(db='instance/trackings.db')

# Flag to stop processing thread gracefully
processing_active = True


# Dummy training function
def train_model(name, folder_path):
    print(f"Training model for {name} with images in {folder_path}")
    if os.path.exists(FACE_RECOGNITION_MODEL):
        face_recognizer.load(FACE_RECOGNITION_MODEL)

    if face_recognizer.weights_exist:
        face_recognizer.update(folder_path)
    else:
        face_recognizer.train(folder_path)

    face_recognizer.save(
        method='sql',
        encodings_path=FACE_RECOGNITION_MODEL
    )

# returns the total storage usage of a specific folder
def get_folder_size(folder_path):
    """Recursively calculate the total size of a folder (including subfolders)."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

# Start the processing thread
processing_thread = threading.Thread(target=satrt_video, daemon=True)
processing_thread.start()

def gen():
    """Generate frames for the real-time video stream."""
    while processing_active:
        frame = video.get_frame()
        if frame is not None:
            # Detect objects in the frame
            frame = detection.detect_objects(frame=frame, confidence=0.7)

            # Get the detected faces
            #detected_faces = face_recognizer.detect(frame)
            
            # Update attendance (and show names on the frame)
            #attendance_tracker.record_attendance(detected_faces)
            
            # Plot the attendance on the frame
            #frame = face_recognizer.plot(frame, detected_faces)
            
            # Convert the frame to JPEG and yield as response for streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                # Yield the frame to be displayed in the browser
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """Video feed route."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# PAGES

# HOME PAGE
@app.route('/')
def home():
    # Get metadata from the camera feed
    fps = video.get_fps()
    resolution = video.get_resolution()
    metadata = {
        'fps': fps,
        'resolution': resolution
    }
    # Pass metadata and other statistics to the template
    return render_template('index.html', title="Home", metadata=metadata)

# DASHBOARD PAGES
@app.route('/attendance')
def attendance():
    return render_template('attendance.html', title="Attendance")

# SETTINGS PAGE
@app.route('/settings')
def settings():
    return render_template('settings.html', title="Settings")

@app.route('/drow')
def drow():
    return render_template('drow.html')

# ABOUT PAGE
@app.route('/about')
def about():
    return render_template('about.html', title="About")


# API ENDPOINTS

@app.route('/api/attendance', methods=['GET'])
def get_attendance():
    """
    Endpoint to fetch attendance records based on the time period filter.
    """
    time_period = request.args.get('timePeriod', 'today')
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')

    print(time_period)
    
    if time_period == 'specific' and start_date and end_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        start_date = end_date = None

    records = attendance_tracker.get_attendance_records(time_period, start_date, end_date)
    
    response = {'records': records}
    return jsonify(response)


@app.route('/api/download', methods=['GET'])
def download_records():
    # Get query parameters
    time_period = request.args.get('timePeriod', 'today')
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')
    file_format = request.args.get('fileFormat', 'csv')

    # Convert dates if provided
    if start_date:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Call save_attendance_records to save the file
    file_path = attendance_tracker.save_attendance_records(
        time_period=time_period,
        start_date=start_date,
        end_date=end_date,
        file_format=file_format,
        file_path="attendance_records"
    )

    # Check if file was saved successfully
    if not file_path:
        return jsonify({'error': 'Failed to save the file'}), 500

    # Check if the file exists and send it
    try:
        # Ensure the file exists in the upload folder
        if os.path.exists(file_path):
            directory, filename = os.path.split(file_path)
            return send_from_directory(directory, filename, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404

    except Exception as e:
        print(f"Error serving the file: {e}")
        return jsonify({'error': 'Error serving the file'}), 500

@app.route('/api/add_person', methods=["POST"])
def add_person():
    name = request.form.get("name")
    images = request.files.getlist("images")
    if not name or not images:
        return "Missing data", 400

    # Save images to folder
    folder_path = os.path.join("faces_dataset", name)
    os.makedirs(folder_path, exist_ok=True)
    for image in images:
        image.save(os.path.join(folder_path, image.filename))

    # Train model (placeholder for training logic)
    train_model(name, folder_path)

    return "Person added", 200

@app.route('/api/events', methods=['GET'])
def get_events():
    """
    Endpoint to fetch events.
    Combines object detection and attendance records, 
    adds event types, sorts them by timestamp, and returns a unified response.
    """
    try:
        global total_events
        # Fetch records
        attendance_records = attendance_tracker.get_attendance_records('all')
        object_detection_records = detection.load_from_db()

        # Add event type to each record
        for obj in object_detection_records:
            obj['eventtype'] = 'ObjectDetection'

        for record in attendance_records:
            record['eventtype'] = 'Attendance'

        # Combine and sort the records
        events = object_detection_records + attendance_records
        sorted_data = sorted(events, key=lambda x: datetime.fromisoformat(x['timestamp']), reverse=True)

        # Prepare the final list with the required fields
        total_events = [
            {
                'id': i,
                'eventtype': item['eventtype'],
                'name': item.get('label', item.get('name')),  # Label for ObjectDetection, name for Attendance
                'timestamp': item['timestamp'],
                'img': item.get('img_base64', item.get('face_image'))  # img_base64 for ObjectDetection, face_image for Attendance
            }
            for i, item in enumerate(sorted_data)
        ]

        # Build response
        response = {'events': total_events}
        return jsonify(response), 200
    except Exception as e:
        # Log the error and return a meaningful response
        print(f"Error fetching events: {e}")
        return jsonify({'error': 'Failed to fetch events'}), 500

@app.route('/api/fabLab_statistics', methods=['GET'])
def fabLab_statistics():
    """
    Endpoint to fetch FabLab statistics Today.
    """
    records = attendance_tracker.get_attendance_records('today')
    detected_objects, up_count, down_count = tracker.load_crossings_from_db()
    # Total People Detected
    total_records = len(records)
    # Unknown Visitors/Guests if record['name'] has 'Unknown' on it
    unknown_visitors = len([record for record in records if 'unknown' in record['name'].lower()])

    peopleExited = up_count

    statistics = {
        'total_people_detected': total_records,
        'unknown_visitors': unknown_visitors,
        'known_visitors': total_records - unknown_visitors,
        'people_exited': peopleExited,
        'people_in_fablab': total_records - peopleExited,
    }

    response = {'statistics': statistics}
    return jsonify(response)

@app.route('/api/storage', methods=['GET'])
def get_storage():
    """
    Endpoint to return the total storage used by the faces_dataset and instance folders.
    """
    faces_dataset_path = os.path.join(os.getcwd(), 'faces_dataset')  # Modify as per your folder path
    instance_path = os.path.join(os.getcwd(), 'instance')  # Modify as per your folder path

    # Check if directories exist
    if not os.path.exists(faces_dataset_path) or not os.path.exists(instance_path):
        return jsonify({'error': 'One or more folders not found'}), 404

    # Get the total size of the folders
    faces_dataset_size = get_folder_size(faces_dataset_path)
    instance_size = get_folder_size(instance_path)

    # Calculate the total size
    total_size = faces_dataset_size + instance_size

    # Convert the sizes to MB for easier understanding
    faces_dataset_size_mb = faces_dataset_size / (1024 * 1024)
    instance_size_mb = instance_size / (1024 * 1024)
    total_size_mb = total_size / (1024 * 1024)

    return jsonify({
        'faces_dataset': round(faces_dataset_size_mb, 2),
        'instance': round(instance_size_mb, 2),
        'total_size': round(total_size_mb, 2)
    })

@app.route('/api/objectDetected', methods=['GET'])
def object_detected():
    """
    Endpoint to fetch object detected.
    """
    object_detected = attendance_tracker.get_object_detected()
    response = {'object_detected': object_detected}
    return jsonify(response)

@app.route('/api/totalEvants', methods=['GET'])
def get_total_events():
    if total_events is None:
        total = 0
    else:
        total = len(total_events)
    response = {
        "total_events": total  # Key name should match your frontend's expectation
    }
    return jsonify(response)

@app.route('/drow')
def index():
    # Render the home page with the video stream and drawing interface
    return render_template('drow.html')

@app.route('/add_drawing', methods=['POST'])
def add_drawing():
    """
    Add a drawing (rectangle or line) to the database.
    Expected JSON format: {"type": "rectangle", "a": x1, "b": y1, "c": x2, "d": y2}
    """
    data = request.json
    print(data)
    if data:
        drawing_type = data.get('type')  # Can be "rectangle" or "line"
        a = data.get('a')  # First coordinate (x1 or x)
        b = data.get('b')  # Second coordinate (y1 or y)
        c = data.get('c')  # Third coordinate (x2 or end x)
        d = data.get('d')  # Fourth coordinate (y2 or end y)
        
        # Create a new Drawing object
        new_drawing = Drawing(type=drawing_type, a=a, b=b, c=c, d=d)

        # Add the new drawing to the database
        db.session.add(new_drawing)
        db.session.commit()

        return jsonify({'message': 'Drawing added successfully', 'id': new_drawing.id}), 200
    return jsonify({'error': 'Invalid data'}), 400

@app.route('/get_drawings', methods=['GET'])
def get_drawings():
    """
    Retrieve all saved drawings from the database.
    Returns a list of drawings with their ids, types, and coordinates.
    """
    drawings = Drawing.query.all()
    drawing_list = []

    # Convert each drawing to a dictionary to send in the response
    for drawing in drawings:
        drawing_list.append({
            'id': drawing.id,
            'type': drawing.type,
            'a': drawing.a,
            'b': drawing.b,
            'c': drawing.c,
            'd': drawing.d
        })

    return jsonify(drawing_list)

@app.route('/remove_drawing', methods=['POST'])
def remove_drawing():
    """
    Remove a drawing from the database by its ID.
    Expected JSON format: {"id": drawing_id}
    """
    data = request.json
    drawing_id = data.get('id')

    if drawing_id:
        drawing = Drawing.query.get(drawing_id)
        if drawing:
            db.session.delete(drawing)
            db.session.commit()
            return jsonify({'message': 'Drawing removed successfully'}), 200
        return jsonify({'error': 'Drawing not found'}), 404
    return jsonify({'error': 'No ID provided'}), 400

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Stop processing thread gracefully when the application exits
        processing_active = False
        processing_thread.join()
        video.stop()
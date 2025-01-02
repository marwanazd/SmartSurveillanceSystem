import base64
from datetime import datetime, timedelta
import time
import cv2
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ultralytics import YOLO

# Initialize SQLAlchemy base and session
Base = declarative_base()

# Define the DetectedObject model for SQLAlchemy
class DetectedObject(Base):
    __tablename__ = 'detected_objects'
    
    id = Column(Integer, primary_key=True)
    class_id = Column(Integer)
    timestamp = Column(String)
    label = Column(String)
    img_base64 = Column(String, nullable=True)

model_name = 'best'

class ObjectDetectionModel:
    """
    Return a list of detected objects in an OpenCV video frame with timestamp and box image in base64 format,
    excluding 'person' objects.
    """
    def __init__(self, model_names: list, db_name: str):
        self.models = [YOLO(f'models/{model}.pt') for model in model_names]
        self.engine = create_engine(f'sqlite:///{db_name}.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        # Initialize a list to store the detected objects' data
        self.detected_objects = []
        self.last_detection_time = {}

    def predict(self, model, frame):
        return model(frame)  # Pass frame directly for predictions
    
    def detect_objects(self, frame, confidence=0.5, time_threshold_minutes=1):
        # Time threshold in minutes
        time_threshold = timedelta(minutes=time_threshold_minutes)
        
        # Detect objects in the frame
        for model in self.models:
            results = self.predict(model, frame)

            # Extract detections with confidence > confidence
            detections = results[0].boxes.data if hasattr(results[0], 'boxes') else []
            filtered_detections = [
                box for box in detections if box[4] > confidence  # Assuming box[4] is the confidence
            ]


            # Draw filtered bounding boxes and labels
            for detection in filtered_detections:
                # Extract coordinates and other information from the tensor
                x1, y1, x2, y2, confidence, class_id = detection.tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = int(class_id)
                label = model.names[class_id]
                title = f"{model.names[class_id]} {confidence:.2f}"  # Use model.names for class name

                # Draw the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, title, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Skip detection if it's a 'person' (class_id 0 for person in YOLO)
                clas_to_skip = ['person', 'face_mask', 'no_face_mask', 'incorrect_face_mask', 'violence']
                if label in clas_to_skip:
                    continue

                # Query the database for the last detection of the same label
                db_session = self.Session()
                last_detected = (
                    db_session.query(DetectedObject)
                    .filter(DetectedObject.label == label)
                    .order_by(DetectedObject.timestamp.desc())
                    .first()
                )

                # Get current timestamp
                timestamp = datetime.now()

                # Skip if the label was detected within the time threshold
                if last_detected and timestamp - datetime.strptime(last_detected.timestamp, '%Y-%m-%d %H:%M:%S') <= time_threshold:
                    continue

                # Extract the object image from the bounding box
                object_img = results[0].plot()
                
                # Skip if the extracted object image is empty
                if object_img.size == 0:
                    continue
                
                # Convert the object image to base64 format
                _, buffer = cv2.imencode('.jpg', object_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')

                # Create a new database entry
                detected_obj = DetectedObject(
                    class_id=class_id,
                    label=label,
                    timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    img_base64=img_base64  # Store the base64 image as binary
                )
                db_session.add(detected_obj)

                # Commit the session to save the data
                db_session.commit()
                
                # Close the database session
                db_session.close()

    def load_from_db(self):
        """
        Load the detected objects from the database and return them as a list of dictionaries.
        """
        # Create a new database session
        db_session = self.Session()

        # Query all detected objects from the database
        detected_objects_db = db_session.query(DetectedObject).all()

        # Convert the results to a list of dictionaries
        detected_objects = []
        for obj in detected_objects_db:
            detected_obj = {
                'id': obj.class_id,
                'label': obj.label,
                'timestamp': obj.timestamp,
                'img_base64': obj.img_base64  # Convert binary image back to base64
            }
            detected_objects.append(detected_obj)

        # Close the database session
        db_session.close()

        return detected_objects
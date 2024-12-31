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
    img_base64 = Column(LargeBinary)

model_name = 'best'

class ObjectDetectionModel:
    """
    Return a list of detected objects in an OpenCV video frame with timestamp and box image in base64 format,
    excluding 'person' objects.
    """
    def __init__(self, model_name: str, db_name: str):
        self.model = YOLO(f'models/{model_name}.pt')
        self.engine = create_engine(f'sqlite:///{db_name}.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        # Initialize a list to store the detected objects' data
        self.detected_objects = []
        self.last_detection_time = {}

    def predict(self, frame):
        return self.model(frame)  # Pass frame directly for predictions
    
    def detect_objects(self, frame, time_threshold_minutes=1):
        # Time threshold in minutes (default 5 minutes)
        time_threshold = timedelta(minutes=time_threshold_minutes)
        
        # Detect objects in the frame
        results = self.predict(frame)

        # Extract the boxes, labels, and coordinates
        boxes = results[0].boxes  # Access the first result
        idd = boxes.cls  # Class labels (IDs)
        coordinates = boxes.xywh  # Coordinates in xywh format (center_x, center_y, width, height)

        # Iterate over detected objects
        for label, coordinate in zip(idd, coordinates):
            print(f"Label: {label.item()}, Coordinates (xywh): {coordinate.tolist()}")
            # Get bounding box coordinates and confidence
            [x1, y1, x2, y2] = coordinate.tolist()

            # Convert coordinates to integers for correct slicing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            class_id = int(label.item())
            label_name = self.model.names[class_id]
            
            # Skip the detection if the object is a 'person' (class_id 0 for person in YOLO)
            if label_name == 'person':
                continue

            # Check if the label was detected recently
            timestamp = datetime.now()
            if label_name in self.last_detection_time and timestamp - self.last_detection_time[label_name] <= time_threshold:
                continue  # Skip appending if within the threshold time

            # Extract the object image from the bounding box
            object_img = frame[y1:y2, x1:x2]
            
            # Check if the object image is not empty
            if object_img.size == 0:
                continue
            
            # Convert the object image to base64 format
            _, buffer = cv2.imencode('.jpg', object_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Prepare the data for the detected object
            detected_obj = {
                'id': class_id,
                'label': label_name,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'img_base64': img_base64
            }
            
            # Append the detected object to the list
            self.detected_objects.append(detected_obj)
            
            # Update the last detection time for the label
            self.last_detection_time[label_name] = timestamp
        
        # Print the number of detected objects and return the list
        print(f"Detected {len(self.detected_objects)} objects.")
        return self.detected_objects

    def save_to_db(self, detected_objects):
        """
        Save the list of detected objects to the database.
        """
        # Create a new database session
        db_session = self.Session()

        for obj in detected_objects:
            detected_obj = DetectedObject(
                class_id=obj['id'],
                label=obj['label'],
                timestamp=obj['timestamp'],
                img_base64=base64.b64decode(obj['img_base64'])  # Store the base64 image as binary
            )
            db_session.add(detected_obj)
        
        # Commit the session to save the data
        db_session.commit()
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
                'img_base64': base64.b64encode(obj.img_base64).decode('utf-8')  # Convert binary image back to base64
            }
            detected_objects.append(detected_obj)

        # Close the database session
        db_session.close()

        return detected_objects
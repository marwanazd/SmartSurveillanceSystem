import cv2
import base64
from io import BytesIO
import os
import pickle
import face_recognition
import logging
from imutils import paths
import numpy as np
from datetime import datetime
import time
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from typing import List, Tuple
from supervision.detection.core import Detections
import supervision as sv


from modules.camera import CameraManager

# Define the SQLAlchemy base
Base = declarative_base()

# Define the Attendance model
class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    date = Column(String, nullable=False)
    time = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Person(Base):
    __tablename__ = 'person'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)  # Allow duplicate names
    encodings = relationship("Encoding", back_populates="person")

class Encoding(Base):
    __tablename__ = 'encoding'
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('person.id'))
    encoding = Column(LargeBinary, nullable=False)
    person = relationship("Person", back_populates="encodings")



class FaceSave:
    """
    This class is for taking pictures of faces and saving them to a dataset.

    Args:
        cam (CameraFeed): Camera module that contains video source and frames using the function "CameraFeed.get_frame()".
    """

    def __init__(self, cam: CameraManager) -> None:
        self.cam = cam
        self.photo_count = 0

    def create_folder(self, dataset_folder:str="dataset", name: str=None) -> str:
        """
        Create a folder to store photos for the given person.

        Args:
            name (str): Name of the person.

        Returns:
            str: Path to the created folder.
        """
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        person_folder = os.path.join(dataset_folder, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        else:
            logging.warning('Person folder exists. Please change the name.')
        return person_folder

    def capture_photos(self, dataset_folder:str, name: str, num_photos: int=15):
        """
        Capture a specified number of photos for a specific person and save them to a folder.

        Args:
            name (str): Name or id of the person being photographed.
            num_photos (int): The number of photos to capture.
        """
        folder = self.create_folder(
            dataset_folder=dataset_folder,
            name=name
        )
        
        logging.info(f"Capturing {num_photos} photos for {name}.")

        self.cam.open()
        
        photos_taken = 0
        last_capture_time = 0  # For debouncing key presses

        while photos_taken < num_photos:
            frame = self.cam.get_frame()

            # Capture photo automatically without waiting for a key press
            current_time = time.time()
            if current_time - last_capture_time > 0.5:  # Debounce time between captures
                last_capture_time = current_time
                photos_taken += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, frame)
                logging.info(f"Photo {photos_taken} saved: {filepath}")

        # Clean up
        logging.info(f"Photo capture completed. {photos_taken} photos saved for {name}.")

class FaceRecognitionModel:
    """
    A class to handle facial recognition tasks, including training, saving, and detecting faces.
    """
    def __init__(self):
        # Initialize the ByteTrack tracker
        self.tracker = sv.ByteTrack()
        
        # Initialize the BoxAnnotator and LabelAnnotator for drawing bounding boxes and labels
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Initialize the weights dictionary with empty lists if it doesn't exist
        self.weights = {"encodings": [], "names": [], "id": []}
        if not self.weights["encodings"] and not self.weights["names"] and not self.weights["id"]:
            self.weights_exist = False
        else:
            self.weights_exist = True

        # If the weights have existing IDs, set the person_id_counter to the next available ID
        if self.weights["id"]:
            self.person_id_counter = len(self.weights["id"]) + 1  # Start from the next available ID
        else:
            self.person_id_counter = 1  # No existing data, start from 1
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.cv_scaler = 4  # Used for resizing frames to improve performance

        self.attendance_data = []

        # A dictionary to track unknown faces by encoding and assign unique IDs
        self.unknown_faces = {}

        # Keep track of the last clearance time
        self.last_clearance_time = datetime.now()

    def train(self, dataset_path) -> dict:
        """
        Train the model by encoding faces in the dataset.
        """
        logging.info("[INFO] Start processing faces...")
        weights = {"encodings": [], "names": [], "id": []}
        image_paths = list(paths.list_images(dataset_path))
        
        for (i, image_path) in enumerate(image_paths):
            logging.info(f"[INFO] Processing image {i + 1}/{len(image_paths)}")
            name = image_path.split(os.path.sep)[-2]
            
            image = cv2.imread(image_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            for encoding in encodings:
                weights["encodings"].append(encoding)
                weights["names"].append(name)
                weights["id"].append(self.person_id_counter)
                self.person_id_counter += 1  # Increment unique ID for each person
        
        if not self.weights_exist:
            self.weights = weights
        
        return weights
    
    def update(self, dataset_path) -> dict:
        """
        Add new face encodings to the existing encodings from a new dataset.
        return: Updated model weights
        """
        logging.info("[INFO] Adding new weights...")
        new_weights = self.train(dataset_path)

        # Append new weights (encodings, names, and ids) to the existing ones
        self.weights["encodings"].extend(new_weights["encodings"])
        self.weights["names"].extend(new_weights["names"])
        self.weights["id"].extend(new_weights["id"])

        logging.info("[INFO] New weights added.")

    def save(self, method: str, encodings_path: str, ClearExistingData=False):
        """
        Save the trained encodings to a file or database.
        
        Args:
            method (str): Storage method ('pickle' or 'sql')
            encodings_path (str): Path to save encodings file or database connection string
        """
        if method == "pickle":
            if not encodings_path.lower().endswith('.pickle'):
                encodings_path += '.pickle'
            try:
                with open(encodings_path, "wb") as f:
                    pickle.dump(self.weights, f)
                logging.info(f"[INFO] Encodings saved to '{encodings_path}'")
            except Exception as e:
                logging.error(f"[ERROR] Failed to save pickle file: {e}")

        elif method == "sql":
            try:
                engine = create_engine(f'sqlite:///{encodings_path}')
                Base.metadata.create_all(engine)
                Session = sessionmaker(bind=engine)
                session = Session()

                # If ClearExistingData is False, do not delete existing records
                if ClearExistingData:
                    session.query(Encoding).delete()
                    session.query(Person).delete()
                    session.commit()

                # Add new data
                for name, encoding, person_id in zip(self.weights['names'], self.weights['encodings'], self.weights['id']):
                    # Check if the person already exists in the database
                    person = session.query(Person).filter_by(id=person_id).first()
                    if not person:
                        # If the person does not exist, create a new record with a unique ID
                        person = Person(name=name)
                        session.add(person)
                        session.commit()

                    # Check if the encoding already exists for this person
                    existing_encoding = session.query(Encoding).filter_by(person_id=person.id).first()
                    if existing_encoding:
                        logging.info(f"[INFO] Encoding for person {name} (ID: {person_id}) already exists, skipping...")
                    else:
                        # Add encoding data for the person with the unique person_id
                        encoding_obj = Encoding(person_id=person.id, encoding=pickle.dumps(encoding))
                        session.add(encoding_obj)
                        logging.info(f"[INFO] Added encoding for person {name} (ID: {person_id})")

                session.commit()

                logging.info(f"[INFO] Encodings saved to database at '{encodings_path}'")
            except Exception as e:
                logging.error(f"[ERROR] Failed to save to database: {e}")

    def load(self, encodings_path: str):
        """
        Load pre-trained encodings from a file or database.
        
        Args:
            encodings_path (str): Path to either .pickle file or database connection string
        """
        if encodings_path.endswith('.pickle'):
            try:
                with open(encodings_path, "rb") as f:
                    self.weights = pickle.load(f)
                self.weights_exist = True
                logging.info(f"[INFO] Encodings loaded from '{encodings_path}'")
            except Exception as e:
                logging.error(f"[ERROR] Failed to load pickle file: {e}")

        else:
            try:
                engine = create_engine(f'sqlite:///{encodings_path}')
                Session = sessionmaker(bind=engine)
                session = Session()

                self.weights["encodings"] = []
                self.weights["names"] = []
                self.weights["id"] = []

                people = session.query(Person).all()
                for person in people:
                    for encoding_obj in person.encodings:
                        self.weights["encodings"].append(pickle.loads(encoding_obj.encoding))
                        self.weights["names"].append(person.name)
                        self.weights["id"].append(person.id)
                self.weights_exist = True
                logging.info(f"[INFO] Encodings loaded from database at '{encodings_path}'")
            except Exception as e:
                logging.error(f"[ERROR] Failed to load from database: {e}")


    def encode_face_to_base64(self, face_image) -> str:
        """
        Encode the cropped face image to Base64.

        Args:
            face_image: The cropped face image (in OpenCV format).

        Returns:
            str: The Base64 encoded string of the image.
        """
        _, buffer = cv2.imencode('.jpg', face_image)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        return face_base64

    def detect(self, frame) -> list: 
        """
        Detect faces in a given video frame and recognize them.

        return: Detected faces and their names in a dict {"locations": list, "person_name": list, "base64_images": list}
        """
        # Load model
        self.known_encodings = self.weights["encodings"]
        self.known_names = self.weights["names"]
        self.known_ids = self.weights["id"]

        # Initialize the detected faces list
        detected_faces = []

        # Resize and process the frame
        resized_frame = cv2.resize(frame, (0, 0), fx=(1 / self.cv_scaler), fy=(1 / self.cv_scaler))
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        self.face_locations = face_recognition.face_locations(rgb_resized_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_resized_frame, self.face_locations)

        # Loop through each face encoding and compare to known encodings
        for i, face_encoding in enumerate(self.face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            person_name = "unknown"
            top, right, bottom, left = self.face_locations[i]

            if True in matches:
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                person_name = self.known_names[best_match_index]
            else:
                # Handle unknown faces
                # Use a unique ID for each unknown face
                face_id = self.get_unique_unknown_id(face_encoding)
                person_name = f"unknown_{face_id}"

            # Crop the face from the frame
            face_image = resized_frame[top:bottom, left:right]
            # Encode the face image to Base64
            face_base64 = self.encode_face_to_base64(face_image=face_image)
            
            face = {
                "location": (top, right, bottom, left),
                "encodings": face_encoding,
                "name": person_name,
                "face_base64": face_base64
            }

            detected_faces.append(face)
        
        return detected_faces

    def get_unique_unknown_id(self, face_encoding) -> int:
        """
        Get a unique ID for each unknown face by checking if it already exists.
        """
        # Check if the encoding is already in the unknown_faces dictionary
        for unique_id, stored_encoding in self.unknown_faces.items():
            # Compare face encodings (using a threshold for tolerance)
            if face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)[0]:
                return unique_id  # Return existing unique ID if the face is already tracked
        
        # If the face is not tracked, assign a new unique ID
        new_id = self.person_id_counter
        self.unknown_faces[new_id] = face_encoding  # Store the encoding for future reference
        self.person_id_counter += 1  # Increment the ID for the next unknown face
        return new_id
    
    def clear_unknown_faces(self, times_to_clear: list, interval_minutes: int = None):
        """
        Clears the list of unknown faces at specified times or at regular intervals.

        Args:
            times_to_clear (list): List of tuples where each tuple is (hour, minute).
            interval_minutes (int): Interval in minutes to clear the list regularly.
        """
        current_time = datetime.now()
        current_hour, current_minute = current_time.hour, current_time.minute
        
        # Check specific times
        for time_tuple in times_to_clear:
            hour, minute = time_tuple
            if current_hour == hour and current_minute == minute:  # Matches the given time
                logging.info(f"[INFO] Time is {hour}:{minute}. Clearing unknown faces.")
                self.unknown_faces.clear()
                self.last_clearance_time = current_time  # Reset the last clearance time
                return

        # Check interval
        if interval_minutes is not None:
            time_since_last_clearance = (current_time - self.last_clearance_time).total_seconds() / 60
            if time_since_last_clearance >= interval_minutes:
                logging.info(f"[INFO] Clearing unknown faces at {current_time}. Interval: {interval_minutes} minutes.")
                self.unknown_faces.clear()
                self.last_clearance_time = current_time

    def plot(self, frame, detected_faces: list = None) -> None:
        """
        Plot the detected faces on the frame, showing known and unknown faces differently.

        Args:
            frame: The video frame where faces are to be plotted.
            detected_faces: A dictionary containing detected faces, locations, and names (can be from detect method).
        """
        if detected_faces is None:
            detected_faces = self.detect(frame)

        for face in detected_faces:
            location = face.get('location', None)
            person_name = face.get('name', None)

            if location:  # Check if location is not None
                top, right, bottom, left = location
            
                top *= self.cv_scaler
                right *= self.cv_scaler
                bottom *= self.cv_scaler
                left *= self.cv_scaler
                
                cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 1)
                cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, person_name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def attendance(self,
                    frame, 
                    detected_faces: list = None,
                    times_to_clear:list=[(12, 00), (18, 00)],
                    interval_minutes: int = None
                    ) -> list:
        """
        Plot the detected faces on the frame, showing known and unknown faces differently, and update the attendance data.
        
        Args:
            frame: The video frame where faces are to be plotted.
            detected_faces: A list containing detected faces, locations, and names (can be from the `detect` method).
        """
        if detected_faces is None:
            detected_faces = self.detect(frame)
        
        locations = []
        names = []
        pictures = []
        
        # Process detected faces for locations, names, and base64 images
        for face in detected_faces:
            location = face.get('location', None)
            person_name = face.get('name', "unknown")
            picture = face.get('face_base64', None)

            if location:  # Ensure that a location exists
                locations.append(location)
                names.append(person_name)
                pictures.append(picture)

        # Current timestamp for attendance data
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if the 'class_name' exists in the data
        for name in names:
            if name not in [entry['name'] for entry in self.attendance_data]:
                # Add new attendance record
                self.attendance_data.append({
                    "id": len(self.attendance_data) + 1,
                    "name": name,
                    "date": current_date,  # Only date part
                    "time": current_time,  # Only time part
                    #"picture": picture
                })

        # Check the time to clear unknown faces
        self.clear_unknown_faces(times_to_clear, interval_minutes)
    
        return self.attendance_data
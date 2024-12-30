import time
import cv2
from matplotlib import pyplot as plt
import supervision as sv
from ultralytics import YOLO
import numpy as np
import pandas as pd
import itertools
import os
import base64
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()


# Define the Tracking model for the trackings.db
class Tracking(Base):
    __tablename__ = 'trackings'

    id = Column(Integer, primary_key=True)
    person_id = Column(Integer)
    direction = Column(String(10))
    line_id = Column(Integer)
    timestamp = Column(Float)


class Tracker:
    PERSON_CLASS_ID = 0

    def __init__(self, model='yolov8s.pt', db='trackings.db'):

        self.tracking_db = f'sqlite:///{db}'
        self.engine_tracking = create_engine(self.tracking_db)
        Base.metadata.create_all(self.engine_tracking)  # Create the tracking table if not exists
        
        self.tracks = {}  # Dictionary to store track information {id: [(x, y)]}
        self.crossings = {"up": 0, "down": 0}  # To store crossing counts
        
        self.Session_tracking = sessionmaker(bind=self.engine_tracking)

        # Initialize the YOLO model
        self.model = YOLO(model)

        # Initialize the ByteTrack tracker
        self.tracker = sv.ByteTrack()

        # Initialize the BoxAnnotator and LabelAnnotator for drawing bounding boxes and labels
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # Class variables for tracked detections and center points
        self.tracked_detections = None
        self.center_points = None

        # Dictionary to store the movements of each person (person_id -> [list of (x, y)])
        self.movements = {}

    def process_frame(self, frame, selected_classes=None, track_movements_flag=False) -> np.ndarray:
        if selected_classes is None:
            selected_classes = [self.PERSON_CLASS_ID]

        # Run YOLOv8 inference on the frame
        results = self.model(frame)[0]

        # Convert YOLOv8 results to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter detections to include only selected classes
        detections = detections[np.isin(detections.class_id, selected_classes)]

        # Update tracker with filtered detections
        self.tracked_detections = self.tracker.update_with_detections(detections)

        # Prepare labels with tracker IDs
        labels = [f"Person ID: {tracker_id}" for tracker_id in self.tracked_detections.tracker_id]

        # Annotate the frame with bounding boxes and labels
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=self.tracked_detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=self.tracked_detections,
            labels=labels
        )

        # Add center points for each tracked object
        self.center_points = self.add_center_points(annotated_frame, self.tracked_detections)

        
        # Track movements of each person
        if track_movements_flag:
            self.track_movements()

        return annotated_frame

    def add_center_points(self, frame, tracked_detections) -> list:
        """
        Adds center points for each tracked object and returns their values with IDs.
        
        Args:
            frame (np.ndarray): The frame to draw the points on.
            tracked_detections (sv.Detections): Detections with tracking information.

        Returns:
            List[Tuple[int, Tuple[int, int]]]: A list of (tracker ID, center point) tuples.
        """
        center_points = []
        for detection, tracker_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            # Calculate center point
            x_min, y_min, x_max, y_max = detection
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            # Draw the center point on the frame
            cv2.circle(frame, (center_x, center_y), radius=5, color=(0, 255, 0), thickness=-1)

            # Add to the list with the corresponding ID
            center_points.append((tracker_id, (center_x, center_y)))

        return center_points

    def calculate_distances(
        self,
        frame_width=None,
        real_world_distance=None,
        point1=None,
        point2=None,
        pixel_to_meter_ratio=None,
        use_dynamic_scaling=True,
        distance_precision=4,  # Number of decimal places for distance rounding
    )-> pd.DataFrame:
        """
        Calculates the distances between each detected person using their center points.
        Returns a DataFrame with distances in meters, where rows and columns are tracker IDs.

        Args:
            frame_width (int): Width of the video frame in pixels (used for scaling calibration).
            frame_height (int): Height of the video frame in pixels (used for scaling calibration).
            real_world_distance (float): Real-world distance in meters between two known points.
            point1 (tuple): Pixel coordinates (x, y) of the first known point for calibration.
            point2 (tuple): Pixel coordinates (x, y) of the second known point for calibration.
            pixel_to_meter_ratio (float): Pre-calculated conversion factor from pixels to meters.
            use_dynamic_scaling (bool): Whether to calculate the scaling factor dynamically.
            distance_precision (int): The number of decimal places to round the distances to.

        Returns:
            pd.DataFrame: A DataFrame with tracker IDs as rows and columns, and distances in meters.
        """
        
        # Ensure center points are available
        if not self.center_points:
            raise ValueError("Center points are not available. Ensure the frame has been processed.")

        # Use default points (top-left and top-right corners) if not provided
        if not point1 or not point2:
            point1 = (0, 0)  # Top-left corner
            point2 = (frame_width, 0)  # Top-right corner

        # Dynamic pixel-to-meter ratio calculation
        if use_dynamic_scaling:
            if not all([real_world_distance, point1, point2]):
                raise ValueError("Dynamic scaling requires `real_world_distance`, `point1`, and `point2`.")

            # Calculate pixel distance between point1 and point2
            pixel_distance = np.linalg.norm(np.array(point1) - np.array(point2))

            # Derive the pixel-to-meter ratio
            pixel_to_meter_ratio = real_world_distance / pixel_distance

        if not pixel_to_meter_ratio:
            raise ValueError("Either provide `pixel_to_meter_ratio` or enable `use_dynamic_scaling` with calibration points.")

        # Extract tracker IDs and center points
        tracker_ids = [point[0] for point in self.center_points]
        centers = np.array([point[1] for point in self.center_points])

        # Create an empty DataFrame to store distances
        self.distance_df = pd.DataFrame(index=tracker_ids, columns=tracker_ids)

        # Efficiently calculate pairwise distances using itertools.combinations
        for (id1, center1), (id2, center2) in itertools.combinations(zip(tracker_ids, centers), 2):
            # Calculate Euclidean distance between the two center points
            distance = np.linalg.norm(np.array(center1) - np.array(center2))
            
            # Convert the distance from pixels to meters
            distance_meters = distance * pixel_to_meter_ratio
            
            # Round the distance to improve readability
            distance_meters = round(distance_meters, distance_precision)
            
            # Fill both [id1, id2] and [id2, id1] with the same value (symmetric)
            self.distance_df.at[id1, id2] = distance_meters
            self.distance_df.at[id2, id1] = distance_meters

        # Return the filled DataFrame with distances
        return self.self.distance_df

    def watch_distances(self, distances: pd.DataFrame, frame, min_distance=1.0, output_folder="output", diff=10,
                        save_image_path=True, save_base64=True) -> list:
        """
        Watches distances between detected persons and saves info if distance is below a minimum threshold.
        
        Args:
            distances (pd.DataFrame): DataFrame of distances between detected persons.
            frame (np.ndarray): The current frame being processed.
            min_distance (float): Minimum distance threshold in meters.
            output_folder (str): Folder to save cropped images of persons.
            diff (int): A small difference value added to bounding boxes to ensure both persons fit.
            save_image_path (bool): Whether to save the image to the folder. Default is True.
            save_base64 (bool): Whether to save the image encoded as base64. Default is True.

        Returns:
            List[Dict]: A list of dictionaries containing information about persons too close.
        """
        # Ensure output folder exists if saving image paths
        if save_image_path:
            os.makedirs(output_folder, exist_ok=True)

        close_persons = []
        for id1 in distances.index:
            for id2 in distances.columns:
                if id1 != id2 and distances.loc[id1, id2] < min_distance:
                    # Get the bounding boxes for both persons
                    idx1 = np.where(self.tracked_detections.tracker_id == id1)[0][0]
                    idx2 = np.where(self.tracked_detections.tracker_id == id2)[0][0]
                    bbox1 = self.tracked_detections.xyxy[idx1]
                    bbox2 = self.tracked_detections.xyxy[idx2]

                    # Calculate the combined bounding box
                    x_min1, y_min1, x_max1, y_max1 = map(int, bbox1)
                    x_min2, y_min2, x_max2, y_max2 = map(int, bbox2)

                    # Extend the bounding boxes slightly to fit both people
                    combined_x_min = min(x_min1, x_min2) - diff
                    combined_y_min = min(y_min1, y_min2) - diff
                    combined_x_max = max(x_max1, x_max2) + diff
                    combined_y_max = max(y_max1, y_max2) + diff

                    # Make sure the combined box is within the frame boundaries
                    combined_x_min = max(combined_x_min, 0)
                    combined_y_min = max(combined_y_min, 0)
                    combined_x_max = min(combined_x_max, frame.shape[1])
                    combined_y_max = min(combined_y_max, frame.shape[0])

                    # Crop the frame using the combined bounding box
                    combined_crop = frame[combined_y_min:combined_y_max, combined_x_min:combined_x_max]

                    # Save the image to the output folder if requested
                    image_path = None
                    if save_image_path:
                        image_path = os.path.join(output_folder, f"{id1}_and_{id2}.jpg")
                        cv2.imwrite(image_path, combined_crop)

                    # Encode the combined image to base64 if requested
                    image_base64 = None
                    if save_base64:
                        _, buffer = cv2.imencode('.jpg', combined_crop)
                        image_base64 = base64.b64encode(buffer).decode("utf-8")

                    # Add the information to the dictionary
                    close_persons.append({
                        "person1": id1,
                        "person2": id2,
                        "distance": distances.loc[id1, id2],
                        "img_path": image_path if save_image_path else None,
                        "image_encoded_base": image_base64 if save_base64 else None
                    })

        return close_persons

    def track_movements(self) -> None:
        """
        Track the movements of each person by storing their center point coordinates (x, y)
        for each frame.

        Updates the `self.movements` dictionary where each key is a person_id and the value is a list
        of (x, y) tuples representing the center points of that person across frames.
        """
        for tracker_id, center_point in self.center_points:
            if tracker_id not in self.movements:
                # If the person is new, initialize their movement list
                self.movements[tracker_id] = []

            # Append the current center point (x, y) to the movement list for this person
            self.movements[tracker_id].append(center_point)

    def get_movements(self) -> dict:
        """
        Get the current movements dictionary which holds the center point history for each person.

        Returns:
            dict: A dictionary with person_ids as keys and a list of (x, y) center points as values.
        """

        return self.movements
    
    def track_time(self, frame_rate) -> dict:
        """
        Track the time spent by each person in the frame.

        Args:
            frame_rate (int): The frame rate of the video.

        Returns:
            dict: A dictionary with person_ids as keys and the time spent in seconds as values.
        """
        time_spent = {}
        for person_id, movement in self.movements.items():
            # Calculate the time spent by the person in the frame
            time_spent[person_id] = len(movement) / frame_rate

        return time_spent
    
    def plot_movement_trace(self, selected_persons=['all'], frame_width=1920, frame_height=1080, return_base64_image=False):
        """
        Plot the movement traces of selected persons on a white canvas background and return 
        the image as base64 if requested.

        Args:
            selected_persons (list): List of person IDs to plot traces for. 
                                      Use ['all'] to plot traces for all persons.
            frame_width (int): Width of the video frame.
            frame_height (int): Height of the video frame.
            return_base64_image (bool): If True, return the plot as a base64-encoded image.

        Returns:
            str or None: If return_base64_image is True, return the base64-encoded image as a string.
                         Otherwise, return None.
        """
        frame_width = 1920
        frame_height = 1080
        border_width = 5  # Width of the black border

        # Create a white canvas (background)
        canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # White canvas

        # Add a black border around the canvas
        canvas[:border_width, :] = 0  # Top border
        canvas[-border_width:, :] = 0  # Bottom border
        canvas[:, :border_width] = 0  # Left border
        canvas[:, -border_width:] = 0  # Right border

        # If 'all' is in the selected_persons list, select all persons
        if 'all' in selected_persons:
            selected_persons = list(self.movements.keys())

        # Plot the traces for the selected persons
        for person_id in selected_persons:
            if person_id in self.movements:
                movement_trace = self.movements[person_id]

                # Draw the trace
                for i in range(1, len(movement_trace)):
                    start_point = movement_trace[i - 1]
                    end_point = movement_trace[i]
                    cv2.line(canvas, start_point, end_point, (0, 0, 255), 2)  # Red color for trace

        # If base64 encoding is required, return the image as base64
        if return_base64_image:
            # Convert the canvas to a byte array
            _, img_bytes = cv2.imencode('.png', canvas)
            img_bytes = img_bytes.tobytes()

            # Encode the byte array to base64
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            return base64_image

        # Otherwise, show the plot using OpenCV or Matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(canvas)
        plt.title('Movement Trace')
        plt.axis('off')  # Hide axis for better view
        plt.show()
        return None
    
    def count_crossings(self, frame, line_id=1, drawing_db=None, db_model=None, line_direction='horizontal', plot=True) -> tuple:
        """
        Count how many people crossed a specified line and in which direction.
        """
        # Fetch line coordinates from the database
        if drawing_db:
            engine_drawing = create_engine(f'sqlite:///{drawing_db}')
            Session_drawing = sessionmaker(bind=engine_drawing)
            session = Session_drawing()
            line_coords = session.query(db_model).filter_by(id=line_id).first()
            session.close()

            if not line_coords:
                raise ValueError("Line ID not found in the database.")

            x1, y1, x2, y2 = line_coords.a, line_coords.b, line_coords.c, line_coords.d
        else:
            raise ValueError("Database for drawing lines is required.")

        # Draw the line on the frame
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), thickness=2)

        # Check each tracked person for crossing
        for tracker_id, movement in self.movements.items():
            if len(movement) >= 2:  # Ensure enough points for direction calculation
                start_point = movement[-2]
                end_point = movement[-1]
                direction = None

                if line_direction == 'horizontal':
                    # Crossing logic for horizontal line
                    if start_point[1] < y1 <= end_point[1]:
                        self.crossings['down'] += 1
                        direction = 'down'
                    elif start_point[1] > y1 >= end_point[1]:
                        self.crossings['up'] += 1
                        direction = 'up'
                elif line_direction == 'vertical':
                    # Crossing logic for vertical line
                    if start_point[0] < x1 <= end_point[0]:
                        self.crossings['right'] += 1
                        direction = 'right'
                    elif start_point[0] > x1 >= end_point[0]:
                        self.crossings['left'] += 1
                        direction = 'left'
                # save the crossing information to the database
                if direction:
                    session = self.Session_tracking()
                    tracking = Tracking(person_id=int(tracker_id), direction=direction, line_id=line_id, timestamp=time.time())
                    session.add(tracking)
                    session.commit()
                    session.close()
                    direction = None

        # plot the crossings on the frame
        if plot:
            cv2.putText(frame, f"Up: {self.crossings['up']}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Down: {self.crossings['down']}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame
    
    def load_crossings_from_db(self):
        """
        Load the crossings records from the database, count 'up' and 'down' directions,
        and return the records as a list of dictionaries along with the counts.
        """
        # Create a new database session
        db_session = self.Session_tracking()

        # Query all detected objects from the database
        detected_objects_db = db_session.query(Tracking).all()

        # Convert the results to a list of dictionaries
        detected_objects = []
        up_count = 0
        down_count = 0
        for obj in detected_objects_db:
            detected_obj = {
                'person_id': obj.person_id,
                'direction': obj.direction,
                'timestamp': obj.timestamp,
                'img_base64': base64.b64encode(obj.img_base64).decode('utf-8')  # Convert binary image back to base64
            }
            detected_objects.append(detected_obj)

            # Count 'up' and 'down' directions
            if obj.direction == 'up':
                up_count += 1
            elif obj.direction == 'down':
                down_count += 1

        # Close the database session
        db_session.close()

        # Return the list of detected objects along with the counts
        return detected_objects, up_count, down_count

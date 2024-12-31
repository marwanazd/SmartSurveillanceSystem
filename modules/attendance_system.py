import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_
from datetime import datetime, timedelta
import csv
import json
import pandas as pd

Base = declarative_base()

class AttendanceRecord(Base):
    __tablename__ = 'attendance'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    face_image = Column(String, nullable=True)  # To store the face image as Base64

class AttendanceTracker:
    def __init__(self, db_path, config=None):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Set the base path for the web app (cross-platform)
        self.base_path = os.path.abspath(os.getcwd())
        
        if config is None:
            # Default config with the 'uploads' folder inside the web app's directory
            config = {'UPLOAD_FOLDER': os.path.join(self.base_path, 'uploads')}
        self.config = config

        # Ensure the upload folder exists
        self.ensure_upload_folder_exists()

    def ensure_upload_folder_exists(self):
        """Ensure the upload folder exists."""
        upload_folder = self.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_folder):
            try:
                os.makedirs(upload_folder)
                print(f"Created upload folder: {upload_folder}")
            except Exception as e:
                print(f"Error creating upload folder: {e}")

    def record_attendance(self, recognized_faces):
        """
        Records attendance for recognized faces.
        - recognized_faces: List of dictionaries, each containing `name`, `face_base64`, and other data.
        """
        if not os.path.exists(self.db_path):
            print(f"Database created at: {self.db_path}")

        session = self.Session()

        try:
            for face in recognized_faces:
                name = face.get("name")
                face_image = face.get("face_base64")
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                current_time = datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')

                # Check if an entry for this name already exists for today
                existing_record = session.query(AttendanceRecord).filter(
                    AttendanceRecord.name == name,
                    AttendanceRecord.timestamp >= current_time.replace(hour=0, minute=0, second=0, microsecond=0),
                    AttendanceRecord.timestamp <= current_time.replace(hour=23, minute=59, second=59, microsecond=999999)
                ).first()

                if existing_record:
                    #print(f"Attendance already recorded for {name} today.")
                    continue

                # Create a new attendance record
                new_record = AttendanceRecord(
                    name=name,
                    timestamp=current_time,
                    face_image=face_image
                )
                session.add(new_record)
                print(f"Attendance recorded for {name} at {current_time}.")

            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error recording attendance: {e}")
        finally:
            session.close()

    def get_attendance_records(self, time_period='all', start_date=None, end_date=None):
        """
        Retrieves attendance records for a specific time period.
        - time_period: A string representing the time period ('all', 'today', 'yesterday', 'this week', 'this month').
        - start_date: Optional start date (used when time_period is 'specific').
        - end_date: Optional end date (used when time_period is 'specific').
        Returns a list of dictionaries containing attendance records.
        """
        session = self.Session()
        records = []

        try:
            current_time = datetime.now()

            # Determine date range based on time_period
            if time_period == 'today':
                start_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = current_time.replace(hour=23, minute=59, second=59, microsecond=999999)

            elif time_period == 'yesterday':
                # Yesterday, full day from 00:00:00 to 23:59:59
                start_date = (current_time - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            elif time_period == 'week':
                # Start of the week (Monday)
                start_date = current_time - timedelta(days=current_time.weekday())  # Monday
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                # End of the week (Sunday)
                end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59, microseconds=999999)

            elif time_period == 'month':
                start_date = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                # Get the last day of the month
                next_month = start_date.replace(month=start_date.month + 1) if start_date.month < 12 else start_date.replace(year=start_date.year + 1, month=1)
                end_date = next_month - timedelta(seconds=1)
                end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

            elif time_period == 'specific':
                if not start_date or not end_date:
                    print("Please provide both start_date and end_date.")
                    return []
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # Include the end date

            elif time_period == 'all':
                # No filtering by date
                records_query = session.query(AttendanceRecord).all()
                for record in records_query:
                    records.append({
                        'id': record.id,
                        'name': record.name,
                        'timestamp': record.timestamp,
                        'face_image': record.face_image
                    })
                return records

            # Query attendance records based on the determined date range
            records_query = session.query(AttendanceRecord).filter(
                and_(
                    AttendanceRecord.timestamp >= start_date,
                    AttendanceRecord.timestamp <= end_date
                )
            ).all()

            # Convert to list of dictionaries
            for record in records_query:
                records.append({
                    'id': record.id,
                    'name': record.name,
                    'timestamp': record.timestamp,
                    'face_image': record.face_image
                })

            return records

        except Exception as e:
            print(f"Error retrieving attendance records: {e}")
            return []

        finally:
            session.close()

    def save_attendance_records(self, time_period='all', start_date=None, end_date=None, file_format='csv', file_path='attendance_records'):
        """
        Retrieves attendance records and saves them to the specified file format.
        - time_period: Time period for which records are to be retrieved (e.g., 'all', 'today', 'yesterday', etc.).
        - start_date: Start date for 'specific' time period (in 'YYYY-MM-DD' format).
        - end_date: End date for 'specific' time period (in 'YYYY-MM-DD' format).
        - file_format: Desired file format to save records ('csv', 'xls', 'json').
        - file_path: Path where the file will be saved (without extension).
        """
        # Get the records using the previously defined method
        records = self.get_attendance_records(time_period, start_date, end_date)

        if not records:
            print("No attendance records found.")
            return None

        try:
            # Determine the full file path for the saved file
            full_file_path = None

            if file_format.lower() == 'csv':
                full_file_path = self.save_as_csv(records, file_path)
            
            elif file_format.lower() == 'xls':
                full_file_path = self.save_as_excel(records, file_path)
            
            elif file_format.lower() == 'json':
                full_file_path = self.save_as_json(records, file_path)
            
            else:
                print(f"Unsupported file format: {file_format}")
            
            return full_file_path
        
        except Exception as e:
            print(f"Error saving attendance records: {e}")
            return None

    def save_as_csv(self, records, file_path):
        """Saves the records to a CSV file and returns the full file path."""
        try:
            file_name = f"{file_path}.csv"
            full_file_path = os.path.join(self.config['UPLOAD_FOLDER'], file_name)
            
            with open(full_file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['id', 'name', 'timestamp', 'face_image'])
                writer.writeheader()
                for record in records:
                    writer.writerow(record)
            
            print(f"Records saved successfully to {full_file_path}")
            return full_file_path
        
        except Exception as e:
            print(f"Error saving records as CSV: {e}")
            return None

    def save_as_excel(self, records, file_path):
        """Saves the records to an Excel file and returns the full file path."""
        try:
            file_name = f"{file_path}.xlsx"
            full_file_path = os.path.join(self.config['UPLOAD_FOLDER'], file_name)
            
            # Convert list of records to DataFrame
            df = pd.DataFrame(records)
            df.to_excel(full_file_path, index=False)
            
            print(f"Records saved successfully to {full_file_path}")
            return full_file_path
        
        except Exception as e:
            print(f"Error saving records as Excel: {e}")
            return None

    def save_as_json(self, records, file_path):
        """Saves the records to a JSON file and returns the full file path."""
        try:
            file_name = f"{file_path}.json"
            full_file_path = os.path.join(self.config['UPLOAD_FOLDER'], file_name)
            
            with open(full_file_path, 'w') as file:
                json.dump(records, file, default=str, indent=4)  # default=str to handle datetime formatting
            
            print(f"Records saved successfully to {full_file_path}")
            return full_file_path
        
        except Exception as e:
            print(f"Error saving records as JSON: {e}")
            return None
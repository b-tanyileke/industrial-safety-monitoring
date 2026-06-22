"""
This file defines the IncidentLogger class, which is responsible for logging safety incidents.
The logger creates a new log file if it doesn't exist and appends new incident records 
with details such as timestamp, person ID, violation type, status, and image path.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import cv2


class IncidentLogger:
    """
    Logs safety incidents to a CSV file with details.
    """

    def __init__(self, log_file="logs/incidents.csv", incident_dir="logs/images"):

        self.log_file = Path(log_file)
        self.incident_dir = Path(incident_dir)
        self.active_incidents = set()

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create directory for incident images if it doesn't exist
        self.incident_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with headers if it doesn't exist
        if not self.log_file.exists():

            df = pd.DataFrame(
                columns=[
                    "timestamp",
                    "person_id",
                    "violation_type",
                    "status",
                    "image_path"
                ]
            )

            df.to_csv(
                self.log_file,
                index=False
            )

    # Define a method to save an image related to the incident and return the file path
    def save_incident_image(self, frame, person_id, violation_type):
        """Saves an image related to the incident and returns the file path.
        Args:
            frame (numpy.ndarray): The image frame to save.
            person_id (int): The ID of the person involved in the incident.
            violation_type (str): The type of violation (e.g., "No Helmet", "No Vest", "Restricted Zone").
        Returns:
            str: The file path to the saved image.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = (f"{timestamp}_"f"{person_id}_"f"{violation_type}.jpg")

        path = self.incident_dir / filename

        saved = cv2.imwrite(str(path), frame)

        if not saved:
            raise RuntimeError(f"Could not save incident image: {path}")

        return str(path)

    def log(self, person_id, violation_type, frame=None):
        """
        Logs a safety incident to the CSV file.
        Args:
            person_id (int): The ID of the person involved in the incident.
            violation_type (str): The type of violation (e.g., "No Helmet", "No Vest", "Restricted Zone").
            frame (numpy.ndarray, optional): The image frame related to the incident. Defaults to None.
        Returns:
            None
        """

        # Use a combination of person_id and violation_type to prevent duplicate logging of the same incident
        incident_key = (person_id, violation_type)

        # If this incident has already been logged as active, skip logging it again
        if incident_key in self.active_incidents:
            return

        # Otherwise, log the new incident and mark it as active
        self.active_incidents.add(incident_key)

        image_path = ""

        # If a frame is provided, save the incident image and get the file path
        if frame is not None:
            image_path = self.save_incident_image(frame, person_id, violation_type)

        # Load existing incident data from the CSV log file
        df = pd.read_csv(self.log_file)

        new_row = {
            "timestamp": datetime.now(),
            "person_id": person_id,
            "violation_type": violation_type,
            "status": "Active",
            "image_path": image_path
        }
        # Append new incident record to the DataFrame and save it back to the CSV file
        df.loc[len(df)] = new_row

        df.to_csv(
            self.log_file,
            index=False
        )

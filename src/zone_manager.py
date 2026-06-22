"""
This file defines the ZoneManager class, which is responsible for managing a defined zone in the video frame 
and determining whether detected persons are inside that zone. 
The class uses OpenCV to define the zone as a polygon 
and checks if the center of a detected person's bounding box is within that polygon.
"""
import cv2
import numpy as np


class ZoneManager:
    """
    Manages a defined zone in the video frame and checks if detected persons are inside that zone.
    """

    def __init__(self, zone=None):
        if zone is None:
            zone = [
                [100, 100],
                [500, 100],
                [500, 500],
                [100, 500]
            ]

        self.zone = np.array(zone)

    def is_inside_zone(self, person_box):
        """Determines if the center of the person's bounding box is inside the defined zone.
        Args:
            person_box (list): A list containing the coordinates of the person's bounding box [x1, y1, x2, y2].
        Returns:
            bool: True if the center of the bounding box is inside the zone, False otherwise.
        """

        if len(self.zone) == 0:
            return False

        x1, y1, x2, y2 = person_box

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        result = cv2.pointPolygonTest(
            self.zone,
            (center_x, center_y),
            False
        )

        return result >= 0

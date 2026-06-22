"""
This module contains the DashboardRenderer class, which is responsible for 
rendering the dashboard view of the safety monitoring system. 
It provides methods to draw zones, detections, and violations on the video frames.
"""
import cv2


class DashboardRenderer:
    """
    Renders the dashboard view of the safety monitoring system.
    """

    def draw(self, frame, detections, violations, zone_manager):
        """
        Draws the dashboard elements on the video frame, including zones, detections, and violations.
        Args:
            frame (numpy.ndarray): The video frame to draw on.
            detections (list): A list of detected persons and their information.
            violations (list): A list of detected violations and their information.
            zone_manager (ZoneManager): The zone manager instance.
        Returns:
            frame (numpy.ndarray): The video frame with dashboard elements drawn on it.
        """
        if len(zone_manager.zone) > 0:
            # Draw zone
            cv2.polylines(
                frame,
                [zone_manager.zone],
                True,
                (0, 0, 255),
                2
            )

        # Draw detections
        for detection in detections:
            person_id = detection.get("person_id",-1)

            person_x1, person_y1, person_x2, person_y2 = detection["person_box"]
            # Draw person bounding box
            cv2.rectangle(
                frame,
                (person_x1, person_y1),
                (person_x2, person_y2),
                (0,255,0),
                2
            )
            # Draw label for person
            cv2.putText(
                frame,
                f"Person {person_id}",
                (person_x1, person_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

            for ppe in detection["ppe"]:
                # Extract the PPE bounding box coordinates
                crop_x1, crop_y1, crop_x2, crop_y2 = ppe["box"]

                # Convert PPE box coordinates from crop space to original frame space
                global_x1 = person_x1 + crop_x1
                global_y1 = person_y1 + crop_y1

                global_x2 = person_x1 + crop_x2
                global_y2 = person_y1 + crop_y2

                if ppe["class"] in ["no helmet", "no vest"]:
                    color = (0,0,255)
                else:
                    color = (255,0,0)

                # Draw PPE bounding box
                cv2.rectangle(
                    frame,
                    (global_x1, global_y1),
                    (global_x2, global_y2),
                    color,
                    2
                )
                # Draw label for PPE
                cv2.putText(
                    frame,
                    ppe["class"],
                    (global_x1, global_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # Draw violations

        y = 30 

        for violation in violations:

            cv2.putText(
                frame,
                violation["type"],
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,0,255),
                2
            )

            y += 35

        return frame

""" PersonTracker class for tracking detected people in video frames. """


import supervision as sv
import numpy as np


class PersonTracker:
    """Tracks detected people across video frames using ByteTrack algorithm. """

    def __init__(self):

        self.tracker = sv.ByteTrack()

    def update(self, detections):
        """Updates the tracker with new detections and assigns unique IDs to each person.
        Args:
            detections (list): A list of dictionaries containing detection information for each person.
        Returns:
            list: A list of dictionaries containing the updated detection information with unique IDs.
        """

        if len(detections) == 0:
            return []

        xyxy = []
        confidence = []

        for det in detections:

            xyxy.append(det["person_box"])
            confidence.append(det["person_confidence"])

        detections_sv = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.zeros(len(xyxy), dtype=int)
        )

        tracked = self.tracker.update_with_detections(detections_sv)

        results = []

        for det, track_id in zip(detections, tracked.tracker_id):

            det["person_id"] = int(track_id)

            results.append(det)

        return results

"""
SafetyDetector class for detecting people and their PPE in video frames.
"""

from pathlib import Path

from ultralytics import YOLO


class SafetyDetector:
    """
    Detects people and their PPE in video frames using YOLO models.
    """

    def __init__(
        self,
        person_model_path="yolo11n.pt",
        ppe_model_path="models/ppe_yolo.pt",
        person_confidence=0.6,
        ppe_confidence=0.6
    ):

        self.person_model_path = Path(person_model_path)
        self.ppe_model_path = Path(ppe_model_path)
        self.person_confidence = person_confidence
        self.ppe_confidence = ppe_confidence

        if not self.person_model_path.exists():
            raise FileNotFoundError(
                f"Person model not found: {self.person_model_path}"
            )

        if not self.ppe_model_path.exists():
            raise FileNotFoundError(
                f"PPE model not found: {self.ppe_model_path}"
            )

        self.person_model = YOLO(str(self.person_model_path))
        self.ppe_model = YOLO(str(self.ppe_model_path))

    def detect(self, frame):
        """Detects people and their PPE in the given video frame.
        Args:
            frame (numpy.ndarray): The input video frame in which to detect people and PPE.
        Returns:
            list: A list of dictionaries containing detection information for each person.
        """

        detections = []

        person_results = self.person_model(
            frame,
            conf=self.person_confidence,
            verbose=False
        )

        # Loop through detected people and check for PPE
        for box in person_results[0].boxes:
            cls_id = int(box.cls[0])

            # COCO person class
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Crop the detected person from the frame for PPE detection
            crop = frame[y1:y2, x1:x2]
            # Skip PPE detection if the crop is empty (e.g., due to incorrect bounding box)
            if crop.size == 0:
                continue

            # Run PPE detection on the cropped person image
            ppe_results = self.ppe_model(
                crop,
                conf=self.ppe_confidence,
                verbose=False
            )

            ppe_items = []
            # Loop through detected PPE items and store their information
            for ppe_box in ppe_results[0].boxes:
                ppe_cls = int(ppe_box.cls[0])

                # Extract the PPE bounding box coordinates 
                crop_x1, crop_y1, crop_x2, crop_y2 = map(int, ppe_box.xyxy[0])

                ppe_items.append({
                    "class": self.ppe_model.names[ppe_cls],
                    "confidence": float(ppe_box.conf[0]),
                    "box": [crop_x1, crop_y1, crop_x2, crop_y2]
                })

            detections.append({
                "person_box": [x1, y1, x2, y2],
                "person_confidence": float(box.conf[0]),
                "ppe": ppe_items
            })

        return detections

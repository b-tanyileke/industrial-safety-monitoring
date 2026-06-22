"""
Video processing module for safety monitoring system. This module handles video input, 
applies detection and tracking algorithms, evaluates safety violations, 
and renders the output with annotations and a dashboard.
"""

from src.config import AppConfig
from src.detector import SafetyDetector
from src.tracker import PersonTracker
from src.zone_manager import ZoneManager
from src.violation_engine import ViolationEngine
from src.dashboard import DashboardRenderer
from src.logger import IncidentLogger


class VideoProcessor:
    """Processes video frames for safety monitoring"""

    def __init__(self, config=None):
        if config is None:
            config = AppConfig()

        self.config = config

        self.detector = SafetyDetector(
            person_model_path=config.person_model_path,
            ppe_model_path=config.ppe_model_path,
            person_confidence=config.person_confidence,
            ppe_confidence=config.ppe_confidence
        )

        self.tracker = PersonTracker()

        self.zone_manager = ZoneManager(config.restricted_zone)

        self.engine = ViolationEngine()

        self.renderer = DashboardRenderer()

        self.logger = IncidentLogger(
            log_file=config.incident_log_path,
            incident_dir=config.incident_image_dir
        )

    def process_frame(self, frame):
        """
        Processes a single video frame for safety monitoring.
        Args:
            frame (numpy.ndarray): The input video frame to process.
        Returns:
            numpy.ndarray: The processed video frame with annotations and dashboard.
        """

        detections = self.detector.detect(frame)

        detections = self.tracker.update(detections)

        violations = self.engine.evaluate(
            detections,
            self.zone_manager
        )

        for violation in violations:

            self.logger.log(
                person_id=violation["person_id"],
                violation_type=violation["type"],
                frame=frame
            )

        frame = self.renderer.draw(
            frame,
            detections,
            violations,
            self.zone_manager
        )

        return frame

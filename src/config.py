"""
Shared application configuration for the safety monitoring system.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """Configuration class for the safety monitoring application."""
    person_model_path: Path = Path("yolo26s.pt")
    ppe_model_path: Path = Path("runs/detect/train-3/weights/best.pt")
    person_confidence: float = 0.6
    ppe_confidence: float = 0.6
    incident_log_path: Path = Path("logs/incidents.csv")
    incident_image_dir: Path = Path("logs/images")
    output_video_path: Path = Path("data/output/processed_video.mp4")
    restricted_zone: tuple = ()

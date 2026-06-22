"""
This module serves as the entry point for the Industrial Safety Monitoring application. 
It initializes the necessary components, processes video input, detects safety violations, and renders the output to a dashboard.
The application can be run in different modes (webcam or video file) and saves processed video output while logging incidents for review.
Usage:
    python app.py --mode webcam
    python app.py --mode video --source path/to/video.mp4
"""

import os
import argparse
from pathlib import Path
import cv2

from src.config import AppConfig
from src.video_processor import VideoProcessor


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    """
    Parses command-line arguments for video processing mode and source.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["webcam", "video"], default="webcam")
    parser.add_argument("--source", default=0)

    return parser.parse_args()


def get_video_source(args):
    """
    Determines the video source based on command-line arguments.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        int or str: Video source (webcam index or video file path).
    Raises:
        FileNotFoundError: If the specified video file does not exist.
    """
    if args.mode == "webcam":
        return int(args.source)

    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(f"Video source not found: {source}")

    return str(source)


def build_writer(output_path, fps, frame_width, frame_height):
    """
    Builds a video writer for saving processed video output.
    Args:
        output_path (str or Path): The file path to save the output video.
        fps (float): Frames per second for the output video.
        frame_width (int): Width of the video frames.
        frame_height (int): Height of the video frames.
    Returns:
        cv2.VideoWriter: An OpenCV VideoWriter object for saving video.
    Raises:
        RuntimeError: If the VideoWriter cannot be created.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    return writer


def main():
    args = parse_args()
    config = AppConfig()
    source = get_video_source(args)
    processor = VideoProcessor(config)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Video source returned invalid frame dimensions.")

    out = build_writer(
        config.output_video_path,
        fps,
        frame_width,
        frame_height
    )

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = processor.process_frame(frame)

            out.write(frame)

            cv2.imshow("Industrial Safety Monitor", frame)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")

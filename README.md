# Industrial Safety Monitoring

Computer vision prototype for detecting basic PPE and safety violations in webcam stream or video footage. The app detects people, checks each person crop for PPE, tracks people across frames, logs incidents, and writes an annotated output video.

## What It Does

- Detects people with a YOLO person model.
- Detects PPE with a trained YOLO PPE model.
- Tracks people across frames with ByteTrack.
- Flags missing helmet and missing vest violations.
- Optionally checks whether a person is inside a restricted zone.
- Saves incident records to `logs/incidents.csv`.
- Saves incident images to `logs/images/`.
- Provides a Streamlit dashboard for reviewing logged incidents.

## Project Structure

```text
app.py                  OpenCV runner for webcam/video processing
streamlit_app.py        Incident review dashboard
src/config.py           Central paths, confidence thresholds, and zone config
src/detector.py         YOLO person/PPE detection
src/tracker.py          ByteTrack person tracking
src/violation_engine.py Safety rule checks
src/zone_manager.py     Restricted-zone checks
src/logger.py           CSV and incident-image logging
src/dashboard.py        Frame annotation rendering
src/video_processor.py  Per-frame processing pipeline
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Model Files

The current defaults are defined in `src/config.py`:

```python
person_model_path = Path("yolo26s.pt")
ppe_model_path = Path("runs/detect/train-3/weights/best.pt")
person_confidence = 0.6
ppe_confidence = 0.6
```

Those model files must exist locally before running the app. Model and training artifacts are ignored by git by default because they are large and environment-specific.

## Run The Monitor

Use a webcam:

```bash
python app.py --mode webcam --source 0
```

Use a video file:

```bash
python app.py --mode video --source data/videos/example.mp4
```

Press `q` in the OpenCV window to stop processing. The processed video is written to:

```text
data/output/processed_video.mp4
```

## Run The Dashboard

```bash
streamlit run streamlit_app.py
```

If no incidents have been logged yet, the dashboard shows empty metrics and an empty table.

## Sample Output Video

<video width="640" height="480" controls>
  <source src="data/output/processed_video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Configuration

Most local settings live in `src/config.py`:

- `person_model_path`
- `ppe_model_path`
- `person_confidence`
- `ppe_confidence`
- `incident_log_path`
- `incident_image_dir`
- `output_video_path`
- `restricted_zone`

Restricted-zone checks are currently disabled:

```python
restricted_zone: tuple = ()
```

To enable them, set `restricted_zone` to polygon points:

```python
restricted_zone: tuple = (
    (100, 100),
    (500, 100),
    (500, 500),
    (100, 500),
)
```

## Notes

This project is intentionally kept simple: plain Python modules, small classes, and direct OpenCV/YOLO calls. `app.py` handles input/output, while `VideoProcessor` owns the per-frame detection, tracking, violation, logging, and rendering pipeline.


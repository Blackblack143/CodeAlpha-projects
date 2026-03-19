# Real-Time Object Detection and Tracking (YOLOv8 + OpenCV + SORT)

## Project Overview

This project performs **real-time object detection and tracking** using
a webcam or video file.

It uses: - **YOLOv8** for object detection - **OpenCV** for video
processing - **Simple SORT tracker** for assigning unique IDs to
objects - **CSV logging** to store detected object data

The program detects objects in each video frame, assigns them a
**tracking ID**, draws bounding boxes, and saves detection results into
a **CSV file**.

------------------------------------------------------------------------

## Features

-   Real-time webcam video input
-   Pretrained YOLOv8 object detection model
-   Object tracking with unique IDs
-   Bounding boxes with labels
-   Detection logging to CSV

------------------------------------------------------------------------

## Installation

Install required libraries:

``` bash
pip install opencv-python ultralytics filterpy pandas
```

------------------------------------------------------------------------

## Project Structure

    project/
    │
    ├── object_tracking.py
    ├── detections_log.csv
    └── README.md

------------------------------------------------------------------------

## CSV File Format

The CSV file stores tracking results for every frame.

  frame   object_id   x1   y1   x2   y2
  ------- ----------- ---- ---- ---- ----

Example:

  frame   object_id   x1    y1    x2    y2
  ------- ----------- ----- ----- ----- -----
  1       0           120   200   250   400
  1       1           420   210   510   390

------------------------------------------------------------------------

## How CSV is Connected to the Code

Add the following lines to your script.

### 1. Import pandas

``` python
import pandas as pd
```

### 2. Create a list before the loop

``` python
data = []
frame_id = 0
```

### 3. Update data inside the tracking loop

``` python
frame_id += 1

data.append({
    "frame": frame_id,
    "object_id": track_id,
    "x1": x1,
    "y1": y1,
    "x2": x2,
    "y2": y2
})
```

### 4. Save CSV when program ends

``` python
df = pd.DataFrame(data)
df.to_csv("detections_log.csv", index=False)
```

------------------------------------------------------------------------

## Running the Project

Run the script:

``` bash
python object_tracking.py
```

Press **Q** to quit the video window.

------------------------------------------------------------------------

## Applications

-   Smart surveillance systems
-   Traffic monitoring
-   Retail analytics
-   Autonomous driving
-   Crowd tracking

------------------------------------------------------------------------

## Future Improvements

-   Use **DeepSORT instead of simple SORT**
-   Add **object class labels**
-   Add **object counting**
-   Add **speed estimation**
-   Deploy using **Streamlit or Flask**

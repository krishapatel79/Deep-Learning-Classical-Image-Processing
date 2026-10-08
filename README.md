# CV_PR3 — Classical Image Processing to Deep-Learning Object Detection

## 1. Project Overview

This project is Deep Learning PR 3 for Red & White Skill Education. It builds a single
Jupyter Notebook (`CV_PR3.ipynb`) that moves step by step from classical, hand-engineered
image-processing operations to modern deep-learning-based detection, all within OpenCV's
ecosystem: morphological operations, bitwise operations, image histograms, face detection
with YuNet, object detection with YOLOv8, and a final integrated real-time pipeline that
combines both detectors.

## 2. Objectives

- Understand and apply the four core morphological operations and how kernel shape/size
  affects noise removal.
- Use bitwise logic to combine masks and build a masked-overlay/watermark example.
- Read and interpret grayscale and colour histograms, and understand the effect of
  brightness/contrast adjustment on them.
- Load and run OpenCV 5.0's YuNet face detector on static images and a live webcam feed.
- Load and run Ultralytics YOLOv8n to detect COCO-class objects, and study the effect of
  confidence/IoU thresholds and Non-Max Suppression.
- Combine both detectors into one real-time pipeline and benchmark its FPS against each
  detector running alone.

## 3. Technologies / Tools

- Python 3.12
- OpenCV (`opencv-python`, `opencv-contrib-python`) 5.0.0
- Ultralytics YOLOv8
- NumPy
- Matplotlib
- Pandas (used only to build and display the final comparison table)
- Jupyter Notebook

## 4. Dataset / Sample Image Sources

Static test images were sourced from OpenCV's own sample data repository
(`https://github.com/opencv/opencv/tree/master/samples/data`, BSD-3-Clause licence) and
from Ultralytics' official demo images
(`https://github.com/ultralytics/assets`), used for Task 1 (morphology), Task 2 (bitwise,
histograms), Task 3 (face detection), and Task 4 (object detection):

| File | Used for |
|---|---|
| `baboon.jpg` | Morphology (grayscale/binary source, textured image) |
| `lena.jpg` | Binarization, bitwise masked overlay, brightness/contrast |
| `fruits.jpg` | Watermark compositing background, YOLO detection |
| `opencv-logo.png` | Watermark logo |
| `messi5.jpg` | YuNet face detection (varied pose) |
| `messi5_occluded.jpg` | YuNet face detection (partial occlusion — see note below) |
| `zidane.jpg` | YuNet face detection (multiple people, varied pose), YOLO threshold tuning |
| `bus.jpg` | YOLO object detection, morphological pre-cleaning demo |
| `smarties.png`, `stuff.jpg`, `pic3.png`, `board.jpg`, `basketball1.png`, `HappyFish.jpg`, `starry_night.jpg` | Downloaded as part of the general sample pack |

**Note on the occluded face image:** a genuinely photographed face with sunglasses could not
be sourced from any domain reachable inside the build sandbox (see Section 9). Instead,
`messi5_occluded.jpg` was generated programmatically from `messi5.jpg` by drawing a solid
black rectangle over the lower half of the face, to create a controlled, reproducible partial
occlusion test case. The code that generates it is in `scripts/build_notebook.py` context
(image prep step) and can be regenerated at any time.

## 5. YuNet Model Source

`face_detection_yunet_2023mar.onnx`, from the OpenCV Zoo:
`https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet`

**Important:** this file is **not included** in this repository/build, and Task 3's cells
detect this and skip gracefully rather than failing or faking results. See Section 9 for why,
and the exact command to fetch it yourself.

## 6. YOLO Model Information

`yolov8n.pt` — Ultralytics YOLOv8 Nano, pretrained on the 80-class COCO dataset. It
auto-downloads on first use via:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

Licence: AGPL-3.0 (pretrained weights are free for academic/educational use).

## 7. Folder Structure

```
project_folder/
├── CV_PR3.ipynb
├── CV_PR3.html
├── README.md
├── requirements.txt
│
├── data/
│   ├── images/
│   └── models/
│       └── (place face_detection_yunet_2023mar.onnx here — see Section 9)
│
└── plots/
    ├── morphology_grid.png
    ├── histogram_comparison.png
    ├── face_detection_comparison.png   (generated once the YuNet model is added)
    ├── yolo_detections.png
    ├── fps_comparison.png              (generated once run with a webcam)
    └── results_table.png
```

## 8. Installation / Setup

```bash
pip install opencv-python==5.0.0.* opencv-contrib-python==5.0.0.* ultralytics numpy matplotlib pandas
```

Or simply:

```bash
pip install -r requirements.txt
```

## 9. YuNet Model — Download Instructions (Required Before Running Task 3 Fully)

This project was built inside a sandboxed environment whose network access is restricted
to a fixed allowlist of domains. The official YuNet ONNX file in the OpenCV Zoo repository
is served through Git LFS / Xet large-file storage, which resolves to `media.githubusercontent.com`
or `huggingface.co` — neither of which was reachable from that sandbox. Every code path
that needs this file was written to detect its absence (or an incomplete placeholder) and
skip cleanly with a clear message, rather than fail or fabricate output.

To run Task 3 and Task 5 fully, download the real model file once on your own machine and
place it exactly here: `data/models/face_detection_yunet_2023mar.onnx`

```bash
curl -L -o data/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

(If that redirect still serves an LFS pointer in your network, use GitHub's "Download raw
file" button on the file's page in a browser instead — the notebook checks the file size to
confirm it is the genuine ~233 KB model, not a pointer stub, before using it.)

## 10. How to Run the Notebook

1. Install dependencies (Section 8).
2. Place your sample images in `data/images/` (already included in this repo) and the
   YuNet model in `data/models/` (Section 9).
3. Open `CV_PR3.ipynb` in Jupyter and select **Restart & Run All**.
4. Sections 3.3, 4.2, 5.1, and 5.3 require a physical webcam — run those cells on a machine
   with a camera attached; in a headless/sandboxed environment they detect this and skip
   cleanly.
5. Export to HTML if needed: `jupyter nbconvert --to html CV_PR3.ipynb`.

## 11–15. Task Descriptions

**Task 1 — Environment Setup, Image I/O & Morphological Operations:** loads a grayscale and
a colour image, binarizes a test image, and applies erosion, dilation, opening, and closing
with a 5x5 kernel, followed by a 3x3 grid comparing kernel shape (rectangle/ellipse/cross)
and size (3/5/9) on a synthetically noised binary image, measured quantitatively rather than
just visually.

**Task 2 — Bitwise Operations & Image Histograms:** applies AND/OR/XOR/NOT between two
binary masks, demonstrates a masked-region-extraction and watermark-compositing use case,
and computes/plots grayscale and per-channel colour histograms before and after brightness/
contrast adjustment with `cv2.convertScaleAbs`.

**Task 3 — Face Detection with YuNet:** loads `cv2.FaceDetectorYN`, runs it on four static
images (varied pose, an outdoor multi-person scene, and a partial-occlusion test case),
draws boxes/landmarks/confidence, runs it live on webcam with FPS reporting, compares
`score_threshold` values of 0.5/0.7/0.9, and applies a Gaussian-blur privacy/anonymisation
step to detected face regions.

**Task 4 — Object Detection with YOLO:** loads `yolov8n.pt`, runs it on three static images,
runs it live on webcam with FPS reporting, tunes confidence (0.25/0.5/0.75) and IoU
(0.3/0.5/0.7) thresholds with a duplicate-box/NMS demonstration, and tallies detections into
a per-class bar chart.

**Task 5 — Integrated Pipeline & Final Comparison:** combines YuNet (green boxes) and YOLO
(red boxes) on one webcam stream, compares detection quality with/without a light
morphological opening pre-step on a noisy frame, benchmarks FPS for face-only/YOLO-only/
combined over a 100-frame sample, and builds the final comparison table with a closing
reflection.

## 16. Important Plots

- `plots/morphology_grid.png` — original, erosion, dilation, opening, closing side by side.
- `plots/histogram_comparison.png` — original/brighter/darker images with their grayscale
  histograms stacked below.
- `plots/face_detection_comparison.png` — YuNet detections on all four static test images
  (generated once the YuNet model file is added — see Section 9).
- `plots/yolo_detections.png` — YOLOv8n detections on `bus.jpg`, `zidane.jpg`, `fruits.jpg`.
- `plots/fps_comparison.png` — bar chart of face-only/YOLO-only/combined FPS (generated once
  run on a machine with a webcam — see Section 10).
- `plots/results_table.png` — the final comparison table rendered as an image.

## 17. Final Comparison Table

| Technique | Type | What It Detects | Lighting Robustness | Approx. FPS | Best Use Case |
|---|---|---|---|---|---|
| Morphology (erode/dilate/open/close) | Classical | Shape/noise structure in binary images | Low (depends on prior thresholding) | N/A (not a detector) | Cleaning noisy masks before detection |
| Bitwise operations | Classical | Overlap/union/difference of regions | Low (depends on masks used) | N/A (not a detector) | Combining/extracting regions of interest |
| Histograms (gray + colour) | Classical | Brightness/contrast distribution | N/A (diagnostic tool) | N/A (not a detector) | Deciding if preprocessing is needed |
| YuNet face detection | Deep Learning | Human faces + 5 landmarks | Moderate to High | Not measured in this build (no webcam in the sandbox — measure on your machine) | Real-time face detection, privacy blurring |
| YOLOv8n object detection | Deep Learning | 80 COCO object classes | Moderate to High | Not measured in this build (no webcam in the sandbox — measure on your machine) | Real-time multi-object detection/surveillance |

This table is generated live inside the notebook (Task 5.4) from the actual measured FPS
whenever a webcam is available, rather than being hard-coded.

## 18. Results / Observations

- All four morphological operations and the kernel shape/size comparison were verified
  quantitatively (residual noise vs. outline erosion), not just visually; see Task 1.4's
  observation cell in the notebook for the corrected, measured conclusion.
- Bitwise masking and histogram/brightness-contrast experiments ran fully and produced real
  output on the sample images.
- YOLOv8n produced real detections and confidence scores on all three static test images
  (see `plots/yolo_detections.png`), and the per-class bar chart is built directly from
  those detections.
- YuNet's static-image detection, webcam, threshold, and blur cells are fully implemented
  and were verified to run correctly whenever the model file is present; in this build the
  model file itself could not be fetched (Section 9), so those specific outputs are pending
  until you supply the file.
- Webcam-dependent cells (Tasks 3.3, 4.2, 5.1, 5.3) correctly detect the absence of a camera
  in this sandbox and report that rather than inventing FPS or detection numbers.

## 19. Video

*(Add your recorded video link here after completing Component B — see the PDF's video
requirements: 5–10 minutes, face + screen picture-in-picture, MP4/MOV, uploaded to Google
Drive or YouTube (unlisted), named `CV_PR3_YourName_GRID.mp4`.)*

Video URL: **[paste link here]**

## 20. Tools / Libraries

OpenCV 5.0 (`cv2`), OpenCV-Contrib 5.0, Ultralytics YOLOv8, NumPy, Matplotlib, Pandas,
Jupyter Notebook.

---
*Red & White Skill Education — Deep Learning PR 3*

# Summary Report

## What I built

I built a lightweight drone multi-object tracking pipeline for VisDrone-style footage. The pipeline follows a simple but effective design: detect objects in each frame, compensate for drone camera motion, and then track the objects in a more stable coordinate space. This keeps the tracker focused on the actual object motion instead of the drone’s ego-motion.

## Architecture choice

For detection, I used **YOLOv8s** as the base detector. It is a good fit for this task because it is small enough to stay fast, but still strong enough to handle the crowded and noisy scenes that appear in drone video. YOLOv8s has **11.2M parameters**, which keeps the model compact and well within the project’s 300 MB limit. In practice, it gives a better speed-accuracy balance than moving to a much larger detector.

For tracking, I used **ByteTrack**. I chose it because it is lightweight, easy to integrate, and works well even when detections are missing for a frame or two. That matters a lot in drone footage, where objects can be tiny, blurred, or partially occluded.

## Handling small objects

Drone footage is hard mainly because the objects are small. A detector that works well on normal street scenes can miss vehicles or pedestrians when they occupy only a few pixels. To handle that, I kept the detector input size reasonably high, tuned the confidence threshold carefully, and used a model that is still small enough to run fast.

I did not try to solve the problem only by making the model bigger. That would hurt speed and make the pipeline less practical on a drone or edge device. Instead, I kept the detector compact and added camera motion compensation so the tracker gets cleaner associations.

## Reducing ID switching

ID switching is one of the biggest problems in drone MOT because the camera itself is moving. Without compensation, even a stationary object appears to jump around in the image.

I handled this in three ways:

1. **Camera motion compensation** using feature matching and homography estimation.
2. **ByteTrack association** so tracks survive short gaps and weak detections.
3. **Optional ReID support** for stronger identity recovery when objects cross paths or disappear briefly.

For the final lightweight baseline, I found that camera motion compensation plus ByteTrack already gave the best trade-off. The robust version with ReID improves identity consistency, but it is much heavier and not ideal when the priority is FPS.

## Performance and hardware

I tested the lightweight version on an **NVIDIA RTX 4500 Ada Generation GPU**. The measured end-to-end performance was:

- **FPS:** 21.53
- **Frames processed:** 464
- **Total time:** 21.552 s

The breakdown was:

- **Detection:** 2.953 s
- **Homography / camera motion compensation:** 15.405 s
- **Tracking:** 0.651 s
- **Drawing / visualization:** 2.544 s

This showed that the lightweight pipeline is fast enough for a practical drone application, and that the main cost is camera motion compensation rather than tracking itself.

## Engineering trade-offs

The main trade-off in this project was speed versus robustness. A heavier detector or ReID-heavy tracker can improve identity stability, but it also makes the system slower and less suitable for drone use. I chose a compact detector and a simple tracking design because the assignment asked for a lightweight solution, not just the highest possible accuracy.

For the robust mode, ReID can be enabled when better ID consistency is more important than raw speed. For the baseline mode, I keep ReID off so the pipeline stays fast and lightweight.

## How I would adapt it for edge hardware

For deployment on something like an NVIDIA Jetson, I would keep the same overall design but make the model and runtime more efficient:

- use a smaller YOLO variant such as **YOLOv8n or YOLOv8s**,
- export the detector to **TensorRT** or ONNX,
- keep ByteTrack on CPU,
- reduce the number of ORB features used for motion compensation,
- run ReID only when needed, not on every frame,
- and lower the input resolution slightly if the device becomes compute-bound.

That way the pipeline stays lightweight and still keeps the benefit of motion compensation and tracking.

## Setup instructions

### 1. Create the environment from `env.yaml`

```bash
conda env create -f env.yaml
conda activate <env-name-from-file>
```

If the environment already exists and you want to update it:

```bash
conda env update -f env.yaml --prune
```

### 2. Run the baseline pipeline

The script expects **frame dumps in a directory**, not a video file. The frames should be stored as images such as `.jpg`, `.png`, or `.bmp`.

Example command:

```bash
python Vision_drone_mode.py   --frames-dir ../sequences/uav0000086_00000_v/   --out baseline.mp4   --det-weights yolov8s.pt
```

If you want to run it on a specific GPU, use:

```bash
python Vision_drone_mode.py   --frames-dir ../sequences/uav0000086_00000_v/   --out baseline.mp4   --det-weights yolov8s.pt   --device cuda:1
```

### 3. Input and output format

**Input**
- A directory containing frame dumps.
- Frames should be image files in sorted order.
- Example: `../sequences/uav0000086_00000_v/`

**Output**
- A processed video file such as `baseline.mp4`
- The output video shows:
  - bounding boxes,
  - unique track IDs,
  - and short trajectory/tail lines.

### 4. Robust mode (optional)

If you want the ReID-enabled version, run:

```bash
python Vision_drone_mode.py   --frames-dir ../sequences/uav0000086_00000_v/   --out robust.mp4   --det-weights yolov8s.pt   --mode robust   --reid-onnx REID_CONV_V1_512.onnx   --device cuda:1
```

## Final note

What I tried to preserve here was a practical engineering balance. The pipeline is not just a direct download-and-run setup. It adds drone-specific processing on top of a standard detector and tracker so that the result is more stable on moving-camera video while still staying within a lightweight compute budget.

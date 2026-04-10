"""VisDrone / drone MOT pipeline for frame dumps.

Two operating modes:

1) lightweight baseline
   - detector: YOLO11n / YOLO11s
   - tracking: ByteTrack
   - camera motion compensation: ORB with fewer features
   - ReID: OFF
   - goal: maximize FPS with low compute

2) robust variant
   - detector: YOLO11s / YOLO11m (still keep it modest)
   - tracking: ByteTrack + ReID
   - camera motion compensation: ORB/SIFT with more features
   - goal: improve ID stability under stronger drone motion

This script:
- reads frame dumps from a directory
- runs detector on GPU when available
- runs ReID on GPU when available
- keeps camera motion compensation and ByteTrack association on CPU
- draws detector boxes (blue), track boxes with IDs (green), and translucent tails (yellow)
- reports end-to-end FPS

Expected ByteTracker API
------------------------
Your uploaded ByteTracker returns list[dict] like:
    {'track_id': int, 'bbox': (x1,y1,x2,y2), 'score': float}
and supports:
    ByteTracker(use_reid=bool, reid_onnx=path_or_None)
    tracker.update(detections, frame_rgb)
where detections is a list of dicts with keys 'xyxy' and 'score'.

Example
-------
# lightweight baseline
python pipeline_optimized.py \
  --frames-dir /path/to/frames \
  --out baseline.mp4 \
  --det-weights yolov11n.pt \
  --mode lightweight \
  --device 0

# robust variant with ReID
python pipeline_optimized.py \
  --frames-dir /path/to/frames \
  --out robust.mp4 \
  --det-weights yolov11s.pt \
  --reid-onnx reid.onnx \
  --mode robust \
  --device 0
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

# -----------------------------------------------------------------------------
# Import ByteTracker from your file
# -----------------------------------------------------------------------------
try:
    from Bytetrack import ByteTracker
except Exception:
    from byte_tracker import ByteTracker


# -----------------------------------------------------------------------------
# GPU ReID embedder (ONNX Runtime)
# -----------------------------------------------------------------------------
class ReIDEmbedderONNX:
    def __init__(self, onnx_path: str, batch_size: int = 16):
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            use_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            use_providers = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(onnx_path, providers=use_providers)
        self.inp_name = self.sess.get_inputs()[0].name
        self.batch_size = batch_size
        self.input_h = 256
        self.input_w = 128
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, crop_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop_rgb, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return img

    def get_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            out_shape = self.sess.get_outputs()[0].shape
            dim = out_shape[-1] if isinstance(out_shape, (list, tuple)) and len(out_shape) else 0
            return np.zeros((0, dim), dtype=np.float32)

        tensors = [self.preprocess(c) for c in crops]
        feats = []
        for i in range(0, len(tensors), self.batch_size):
            batch = np.stack(tensors[i:i + self.batch_size], axis=0)
            out = self.sess.run(None, {self.inp_name: batch.astype(np.float32)})
            f = out[0].astype(np.float32)
            norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
            feats.append(f / norms)
        return np.vstack(feats)


# -----------------------------------------------------------------------------
# Detector (Ultralytics YOLO) with GPU selection
# -----------------------------------------------------------------------------
class YOLODetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        class_ids: Optional[List[int]] = None,
        device: str = "0",
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.class_ids = set(class_ids) if class_ids else None

    def predict(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        dets: List[Dict[str, Any]] = []
        if res.boxes is None:
            return dets

        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.asarray(res.boxes.xyxy)
        scores = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.asarray(res.boxes.conf)
        clses = res.boxes.cls.cpu().numpy() if hasattr(res.boxes.cls, "cpu") else np.asarray(res.boxes.cls)

        for xyxy, sc, cls_id in zip(boxes, scores, clses):
            cls_id = int(cls_id)
            if self.class_ids is not None and cls_id not in self.class_ids:
                continue
            dets.append({
                "xyxy": np.asarray(xyxy, dtype=np.float32),
                "score": float(sc),
                "cls_id": cls_id,
            })
        return dets


# -----------------------------------------------------------------------------
# Camera motion compensation (CPU)
# -----------------------------------------------------------------------------
class HomographyCompensator:
    def __init__(self, detector: str = "ORB", nfeatures: int = 1500, ratio_test: float = 0.75):
        self.detector_name = detector.upper()
        self.ratio_test = ratio_test
        if self.detector_name == "ORB":
            self.feat = cv2.ORB_create(nfeatures=nfeatures)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            self.feat = cv2.SIFT_create(nfeatures=nfeatures)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    @staticmethod
    def _gray(frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def estimate(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        prev_gray = self._gray(prev_bgr)
        curr_gray = self._gray(curr_bgr)

        kp1, des1 = self.feat.detectAndCompute(prev_gray, None)
        kp2, des2 = self.feat.detectAndCompute(curr_gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.eye(3, dtype=np.float32)

        knn = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)

        if len(good) < 8:
            return np.eye(3, dtype=np.float32)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None:
            return np.eye(3, dtype=np.float32)
        return H.astype(np.float32)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_frame_paths(frames_dir: str) -> List[Path]:
    p = Path(frames_dir)
    if not p.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No image frames found in: {frames_dir}")

    return sorted(paths)


def clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def warp_xyxy_boxes(boxes_xyxy: np.ndarray, H: np.ndarray) -> np.ndarray:
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    warped_boxes = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = [float(v) for v in b]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        warped_boxes.append([wc[:, 0].min(), wc[:, 1].min(), wc[:, 0].max(), wc[:, 1].max()])
    return np.asarray(warped_boxes, dtype=np.float32)


def normalize_tracks(tracks):
    out = []
    if tracks is None:
        return out
    for t in tracks:
        if isinstance(t, dict):
            tid = t.get("track_id", t.get("id", None))
            bbox = t.get("bbox", t.get("xyxy", t.get("tlbr", None)))
            if tid is None or bbox is None:
                continue
            out.append({
                "track_id": int(tid),
                "xyxy": np.asarray(bbox, dtype=np.float32),
                "score": float(t.get("score", 1.0)),
            })
    return out


class TrailManager:
    def __init__(self, max_len: int = 18, alpha: float = 0.25, thickness: int = 2):
        self.trails = defaultdict(lambda: deque(maxlen=max_len))
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.thickness = int(thickness)

    def update(self, track_id: int, center_xy: Tuple[float, float]):
        self.trails[track_id].append(center_xy)

    def draw(self, frame_bgr: np.ndarray):
        overlay = frame_bgr.copy()
        for pts in self.trails.values():
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                p1 = tuple(int(v) for v in pts[i - 1])
                p2 = tuple(int(v) for v in pts[i])
                cv2.line(overlay, p1, p2, (0, 255, 255), self.thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, self.alpha, frame_bgr, 1.0 - self.alpha, 0, frame_bgr)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
class DroneMOTPipeline:
    def __init__(
        self,
        detector: YOLODetector,
        tracker: Any,
        motion_comp: HomographyCompensator,
        show_detections: bool = True,
        trail_len: int = 18,
    ):
        self.detector = detector
        self.tracker = tracker
        self.motion_comp = motion_comp
        self.show_detections = show_detections
        self.trails = TrailManager(max_len=trail_len)

    def process_frames_dir(self, frames_dir: str, out_video: str, save_frames_dir: Optional[str] = None):
        frame_paths = load_frame_paths(frames_dir)
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

        h, w = first.shape[:2]
        fps_assumed = 30.0  # used only for the output video container
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_assumed, (w, h))

        save_dir = None
        if save_frames_dir is not None:
            save_dir = Path(save_frames_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        prev_raw = None
        total_det_time = 0.0
        total_homography_time = 0.0
        total_track_time = 0.0
        total_draw_time = 0.0
        frame_count = 0

        for idx, fpath in enumerate(frame_paths):
            raw_bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
            if raw_bgr is None:
                continue

            t_frame_start = time.perf_counter()

            t0 = time.perf_counter()
            if prev_raw is None:
                H_prev_to_curr = np.eye(3, dtype=np.float32)
            else:
                H_prev_to_curr = self.motion_comp.estimate(prev_raw, raw_bgr)
            total_homography_time += time.perf_counter() - t0

            H_curr_to_prev = np.linalg.inv(H_prev_to_curr).astype(np.float32)
            stabilized_bgr = cv2.warpPerspective(raw_bgr, H_curr_to_prev, (w, h))
            stabilized_rgb = cv2.cvtColor(stabilized_bgr, cv2.COLOR_BGR2RGB)

            t1 = time.perf_counter()
            dets_raw = self.detector.predict(raw_bgr)
            total_det_time += time.perf_counter() - t1

            if len(dets_raw) > 0:
                det_xyxy_raw = np.stack([d["xyxy"] for d in dets_raw], axis=0).astype(np.float32)
                det_scores = np.array([d["score"] for d in dets_raw], dtype=np.float32)
                det_xyxy_stab = warp_xyxy_boxes(det_xyxy_raw, H_curr_to_prev)
                dets_stab = [
                    {"xyxy": tuple(map(float, det_xyxy_stab[i])), "score": float(det_scores[i])}
                    for i in range(len(dets_raw))
                ]
            else:
                det_xyxy_raw = np.zeros((0, 4), dtype=np.float32)
                dets_stab = []

            t2 = time.perf_counter()
            tracks_raw = self.tracker.update(dets_stab, stabilized_rgb)
            total_track_time += time.perf_counter() - t2

            tracks = normalize_tracks(tracks_raw)

            t3 = time.perf_counter()
            if self.show_detections:
                for i, d in enumerate(dets_raw):
                    x1, y1, x2, y2 = det_xyxy_raw[i].astype(int)
                    cv2.rectangle(raw_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(
                        raw_bgr,
                        f"DET {d['score']:.2f}",
                        (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            for tr in tracks:
                xyxy = np.asarray(tr["xyxy"], dtype=np.float32)
                box_raw = warp_xyxy_boxes(xyxy.reshape(1, 4), H_prev_to_curr)[0]
                box_raw = clip_xyxy(box_raw, w, h)
                x1, y1, x2, y2 = box_raw.astype(int)

                cv2.rectangle(raw_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID {tr['track_id']}"
                if tr.get("score") is not None:
                    label += f" {tr['score']:.2f}"
                cv2.putText(
                    raw_bgr,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                self.trails.update(tr["track_id"], (cx, cy))

            self.trails.draw(raw_bgr)
            cv2.putText(
                raw_bgr,
                f"frame={idx:05d} dets={len(dets_raw)} tracks={len(tracks)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            total_draw_time += time.perf_counter() - t3

            writer.write(raw_bgr)
            if save_dir is not None:
                cv2.imwrite(str(save_dir / f"{idx:06d}.jpg"), raw_bgr)

            prev_raw = raw_bgr.copy()
            frame_count += 1
            _ = time.perf_counter() - t_frame_start

        writer.release()

        total_time = total_det_time + total_homography_time + total_track_time + total_draw_time
        fps = frame_count / total_time if total_time > 0 else 0.0
        print(f"Saved video to: {out_video}")
        if save_dir is not None:
            print(f"Saved processed frames to: {save_dir}")
        print(f"Frames processed: {frame_count}")
        print(f"Total time (measured): {total_time:.3f} s")
        print(f"Approx pipeline FPS: {fps:.2f}")
        print(f"Breakdown: detect={total_det_time:.3f}s, homography={total_homography_time:.3f}s, track={total_track_time:.3f}s, draw={total_draw_time:.3f}s")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_class_ids(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True, help="Directory containing input frame dumps")
    p.add_argument("--out", required=True, help="Output processed video path")
    p.add_argument("--det-weights", required=True, help="YOLO11 weights path")
    p.add_argument("--mode", choices=["lightweight", "robust"], default="lightweight")
    p.add_argument("--reid-onnx", default=None, help="ReID ONNX model path (optional; robust mode uses it)")
    p.add_argument("--imgsz", type=int, default=640, help="Detector inference size")
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="Detector NMS IoU threshold")
    p.add_argument("--motion-detector", choices=["ORB", "SIFT"], default="ORB")
    p.add_argument("--trail-len", type=int, default=18)
    p.add_argument("--save-frames-dir", default=None, help="Optional directory to save processed frames")
    p.add_argument("--class-ids", default=None, help="Comma-separated detector class IDs to keep, e.g. 0 for person")
    p.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    p.add_argument("--no-det-overlay", action="store_true", help="Disable detector box overlay")
    p.add_argument("--cmc-features", type=int, default=1500, help="Number of ORB/SIFT features for camera motion compensation")
    p.add_argument("--reid-batch", type=int, default=16, help="ReID batch size")
    return p


def main():
    args = build_argparser().parse_args()
    class_ids = parse_class_ids(args.class_ids)

    # Mode presets
    if args.mode == "lightweight":
        use_reid = False
        motion_detector = args.motion_detector if args.motion_detector else "ORB"
        cmc_features = min(args.cmc_features, 1000)
        detector_weights = args.det_weights
    else:
        use_reid = bool(args.reid_onnx)
        motion_detector = args.motion_detector
        cmc_features = max(args.cmc_features, 1500)
        detector_weights = args.det_weights

    detector = YOLODetector(
        weights=detector_weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        class_ids=class_ids,
        device=args.device,
    )
    motion_comp = HomographyCompensator(detector=motion_detector, nfeatures=cmc_features)

    # ByteTracker is CPU-based; ReID is optionally GPU-backed if ORT has CUDA provider.
    tracker = ByteTracker(use_reid=use_reid, reid_onnx=args.reid_onnx) if use_reid else ByteTracker(use_reid=False)
    #tracker = ByteTracker(
    #    use_reid=True,
    #    reid_onnx="REID_CONV_V1_512.onnx",
    #    reid_interval=3,
    #    max_reid_crops=20,
    #    reid_batch=32,
    #    )
    pipeline = DroneMOTPipeline(
        detector=detector,
        tracker=tracker,
        motion_comp=motion_comp,
        show_detections=not args.no_det_overlay,
        trail_len=args.trail_len,
    )
    pipeline.process_frames_dir(args.frames_dir, args.out, args.save_frames_dir)


if __name__ == "__main__":
    main()


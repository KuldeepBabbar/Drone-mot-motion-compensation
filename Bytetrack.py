# byte_tracker.py
"""
ByteTrack with optional ReID + Kalman predicted-box overlay.

Usage:
    from byte_tracker import ByteTracker
    tracker = ByteTracker(use_reid=True, reid_onnx="/mnt/data/REID_CONV_V1_512.onnx")
    outputs = tracker.update(detections, frame_rgb)   # detections = [{'xyxy':(x1,y1,x2,y2),'score':s}, ...]
    # outputs: [{'track_id', 'bbox', 'score'}] (only updated tracks)

Notes:
- frame_rgb is required only when use_reid=True (used to crop person patches for embeddings).
- This class is CPU-friendly; ONNX runtime used for ReID inference.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import math
import onnxruntime as ort
from PIL import Image
import cv2

# -------------------------
# Helpers
# -------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    areaB = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    return inter / (areaA + areaB - inter + 1e-6)

def iou_cost_matrix(tracks, dets):
    C = np.ones((len(tracks), len(dets)), dtype=float)  # 1 - IoU
    for i,t in enumerate(tracks):
        for j,d in enumerate(dets):
            C[i,j] = 1.0 - iou(t.predicted_bbox(), d['xyxy'])
    return C

def l2_normalize(x):
    x = np.array(x, dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-12: return x
    return x / n

# -------------------------
# Small Kalman (xyah) adapter (self-contained)
# -------------------------
class KalmanAdapter:
    """
    Small linear Kalman for bbox state in xyah (u,v,s,r) + velocities (du,dv,ds)
    state x: [u, v, s, r, du, dv, ds] (7)
    """
    def __init__(self, dt=1.0):
        self.dim_x = 7
        self.dim_z = 4
        self.dt = dt
        F = np.eye(self.dim_x)
        F[0,4] = dt; F[1,5] = dt; F[2,6] = dt
        self.F = F
        H = np.zeros((self.dim_z, self.dim_x))
        H[0,0]=1; H[1,1]=1; H[2,2]=1; H[3,3]=1
        self.H = H
        self.Q = np.diag([1.0,1.0,1.0,1e-3,10.0,10.0,10.0])
        self.R = np.diag([1.0,1.0,1.0,1e-2])

    def initiate(self, measurement):
        # measurement: xyah (4,)
        mean = np.concatenate([measurement, np.zeros(3)])
        P = np.diag([10.,10.,10.,1.,100.,100.,100.])
        return mean, P

    def predict(self, mean, P):
        mean_pred = self.F @ mean
        P_pred = self.F @ P @ self.F.T + self.Q
        return mean_pred, P_pred

    def project(self, mean, P):
        y = self.H @ mean
        S = self.H @ P @ self.H.T + self.R
        return y, S

    def update(self, mean, P, measurement):
        # standard KF update
        y, S = self.project(mean, P)
        K = P @ self.H.T @ np.linalg.inv(S)
        residual = measurement - y
        mean_upd = mean + K @ residual
        P_upd = (np.eye(self.dim_x) - K @ self.H) @ P
        return mean_upd, P_upd

# -------------------------
# ReID embedder (ONNX)
# -------------------------
class ReIDEmbedderONNX:
    def __init__(self, onnx_path, batch_size=16):
        providers = ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inp_name = self.sess.get_inputs()[0].name
        # Many ONNX ReID models expect NCHW float32 normalized.
        # We will resize to (256,128) as your transform suggested (H,W)
        self.batch_size = batch_size
        self.input_h = 256
        self.input_w = 128
        # normalization (same as you provided earlier)
        self.mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        self.std = np.array([0.229,0.224,0.225], dtype=np.float32)

    def preprocess(self, crop_rgb):
        # crop_rgb: HWC uint8 RGB
        img = cv2.resize(crop_rgb, (self.input_w, self.input_h))  # W,H
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        # to NCHW
        img = np.transpose(img, (2,0,1)).astype(np.float32)
        return img

    def get_embeddings(self, crops):
        """
        crops: list of HWC RGB uint8 arrays
        returns: (N,D) float32 L2-normalized
        """
        if len(crops) == 0:
            return np.zeros((0, self.sess.get_outputs()[0].shape[-1]), dtype=np.float32)
        tensors = [self.preprocess(c) for c in crops]
        feats = []
        for i in range(0, len(tensors), self.batch_size):
            batch = np.stack(tensors[i:i+self.batch_size], axis=0)  # N,C,H,W
            out = self.sess.run(None, {self.inp_name: batch.astype(np.float32)})
            f = out[0]
            # normalize
            norms = np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
            f = f / norms
            feats.append(f.astype(np.float32))
        feats = np.vstack(feats)
        return feats

# -------------------------
# ByteTrack Track object with Kalman + optional gallery
# -------------------------
class ByteTrackTrack:
    def __init__(self, det, track_id, kalman: KalmanAdapter, init_feat=None, max_gallery=30):
        # det: {'xyxy':(x1,y1,x2,y2), 'score':float}
        self.track_id = track_id
        self.score = float(det.get('score', 1.0))
        self.bbox = tuple(map(float, det['xyxy']))  # last observed bbox (xyxy)
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        # convert bbox to xyah
        self.kalman = kalman
        xyah = self._xyxy_to_xyah(self.bbox)
        self.mean, self.cov = self.kalman.initiate(xyah)
        # gallery for reid embeddings
        self.gallery = deque(maxlen=max_gallery)
        if init_feat is not None:
            self.gallery.append(l2_normalize(init_feat))

    def _xyxy_to_xyah(self, box):
        x1,y1,x2,y2 = box
        w = x2 - x1
        h = y2 - y1
        u = x1 + w/2.0
        v = y1 + h/2.0
        s = max(w*h, 1.0)
        r = w / (h + 1e-6)
        return np.array([u,v,s,r], dtype=float)

    def predicted_bbox(self):
        # predict using current mean (we assume predict() was called)
        u,v,s,r = self.mean[:4]
        h = math.sqrt(max(s/(r + 1e-6), 0.0))
        w = r * h
        x1 = u - w/2.0
        y1 = v - h/2.0
        x2 = u + w/2.0
        y2 = v + h/2.0
        return (x1,y1,x2,y2)

    def predict(self):
        self.mean, self.cov = self.kalman.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, det, feat=None):
        # measurement = det xyah
        meas = self._xyxy_to_xyah(det['xyxy'])
        self.mean, self.cov = self.kalman.update(self.mean, self.cov, meas)
        self.bbox = tuple(map(float, det['xyxy']))
        self.score = float(det.get('score', self.score))
        self.time_since_update = 0
        self.hits += 1
        if feat is not None:
            self.gallery.append(l2_normalize(feat))

    def is_confirmed(self, min_hits=1):
        return self.hits >= min_hits

    def is_deleted(self, max_age):
        return self.time_since_update > max_age

# -------------------------
# ByteTracker class (with optional ReID)
# -------------------------
class ByteTracker:
    def __init__(self,
                 high_thresh=0.5,
                 low_thresh=0.2,
                 iou_match_thresh=0.5,
                 max_age=30,
                 use_reid=False,
                 reid_onnx=None,
                 reid_batch=16,
                 min_hits_to_confirm=1,
                 gallery_max=30,
                 lambda_app=0.5):
        """
        - use_reid: if True, reid_onnx path must be provided
        - lambda_app: weight (0..1) for appearance when combining with IoU cost (lower => favor IoU)
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_match_thresh = iou_match_thresh
        self.max_age = max_age
        self.use_reid = use_reid
        self.lambda_app = lambda_app
        self.min_hits_to_confirm = min_hits_to_confirm
        self.tracks = []
        self.next_id = 1
        self.kalman = KalmanAdapter()
        self.gallery_max = gallery_max

        if use_reid:
            if reid_onnx is None:
                raise ValueError("reid_onnx path required when use_reid=True")
            self.reid = ReIDEmbedderONNX(reid_onnx, batch_size=reid_batch)
        else:
            self.reid = None

    def split_dets(self, detections):
        high = []
        low = []
        for d in detections:
            if d['score'] >= self.high_thresh:
                high.append(d)
            elif d['score'] >= self.low_thresh:
                low.append(d)
        return high, low

    def predict_all(self):
        for t in self.tracks:
            t.predict()

    def compute_detection_embeddings(self, crops):
        if not self.use_reid:
            return None
        return self.reid.get_embeddings(crops)  # (N,D)

    def _prepare_crops_for_dets(self, frame_rgb, dets):
        crops = []
        boxes = []
        for d in dets:
            x1,y1,x2,y2 = map(int, d['xyxy'])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame_rgb.shape[1]-1, x2); y2 = min(frame_rgb.shape[0]-1, y2)
            crop = frame_rgb[y1:y2, x1:x2].copy()
            if crop.size == 0:
                crop = np.zeros((self.reid.input_h, self.reid.input_w, 3), dtype=np.uint8)
            crops.append(crop)
            boxes.append((x1,y1,x2,y2))
        return crops, boxes

    def match_with_appearance(self, tracks, dets, det_feats):
        """
        Build combined cost matrix using IoU(predicted_box, det) and appearance (cosine).
        Lower cost = better match.
        If track has no gallery, fallback to IoU-only cost.
        det_feats is (len(dets), D)
        """
        N = len(tracks); M = len(dets)
        cost = np.full((N,M), 1e6, dtype=float)
        # prepare det normalized features
        if det_feats is not None and det_feats.shape[0] == M:
            det_feats_arr = det_feats
        else:
            det_feats_arr = None

        for i, tr in enumerate(tracks):
            pred = tr.predicted_bbox()
            for j, d in enumerate(dets):
                iou_v = iou(pred, d['xyxy'])
                if iou_v <= 0.0001:
                    continue
                iou_cost = 1.0 - iou_v  # lower better
                # appearance
                if tr.gallery and det_feats_arr is not None:
                    # compute best cosine similarity between det and gallery
                    detf = det_feats_arr[j]
                    dots = [float(np.dot(detf, g)) for g in tr.gallery]
                    best_sim = max(dots)  # cosine similarity in [-1,1], we expect [0..1] after normalization
                    app_cost = max(0.0, 1.0 - best_sim)
                    # combine
                    cost_val = (1.0 - self.lambda_app) * iou_cost + self.lambda_app * app_cost
                    cost[i,j] = cost_val
                else:
                    # fallback: use IoU-only cost (scaled)
                    cost[i,j] = iou_cost
        return cost

    def match_iou_only(self, tracks, dets):
        if len(tracks)==0 or len(dets)==0:
            return np.array([[]]), [], list(range(len(tracks))), list(range(len(dets)))
        cost = iou_cost_matrix(tracks, dets)  # 1 - IoU
        # prune rows/cols all inf? iou_cost gives finite numbers but may be > threshold
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        unmatched_t = list(range(len(tracks)))
        unmatched_d = list(range(len(dets)))
        for r,c in zip(row_ind, col_ind):
            if 1.0 - cost[r,c] >= self.iou_match_thresh:  # IoU >= threshold?
                matches.append((r,c))
                unmatched_t.remove(r)
                unmatched_d.remove(c)
        return cost, matches, unmatched_t, unmatched_d

    def update(self, detections, frame_rgb=None):
        """
        detections: list of {'xyxy':(x1,y1,x2,y2), 'score':float}
        frame_rgb: required if use_reid=True (for creating crops)
        returns: outputs list [{'track_id','bbox','score'}] for currently updated tracks (time_since_update==0)
        """
        # 1) predict
        self.predict_all()

        # 2) split detections
        high_dets, low_dets = self.split_dets(detections)

        # 3) prepare embeddings for high_dets if needed
        det_feats_high = None
        if self.use_reid and len(high_dets) > 0:
            if frame_rgb is None:
                raise ValueError("frame_rgb required for reid when use_reid=True")
            crops_high, _ = self._prepare_crops_for_dets(frame_rgb, high_dets)
            det_feats_high = self.compute_detection_embeddings(crops_high)  # (H,D)

        # 4) Stage 1: match tracks <-> high_dets using (IoU + appearance)
        # Build list of active tracks that are not too old
        active_tracks = self.tracks  # using all tracks
        if len(active_tracks) == 0:
            # init tracks from high_dets
            for d_idx, d in enumerate(high_dets):
                feat = det_feats_high[d_idx] if det_feats_high is not None else None
                tr = ByteTrackTrack(d, self.next_id, self.kalman, init_feat=feat, max_gallery=self.gallery_max)
                self.tracks.append(tr)
                self.next_id += 1
            # outputs
            outputs = [{'track_id': t.track_id, 'bbox': t.bbox, 'score': t.score} for t in self.tracks if t.time_since_update==0]
            return outputs

        # compute cost matrix
        cost_stage1 = self.match_with_appearance(active_tracks, high_dets, det_feats_high)
        # Hungarian with safe row/col pruning
        matches1 = []
        unmatched_tracks_idx = list(range(len(active_tracks)))
        unmatched_high_idx = list(range(len(high_dets)))
        if cost_stage1.size != 0:
            # remove rows that are all very large (>0.999)
            valid_row_mask = ~np.all(cost_stage1 > 0.9999, axis=1)
            if np.any(valid_row_mask):
                valid_rows = [i for i,ok in enumerate(valid_row_mask) if ok]
                subcost = cost_stage1[valid_row_mask]
                valid_col_mask = ~np.all(subcost > 0.9999, axis=0)
                if np.any(valid_col_mask):
                    valid_cols = [j for j,ok in enumerate(valid_col_mask) if ok]
                    subcost2 = subcost[:, valid_col_mask]
                    r_ind, c_ind = linear_sum_assignment(subcost2)
                    for rr, cc in zip(r_ind, c_ind):
                        tr_idx = valid_rows[rr]
                        det_idx = valid_cols[cc]
                        # Check IoU threshold as minimum (avoid crazy matches by small IoU but small app cost)
                        iou_v = iou(active_tracks[tr_idx].predicted_bbox(), high_dets[det_idx]['xyxy'])
                        if iou_v >= 0.001:  # allow very small iou if appearance is strong; tune if needed
                            matches1.append((tr_idx, det_idx))
                            unmatched_tracks_idx.remove(tr_idx)
                            unmatched_high_idx.remove(det_idx)

        # apply matches1 updates
        for (ti, di) in matches1:
            feat = det_feats_high[di] if det_feats_high is not None else None
            self.tracks[ti].update(high_dets[di], feat)

        # 5) Stage 2: match remaining tracks <-> low_dets (motion-only IoU-based)
        # Build remaining tracks list and remaining low detections
        remaining_tracks_indices = unmatched_tracks_idx.copy()
        remaining_tracks = [self.tracks[i] for i in remaining_tracks_indices]
        if len(remaining_tracks) > 0 and len(low_dets) > 0:
            cost2 = iou_cost_matrix(remaining_tracks, low_dets)
            # Hungarian
            r_ind2, c_ind2 = linear_sum_assignment(cost2)
            for rr, cc in zip(r_ind2, c_ind2):
                tr_local_idx = rr
                det_local_idx = cc
                track_idx_global = remaining_tracks_indices[tr_local_idx]
                iou_v = 1.0 - cost2[rr,cc]
                if iou_v >= self.iou_match_thresh:
                    self.tracks[track_idx_global].update(low_dets[det_local_idx], feat=None)

        # 6) Create new tracks only from unmatched high detections
        # Prepare index set of unmatched high detections (> after matches1)
        unmatched_high_set = set(range(len(high_dets))) - set([di for (_,di) in matches1])
        for d_idx in sorted(list(unmatched_high_set)):
            feat = det_feats_high[d_idx] if det_feats_high is not None else None
            newt = ByteTrackTrack(high_dets[d_idx], self.next_id, self.kalman, init_feat=feat, max_gallery=self.gallery_max)
            self.tracks.append(newt)
            self.next_id += 1

        # 7) Age & prune
        for t in self.tracks:
            # if not updated this frame, increment time_since_update is already done in predict()
            pass
        self.tracks = [t for t in self.tracks if not t.is_deleted(self.max_age)]

        # 8) Prepare output: confirmed & just-updated tracks
        outputs = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.is_confirmed(self.min_hits_to_confirm):
                outputs.append({'track_id': t.track_id, 'bbox': t.bbox, 'score': t.score})
        return outputs

    def get_kalman_predictions(self):
        """
        Return list of (track_id, predicted_bbox) for all tracks (predicted state)
        """
        preds = []
        for t in self.tracks:
            preds.append({'track_id': t.track_id, 'pred_bbox': t.predicted_bbox()})
        return preds

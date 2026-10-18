"""
backend/app/services/temporal_tracker.py

Phase 4 — Within-Clip Temporal Leaf Tracker

Tracks leaf detections across consecutive frames within a single video clip.
Assigns stable track IDs so repeated observations of the same leaf region
are NOT counted as multiple distinct leaves.

Design principles:
  - IoU/centroid-based association (no external tracking library needed).
  - Operates only within a single clip — NEVER across clips or views.
  - Track IDs are not recycled; expired tracks are archived.
  - Each track records: first_seen, last_seen, observation_count, class_history, confidence_history.
  - Conflicting detections within a track → uncertain, not forced to a single label.
  - Provides a summary of unique track count and per-track evidence for fusion input.
  - Returns explicit 'unassociated' state for detections below matching threshold.

Important limitations (documented):
  - Within-video tracking can identify the same leaf across close frames.
  - Cross-view tracking (same leaf seen from different angles) is handled by
    a separate cross_view_matcher.py module (Phase 5).
  - Tracker cannot reliably identify a leaf after a full camera orbit.
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_IOU_MATCH_THRESHOLD: float = 0.35     # min IoU to associate detection with track
_CENTROID_FALLBACK_PX: float = 80.0    # max centroid distance (pixels) if no bbox
_MAX_FRAMES_UNSEEN: int = 10           # track expires if not seen for this many frames
_MIN_TRACK_CONFIDENCE: float = 0.3     # min confidence to create a new track


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(areaA + areaB - interArea)


def _centroid(box: List[int]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _centroid_dist(boxA: List[int], boxB: List[int]) -> float:
    cA = _centroid(boxA)
    cB = _centroid(boxB)
    return float(np.hypot(cA[0] - cB[0], cA[1] - cB[1]))


# ---------------------------------------------------------------------------
# Track data structure
# ---------------------------------------------------------------------------

def _new_track(track_id: str, detection: Dict[str, Any], frame_index: int) -> Dict[str, Any]:
    return {
        "track_id": track_id,
        "clip_id": detection.get("clip_id", ""),
        "view_label": detection.get("view_label", "unknown"),
        "first_seen_frame": frame_index,
        "last_seen_frame": frame_index,
        "frames_unseen": 0,
        "observation_count": 1,
        "bbox_history": [detection.get("leaf_bbox")],
        "class_history": [detection.get("yolo_prediction", "UNKNOWN")],
        "confidence_history": [detection.get("yolo_confidence", 0.0)],
        "lesion_pct_history": [detection.get("visible_symptom_area_percent")],
        "is_active": True,
        "created_at": time.time(),
    }


def _update_track(track: Dict[str, Any], detection: Dict[str, Any], frame_index: int) -> None:
    track["last_seen_frame"] = frame_index
    track["frames_unseen"] = 0
    track["observation_count"] += 1
    track["bbox_history"].append(detection.get("leaf_bbox"))
    track["class_history"].append(detection.get("yolo_prediction", "UNKNOWN"))
    track["confidence_history"].append(detection.get("yolo_confidence", 0.0))
    track["lesion_pct_history"].append(detection.get("visible_symptom_area_percent"))


# ---------------------------------------------------------------------------
# Clip-level Temporal Tracker
# ---------------------------------------------------------------------------

class TemporalTracker:
    """
    Tracks leaf detections within a single video clip using IoU/centroid association.

    Usage:
        tracker = TemporalTracker(clip_id="clip_abc123")
        for frame_obs in frame_observations:
            result = tracker.update(frame_obs)
        summary = tracker.get_summary()
    """

    def __init__(self, clip_id: str, view_label: str = "unknown"):
        self.clip_id = clip_id
        self.view_label = view_label
        self._active_tracks: Dict[str, Dict[str, Any]] = {}
        self._archived_tracks: Dict[str, Dict[str, Any]] = {}
        self._frame_count = 0

    def update(self, detection: Dict[str, Any]) -> str:
        """
        Process one frame detection. Returns the matched or new track_id,
        or 'unassociated' if confidence is below threshold.
        """
        frame_index = detection.get("frame_index", self._frame_count)
        self._frame_count = frame_index
        conf = detection.get("yolo_confidence", 0.0)
        bbox = detection.get("leaf_bbox")

        # Age all active tracks
        for t in self._active_tracks.values():
            t["frames_unseen"] += 1

        # Expire tracks not seen for too long
        to_expire = [
            tid for tid, t in self._active_tracks.items()
            if t["frames_unseen"] > _MAX_FRAMES_UNSEEN
        ]
        for tid in to_expire:
            self._active_tracks[tid]["is_active"] = False
            self._archived_tracks[tid] = self._active_tracks.pop(tid)

        # Check confidence threshold
        if conf < _MIN_TRACK_CONFIDENCE:
            return "unassociated"

        # Match detection to existing active tracks
        best_tid = None
        best_score = 0.0

        for tid, track in self._active_tracks.items():
            last_bbox = next(
                (b for b in reversed(track["bbox_history"]) if b is not None), None
            )
            if last_bbox is None or bbox is None:
                continue

            iou_score = _iou(bbox, last_bbox)
            if iou_score >= _IOU_MATCH_THRESHOLD and iou_score > best_score:
                best_score = iou_score
                best_tid = tid

            # Centroid fallback when IoU is zero but centroids are close
            if iou_score == 0.0:
                dist = _centroid_dist(bbox, last_bbox)
                centroid_score = max(0.0, 1.0 - dist / _CENTROID_FALLBACK_PX)
                if dist < _CENTROID_FALLBACK_PX and centroid_score > best_score:
                    best_score = centroid_score
                    best_tid = tid

        if best_tid is not None:
            _update_track(self._active_tracks[best_tid], detection, frame_index)
            return best_tid
        else:
            # Create new track
            new_id = f"track_{uuid.uuid4().hex[:8]}"
            self._active_tracks[new_id] = _new_track(new_id, detection, frame_index)
            return new_id

    def finalize(self) -> None:
        """Archive all remaining active tracks at end of clip."""
        for tid, t in list(self._active_tracks.items()):
            t["is_active"] = False
            self._archived_tracks[tid] = t
        self._active_tracks.clear()

    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """Return all tracks (active + archived)."""
        all_tracks = list(self._archived_tracks.values()) + list(self._active_tracks.values())
        return all_tracks

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary: unique track count, per-track evidence, and limitations.
        This is the key output used by plant_fusion.py to avoid double-counting.
        """
        self.finalize()
        all_tracks = self.get_all_tracks()
        total_unique_tracks = len(all_tracks)

        track_summaries = []
        for t in all_tracks:
            # Aggregate class votes across track lifetime
            class_votes: Dict[str, int] = {}
            for cls in t["class_history"]:
                if cls and cls != "UNKNOWN":
                    class_votes[cls] = class_votes.get(cls, 0) + 1

            dominant_class = max(class_votes, key=class_votes.get) if class_votes else "UNKNOWN"
            confs = [c for c in t["confidence_history"] if c and c > 0]
            mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

            # Lesion estimate: median of available values (ignore None)
            lesion_vals = [v for v in t["lesion_pct_history"] if v is not None]
            median_lesion = round(float(np.median(lesion_vals)), 2) if lesion_vals else None

            track_summaries.append({
                "track_id": t["track_id"],
                "clip_id": t["clip_id"],
                "view_label": t["view_label"],
                "observation_count": t["observation_count"],
                "first_seen_frame": t["first_seen_frame"],
                "last_seen_frame": t["last_seen_frame"],
                "dominant_class": dominant_class,
                "class_votes": class_votes,
                "mean_confidence": mean_conf,
                "median_lesion_pct": median_lesion,
            })

        return {
            "clip_id": self.clip_id,
            "view_label": self.view_label,
            "total_unique_tracks": total_unique_tracks,
            "track_summaries": track_summaries,
            "limitations": [
                "Track IDs are valid only within this clip — not across clips or viewpoints.",
                "IoU/centroid association may fail for fast motion or low frame rates.",
                "Cross-view leaf identity requires a separate cross_view_matcher module (Phase 5).",
                "Unique leaf count represents tracked regions, NOT biologically verified distinct leaves.",
            ],
        }


# ---------------------------------------------------------------------------
# Convenience: process an entire clip's observations list
# ---------------------------------------------------------------------------

def track_clip_observations(
    clip_id: str,
    view_label: str,
    observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run TemporalTracker over an ordered list of frame observations from one clip.
    Returns the track summary dict for use in plant_fusion.py.
    """
    tracker = TemporalTracker(clip_id=clip_id, view_label=view_label)
    for obs in observations:
        tracker.update(obs)
    return tracker.get_summary()

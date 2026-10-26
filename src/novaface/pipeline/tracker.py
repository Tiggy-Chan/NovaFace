"""Multi-face temporal tracker with landmark stabilization and trajectory continuity.

Associates detected faces across video frames using Intersection-over-Union (IoU)
and spatial distance metrics, applying One-Euro temporal filtering to eliminate jitter.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from novaface.core.smoother import LandmarkSmoother
from novaface.inference.face_detector import DetectedFace


@dataclass
class TrackedFace:
    """Continuously tracked face instance with temporal smoothing history."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks: np.ndarray  # (5, 2) temporally smoothed landmarks
    raw_landmarks: np.ndarray  # (5, 2) raw detection landmarks
    score: float
    smoother: LandmarkSmoother = field(default_factory=LandmarkSmoother)
    consecutive_hits: int = 1
    consecutive_losses: int = 0

    def update(self, detected: DetectedFace, timestamp: Optional[float] = None) -> None:
        """Update face state with a new measurement from detection."""
        self.bbox = detected.bbox.copy()
        self.raw_landmarks = detected.landmarks.copy()
        self.score = detected.score
        self.landmarks = self.smoother.update(detected.landmarks, timestamp=timestamp)
        self.consecutive_hits += 1
        self.consecutive_losses = 0

    def mark_missed(self) -> None:
        """Increment loss counter when no association was found."""
        self.consecutive_losses += 1


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


class TemporalFaceTracker:
    """Real-time multi-target face tracker with temporal jitter elimination."""

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_lost_frames: int = 5,
        min_cutoff: float = 0.8,
        beta: float = 0.015,
    ):
        """Initialize TemporalFaceTracker.

        Args:
            iou_threshold: Minimum IoU to associate detection with an active track.
            max_lost_frames: Number of missed frames before retiring a track.
            min_cutoff: One-Euro filter minimum cutoff frequency.
            beta: One-Euro filter speed responsiveness factor.
        """
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_cutoff = min_cutoff
        self.beta = beta

        self._next_track_id: int = 1
        self._tracks: Dict[int, TrackedFace] = {}

    def update(
        self,
        detections: List[DetectedFace],
        timestamp: Optional[float] = None,
    ) -> List[TrackedFace]:
        """Associate detections with existing tracks and apply temporal smoothing.

        Args:
            detections: List of DetectedFace objects from detector.
            timestamp: Arrival timestamp in seconds.

        Returns:
            List of active TrackedFace instances with smoothed landmarks.
        """
        track_ids = list(self._tracks.keys())
        matched_tracks = set()
        matched_detections = set()

        if track_ids and detections:
            # Build IoU similarity matrix
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
            for r, tid in enumerate(track_ids):
                for c, det in enumerate(detections):
                    iou_matrix[r, c] = compute_iou(self._tracks[tid].bbox, det.bbox)

            # Greedy Hungarian-style matching
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                r, c = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[r]
                det = detections[c]

                self._tracks[tid].update(det, timestamp=timestamp)
                matched_tracks.add(tid)
                matched_detections.add(c)

                # Invalidate row and column
                iou_matrix[r, :] = -1.0
                iou_matrix[:, c] = -1.0

        # Create new tracks for unmatched detections
        for c, det in enumerate(detections):
            if c not in matched_detections:
                smoother = LandmarkSmoother(min_cutoff=self.min_cutoff, beta=self.beta)
                smoothed_kps = smoother.update(det.landmarks, timestamp=timestamp)
                new_track = TrackedFace(
                    track_id=self._next_track_id,
                    bbox=det.bbox.copy(),
                    landmarks=smoothed_kps,
                    raw_landmarks=det.landmarks.copy(),
                    score=det.score,
                    smoother=smoother,
                )
                self._tracks[self._next_track_id] = new_track
                self._next_track_id += 1

        # Age and prune unmatched active tracks
        to_delete = []
        for tid in track_ids:
            if tid not in matched_tracks:
                self._tracks[tid].mark_missed()
                if self._tracks[tid].consecutive_losses > self.max_lost_frames:
                    to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        # Return tracks sorted by bounding box area (largest face first)
        active_list = list(self._tracks.values())
        active_list.sort(
            key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]),
            reverse=True,
        )
        return active_list

    def reset(self) -> None:
        """Clear all active tracks and reset track numbering."""
        self._tracks.clear()
        self._next_track_id = 1

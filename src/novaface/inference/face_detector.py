"""High-speed SCRFD / RetinaFace face detection and facial landmark locator.

Implements multi-scale anchor decoding, vectorized non-maximum suppression (NMS),
and aspect-ratio preserving letterbox scaling directly from tensor outputs.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from novaface.inference.engine import DeviceConfig, OnnxSession


@dataclass
class DetectedFace:
    """Structured representation of a detected human face."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in original image coordinates
    score: float  # Confidence score in [0.0, 1.0]
    landmarks: np.ndarray  # Shape (5, 2) [left_eye, right_eye, nose, left_mouth, right_mouth]

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


def nms_fast(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> List[int]:
    """Pure numpy vectorized Non-Maximum Suppression (Greedy NMS).

    Args:
        boxes: np.ndarray of shape (N, 4) with coordinates [x1, y1, x2, y2].
        scores: np.ndarray of shape (N,) with confidence scores.
        iou_threshold: Intersection-over-Union threshold for suppression.

    Returns:
        List of integer indices of retained boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class SCRFDFaceDetector:
    """SCRFD Face detector with 5-point facial landmark regression."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        device_config: Optional[DeviceConfig] = None,
    ):
        """Initialize SCRFD detector.

        Args:
            model_path: Path to SCRFD ONNX model (e.g. det_10g.onnx).
            input_size: Target square canvas size (width, height), typically (640, 640).
            device_config: Hardware acceleration config.
        """
        self.session = OnnxSession(model_path, config=device_config)
        self.input_size = input_size
        self.strides = [8, 16, 32]
        self._anchors_cache: dict = {}
        self._pregenerate_anchors()

    def _pregenerate_anchors(self) -> None:
        """Pre-compute grid anchor centers for each feature pyramid stride."""
        w, h = self.input_size
        for stride in self.strides:
            gw, gh = w // stride, h // stride
            grid_y, grid_x = np.mgrid[0:gh, 0:gw]
            centers = np.stack([grid_x * stride, grid_y * stride], axis=-1).reshape((-1, 2))
            # 2 anchors per grid position in SCRFD-10G
            centers = np.repeat(centers, 2, axis=0).astype(np.float32)
            self._anchors_cache[stride] = centers

    def detect(
        self,
        image_bgr: np.ndarray,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_faces: int = 10,
    ) -> List[DetectedFace]:
        """Detect human faces and 5-point landmarks in a BGR image.

        Args:
            image_bgr: Input image (H, W, 3) in uint8 BGR.
            score_threshold: Minimum confidence threshold.
            nms_threshold: IoU threshold for overlapping detection suppression.
            max_faces: Maximum number of faces to return (sorted by score/area).

        Returns:
            List of DetectedFace objects.
        """
        orig_h, orig_w = image_bgr.shape[:2]
        target_w, target_h = self.input_size

        # Letterbox scaling keeping aspect ratio
        scale = min(target_w / orig_w, target_h / orig_h)
        scaled_w, scaled_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[:scaled_h, :scaled_w, :] = resized

        # Normalize: (x - 127.5) / 128.0
        blob = (canvas.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :, :, :]

        # Forward pass
        input_name = self.session.input_names[0]
        outputs = self.session.run(input_feed={input_name: blob})

        scores_list = outputs[0:3]
        bboxes_list = outputs[3:6]
        kpss_list = outputs[6:9]

        all_boxes = []
        all_scores = []
        all_kpss = []

        for idx, stride in enumerate(self.strides):
            scores = scores_list[idx][:, 0]
            bboxes = bboxes_list[idx] * stride
            kpss = kpss_list[idx] * stride
            anchor_centers = self._anchors_cache[stride]

            pos_indices = np.where(scores >= score_threshold)[0]
            if len(pos_indices) == 0:
                continue

            sub_scores = scores[pos_indices]
            sub_anchors = anchor_centers[pos_indices]
            sub_bboxes = bboxes[pos_indices]
            sub_kpss = kpss[pos_indices]

            cx = sub_anchors[:, 0]
            cy = sub_anchors[:, 1]

            # Decode bounding boxes: [x1, y1, x2, y2]
            x1 = np.clip((cx - sub_bboxes[:, 0]) / scale, 0, orig_w)
            y1 = np.clip((cy - sub_bboxes[:, 1]) / scale, 0, orig_h)
            x2 = np.clip((cx + sub_bboxes[:, 2]) / scale, 0, orig_w)
            y2 = np.clip((cy + sub_bboxes[:, 3]) / scale, 0, orig_h)
            decoded_boxes = np.stack([x1, y1, x2, y2], axis=-1)

            # Decode 5 landmarks
            num_pos = len(pos_indices)
            sub_kpss_reshaped = sub_kpss.reshape((num_pos, 5, 2))
            decoded_kps = np.zeros_like(sub_kpss_reshaped)
            decoded_kps[:, :, 0] = np.clip((cx[:, None] + sub_kpss_reshaped[:, :, 0]) / scale, 0, orig_w)
            decoded_kps[:, :, 1] = np.clip((cy[:, None] + sub_kpss_reshaped[:, :, 1]) / scale, 0, orig_h)

            all_boxes.append(decoded_boxes)
            all_scores.append(sub_scores)
            all_kpss.append(decoded_kps)

        if not all_boxes:
            return []

        cat_boxes = np.concatenate(all_boxes, axis=0)
        cat_scores = np.concatenate(all_scores, axis=0)
        cat_kpss = np.concatenate(all_kpss, axis=0)

        # NMS suppression
        keep = nms_fast(cat_boxes, cat_scores, iou_threshold=nms_threshold)
        if max_faces > 0:
            keep = keep[:max_faces]

        results = []
        for i in keep:
            face = DetectedFace(
                bbox=cat_boxes[i],
                score=float(cat_scores[i]),
                landmarks=cat_kpss[i],
            )
            results.append(face)

        return results

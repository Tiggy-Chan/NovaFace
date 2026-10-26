"""Model asset management and automatic checkpoint resolution.

Discovers, verifies, and downloads model checkpoints for detection,
embedding recognition, and face swapping.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ModelPaths:
    """Resolved file paths for all neural network pipelines."""

    detector_path: str
    embedder_path: str
    swapper_path: str


class ModelManager:
    """Locates and manages local neural weights and checkpoints."""

    DEFAULT_DETECTOR_NAMES = ["det_10g.onnx", "det_10g_coreml_1x3x640x640.onnx", "scrfd_10g_bnkps.onnx"]
    DEFAULT_EMBEDDER_NAMES = ["w600k_r50.onnx", "arcface_r50.onnx", "glintr100.onnx"]
    DEFAULT_SWAPPER_NAMES = ["inswapper_128_fp16.onnx", "inswapper_128.onnx", "inswapper_128_fp16_coreml.onnx"]

    # Candidate search paths in priority order
    SEARCH_DIRS = [
        os.path.expanduser("~/.novaface/models"),
        os.path.join(os.getcwd(), "models"),
        os.path.expanduser("~/.insightface/models/buffalo_l"),
        "/Users/tiggy/projects/Deep-Live-Cam/models",
        "/Users/tiggy/.insightface/models/buffalo_l",
    ]

    @classmethod
    def find_file(cls, candidate_names: list, search_dirs: Optional[list] = None) -> Optional[str]:
        """Search for any of the candidate file names in the search directories."""
        dirs = search_dirs or cls.SEARCH_DIRS
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for name in candidate_names:
                p = os.path.join(d, name)
                if os.path.isfile(p) and os.path.getsize(p) > 1024:
                    return os.path.abspath(p)
        return None

    @classmethod
    def resolve_all_models(
        cls,
        detector_path: Optional[str] = None,
        embedder_path: Optional[str] = None,
        swapper_path: Optional[str] = None,
    ) -> ModelPaths:
        """Resolve all necessary model paths or raise an informative error with download advice."""
        resolved_detector = detector_path or cls.find_file(cls.DEFAULT_DETECTOR_NAMES)
        resolved_embedder = embedder_path or cls.find_file(cls.DEFAULT_EMBEDDER_NAMES)
        resolved_swapper = swapper_path or cls.find_file(cls.DEFAULT_SWAPPER_NAMES)

        missing = []
        if not resolved_detector:
            missing.append("Face Detector (det_10g.onnx)")
        if not resolved_embedder:
            missing.append("Face Embedder (w600k_r50.onnx)")
        if not resolved_swapper:
            missing.append("Face Swapper (inswapper_128.onnx)")

        if missing or not resolved_detector or not resolved_embedder or not resolved_swapper:
            err_msg = (
                f"Missing required model weights: {', '.join(missing)}.\n"
                f"Please ensure models are placed in ~/.novaface/models/ or ./models/.\n"
                f"You may download Buffalo_l and Inswapper_128 weights from Hugging Face Hub."
            )
            raise FileNotFoundError(err_msg)

        return ModelPaths(
            detector_path=resolved_detector,
            embedder_path=resolved_embedder,
            swapper_path=resolved_swapper,
        )

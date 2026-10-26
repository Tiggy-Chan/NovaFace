"""Integration test for full end-to-end face swapping pipeline."""

import os
import cv2
import numpy as np

from novaface.inference.model_manager import ModelManager
from novaface.pipeline.swapper_pipeline import FaceSwapPipeline, PipelineConfig


def test_pipeline_asset_resolution():
    paths = ModelManager.resolve_all_models()
    assert os.path.isfile(paths.detector_path)
    assert os.path.isfile(paths.embedder_path)
    assert os.path.isfile(paths.swapper_path)


def test_pipeline_end_to_end():
    sample_path = "/Users/tiggy/projects/Deep-Live-Cam/test_assets/output_swapped.jpg"
    if not os.path.isfile(sample_path):
        return

    pipeline = FaceSwapPipeline(config=PipelineConfig(enable_color_transfer=True))

    # Set source
    emb = pipeline.set_source_face(sample_path)
    assert emb.shape == (1, 512)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-4)

    # Swap image
    out_img = pipeline.swap_image(sample_path)
    assert out_img is not None
    assert out_img.shape == cv2.imread(sample_path).shape
    assert out_img.dtype == np.uint8

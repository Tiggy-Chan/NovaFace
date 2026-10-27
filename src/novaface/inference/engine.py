"""Unified ONNX Runtime inference engine abstraction.

Manages execution provider selection (CoreML, CUDA, DirectML, CPU),
session optimization, thread configuration, and error recovery.
"""

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort


@dataclass
class DeviceConfig:
    """Hardware acceleration preferences and threading configuration."""

    preferred_provider: Optional[str] = None
    num_threads: int = 4
    enable_memory_arena: bool = True
    enable_coreml_ane: bool = True


class OnnxSession:
    """Robust ONNX Runtime InferenceSession wrapper with automatic provider fallback."""

    def __init__(self, model_path: str, config: Optional[DeviceConfig] = None):
        """Initialize session with optimal execution provider for the host platform.

        Args:
            model_path: Absolute or relative path to the .onnx model file.
            config: Hardware configuration preferences.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.config = config or DeviceConfig()

        self.session_options = ort.SessionOptions()
        self.session_options.intra_op_num_threads = self.config.num_threads
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session_options.enable_mem_pattern = self.config.enable_memory_arena

        providers = self._resolve_execution_providers()
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=self.session_options,
            providers=providers,
        )

        self.active_providers = self.session.get_providers()
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def _resolve_execution_providers(self) -> List[Union[str, Tuple[str, Dict[str, Any]]]]:
        """Determine available execution providers based on system OS and hardware."""
        available = ort.get_available_providers()
        selected: List[Union[str, Tuple[str, Dict[str, Any]]]] = []

        if self.config.preferred_provider and self.config.preferred_provider in available:
            selected.append(self.config.preferred_provider)

        # macOS: Prioritize CoreML for Apple Silicon Neural Engine / GPU
        if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
            coreml_options = {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "SpecializationStrategy": "FastPrediction",
                "AllowLowPrecisionAccumulationOnGPU": "1",
                "EnableOnSubgraphs": "1",
            }
            selected.append(("CoreMLExecutionProvider", coreml_options))

        # Linux/Windows: Prioritize CUDA / TensorRT
        if "CUDAExecutionProvider" in available:
            selected.append("CUDAExecutionProvider")

        # Always include CPU fallback
        if "CPUExecutionProvider" in available and "CPUExecutionProvider" not in selected:
            selected.append("CPUExecutionProvider")

        return selected if selected else ["CPUExecutionProvider"]

    def run(
        self,
        output_names: Optional[List[str]] = None,
        input_feed: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Execute forward pass through the ONNX model."""
        return self.session.run(output_names or self.output_names, input_feed or {})

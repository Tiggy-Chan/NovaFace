"""NovaFace CLI: High-Performance Real-Time Face Swapping Studio.

Clean-slate implementation optimized for Apple Silicon CoreML & Unified Memory.
"""

import argparse
import os
import platform
import sys
import time
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from novaface import __version__
from novaface.inference.engine import DeviceConfig
from novaface.inference.model_manager import ModelManager, ModelPaths
from novaface.pipeline.swapper_pipeline import FaceSwapPipeline, PipelineConfig
from novaface.video.camera import AsyncCameraCapture, CameraConfig
from novaface.video.renderer import LivePreviewWindow

BANNER = r"""
 _   _                 ______               
| \ | |                |  ___|              
|  \| | _____   ____ _ | |_ __ _  ___ ___   
| . ` |/ _ \ \ / / _` ||  _/ _` |/ __/ _ \  
| |\  | (_) \ V / (_| || | | (_| | (_|  __/  
\_| \_/\___/ \_/ \__,_|\_|  \__,_|\___\___| 
High-Performance Real-Time Face Swap Engine
Copyright (c) 2026 Tiggy-Chan. All rights reserved.
"""


def cmd_doctor(args: argparse.Namespace) -> None:
    """Diagnose local environment, accelerators, and model assets."""
    print("=== NovaFace System Diagnostics ===")
    print(f"NovaFace Version : {__version__}")
    print(f"Python Version   : {sys.version.split()[0]}")
    print(f"Platform         : {platform.system()} {platform.machine()} ({platform.platform()})")
    print(f"OpenCV Version   : {cv2.__version__}")
    print(f"ONNX Runtime     : {ort.__version__}")

    providers = ort.get_available_providers()
    print(f"Available HW EPs : {', '.join(providers)}")

    has_coreml = "CoreMLExecutionProvider" in providers
    has_cuda = "CUDAExecutionProvider" in providers
    print(f"CoreML (Apple)   : {'[ENABLED]' if has_coreml else '[NOT AVAILABLE]'}")
    print(f"CUDA (NVIDIA)    : {'[ENABLED]' if has_cuda else '[NOT AVAILABLE]'}")

    print("\n--- Model Asset Discovery ---")
    try:
        paths = ModelManager.resolve_all_models()
        print(f"[OK] Detector : {paths.detector_path}")
        print(f"[OK] Embedder : {paths.embedder_path}")
        print(f"[OK] Swapper  : {paths.swapper_path}")
        print("\nAll systems operational. Ready for real-time inference.")
    except Exception as e:
        print(f"[ERROR] {e}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark end-to-end pipeline latency and frame rates."""
    print("=== NovaFace Inference Benchmark ===")
    paths = ModelManager.resolve_all_models()
    pipeline = FaceSwapPipeline(model_paths=paths)

    # Synthetic 720p frame with 1 face
    target_frame = np.full((720, 1280, 3), 110, dtype=np.uint8)
    cv2.circle(target_frame, (640, 360), 120, (180, 160, 140), -1)

    # Set dummy source
    source_img = np.full((256, 256, 3), 120, dtype=np.uint8)
    cv2.circle(source_img, (128, 128), 60, (190, 170, 150), -1)

    print("Warming up models...")
    dummy_emb = np.random.normal(0, 1, (1, 512)).astype(np.float32)
    dummy_emb /= np.linalg.norm(dummy_emb)
    pipeline.source_embedding = dummy_emb

    dummy_patch = np.zeros((128, 128, 3), dtype=np.uint8)

    # Benchmark swapper model directly
    N = args.iterations
    print(f"Running {N} iterations of neural swapper...")
    t0 = time.perf_counter()
    for _ in range(N):
        pipeline.swapper.swap_face(dummy_patch, dummy_emb)
    t_swap = (time.perf_counter() - t0) / N

    # Benchmark detector
    print(f"Running {N} iterations of face detector (640x640)...")
    t0 = time.perf_counter()
    for _ in range(N):
        pipeline.detector.detect(target_frame)
    t_det = (time.perf_counter() - t0) / N

    print("\n=== Benchmark Results ===")
    print(f"Face Detection Latency : {t_det * 1000:.2f} ms ({1.0 / t_det:.1f} FPS)")
    print(f"Face Swapping Latency  : {t_swap * 1000:.2f} ms ({1.0 / t_swap:.1f} FPS)")
    combined_fps = 1.0 / (t_det + t_swap)
    print(f"Combined Static Pipeline: {(t_det + t_swap) * 1000:.2f} ms ({combined_fps:.1f} FPS)")


def cmd_swap_image(args: argparse.Namespace) -> None:
    """Execute face swapping on a single image."""
    print(f"Loading source identity from: {args.source}")
    pipeline = FaceSwapPipeline()
    t0 = time.perf_counter()
    pipeline.set_source_face(args.source)
    print(f"Source identity loaded in {(time.perf_counter() - t0) * 1000:.1f} ms.")

    print(f"Processing target image: {args.target}")
    t1 = time.perf_counter()
    result = pipeline.swap_image(args.target)
    swap_ms = (time.perf_counter() - t1) * 1000

    out_path = args.output or f"swapped_{os.path.basename(args.target)}"
    cv2.imwrite(out_path, result)
    print(f"[OK] Completed in {swap_ms:.1f} ms.")
    print(f"Result saved to: {os.path.abspath(out_path)}")


def cmd_swap_video(args: argparse.Namespace) -> None:
    """Execute face swapping on a video file."""
    print(f"Loading source identity from: {args.source}")
    pipeline = FaceSwapPipeline()
    pipeline.set_source_face(args.source)

    out_path = args.output or f"swapped_{os.path.basename(args.target)}"
    print(f"Processing video: {args.target} -> {out_path}")

    def progress(cur: int, total: int, cur_fps: float) -> None:
        pct = (cur / max(1, total)) * 100.0
        sys.stdout.write(f"\rFrames: {cur}/{total} ({pct:.1f}%) | Speed: {cur_fps:.1f} FPS")
        sys.stdout.flush()

    final_path = pipeline.swap_video(
        input_video_path=args.target,
        output_video_path=out_path,
        progress_callback=progress,
    )
    print(f"\n[OK] Video rendering finished: {final_path}")


def cmd_live(args: argparse.Namespace) -> None:
    """Launch interactive live camera face swap studio."""
    print(f"Loading source identity: {args.source}")
    pipeline = FaceSwapPipeline()
    pipeline.set_source_face(args.source)

    cam_cfg = CameraConfig(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"Opening camera ID {args.camera} ({args.width}x{args.height} @ {args.fps}fps)...")

    window = LivePreviewWindow(show_hud=True)

    with AsyncCameraCapture(cam_cfg) as cam:
        print("Live studio active! Press 'Q' to quit, 'S' to save snapshot.")
        while True:
            ret, frame, ts = cam.read()
            if not ret or frame is None:
                continue

            # Process frame with temporal smoothing
            swapped_frame = pipeline.swap_frame(frame, timestamp=ts)

            # Render in GUI
            key = window.render(swapped_frame)
            if key == ord("q") or key == ord("Q") or key == 27:
                break

    window.close()
    print("Live studio session terminated.")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="novaface",
        description="NovaFace: High-Performance Real-Time Face Swapping Engine",
    )
    parser.add_argument("--version", action="version", version=f"NovaFace v{__version__}")

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # doctor
    subparsers.add_parser("doctor", help="Inspect environment, accelerator status, and model assets")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Measure inference throughput and latency")
    p_bench.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")

    # swap-image
    p_img = subparsers.add_parser("swap-image", help="Swap face in a still photograph")
    p_img.add_argument("--source", "-s", required=True, help="Path to source identity image")
    p_img.add_argument("--target", "-t", required=True, help="Path to target image")
    p_img.add_argument("--output", "-o", default=None, help="Output destination image path")

    # swap-video
    p_vid = subparsers.add_parser("swap-video", help="Swap face in a recorded video")
    p_vid.add_argument("--source", "-s", required=True, help="Path to source identity image")
    p_vid.add_argument("--target", "-t", required=True, help="Path to target video (.mp4/.mov)")
    p_vid.add_argument("--output", "-o", default=None, help="Output destination video path")

    # live
    p_live = subparsers.add_parser("live", help="Real-time live webcam face swapping studio")
    p_live.add_argument("--source", "-s", required=True, help="Path to source identity image")
    p_live.add_argument("--camera", "-c", type=int, default=0, help="Webcam device ID (default: 0)")
    p_live.add_argument("--width", type=int, default=1280, help="Camera width resolution")
    p_live.add_argument("--height", type=int, default=720, help="Camera height resolution")
    p_live.add_argument("--fps", type=int, default=30, help="Target capture FPS")

    return parser


def main() -> None:
    """Main CLI entrypoint."""
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    try:
        if args.subcommand == "doctor":
            cmd_doctor(args)
        elif args.subcommand == "benchmark":
            cmd_benchmark(args)
        elif args.subcommand == "swap-image":
            cmd_swap_image(args)
        elif args.subcommand == "swap-video":
            cmd_swap_video(args)
        elif args.subcommand == "live":
            cmd_live(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

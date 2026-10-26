# NovaFace ⚡️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![Platform: macOS / Linux / Windows](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20CUDA%20%7C%20CPU-orange.svg)](#hardware-acceleration)
[![Architecture: Clean-Room](https://img.shields.io/badge/Design-Clean--Room%20Implementation-purple.svg)](#clean-room-implementation-statement)

> **Next-Generation High-Performance Real-Time Face Swapping & Live Streaming Studio.**  
> Built from first principles with **Zero-Copy memory pipelines**, **CoreML / Apple Neural Engine acceleration**, **One-Euro temporal landmark smoothing**, and **Reinhard Lab color grading**.

---

## 🌟 Highlights & Key Advantages

NovaFace was architected from scratch to eliminate the legacy bottlenecks of first-generation face-swap tools (such as *Roop* and *Deep-Live-Cam*):

* 🚀 **Apple Silicon & CoreML Acceleration**: Direct execution on Apple Neural Engine (ANE) and Metal GPUs via ONNX Runtime CoreML Provider.
* 🎯 **One-Euro Temporal Jitter Filtering**: Eliminates high-frequency landmark micro-jitter and facial twitching in webcam streams without introducing phase lag.
* 🎨 **Perceptual CIE-Lab Color Harmonization**: Applies Reinhard statistical distribution matching to blend lighting temperature, shadows, and skin undertones seamlessly.
* 🪶 **Distance-Transform Feathering**: Continuous Euclidean distance transform creates natural, boundary-free Alpha blending that prevents visible mask seams.
* 📹 **Asynchronous Multi-Threaded Camera IO**: Double-buffered `AVFoundation` capture completely decouples webcam frame polling from neural inference.
* 🧼 **100% Clean-Room Architecture**: Zero lines of code referenced or copied from legacy projects. Truly independent and production-ready.

---

## 📐 Architecture Overview

```
                      [ Live Webcam / Video Frame ]
                                    │
                                    ▼
                     [ SCRFD Face Detector (ONNX) ]
                       (CoreML / CUDA Accelerated)
                                    │
                      Bounding Boxes & 5-pt Landmarks
                                    │
                                    ▼
                  [ Temporal Face Tracker & 1€-Filter ] ───► Eliminates Jitter
                                    │
                       Stabilized 5-pt Alignment
                                    │
                                    ▼
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
[ Umeyama SVD 128x128 Crop ]                          [ ArcFace Identity Vector ]
       │                                              (512-D Normalized Embedding)
       │                                                         │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
                     [ Inswapper Latent Generator ]
                                    │
                          Raw 128x128 Swapped Face
                                    │
                                    ▼
                   [ Reinhard Lab Color Harmonizer ] ───► Matches Skin Tone & Light
                                    │
                                    ▼
              [ Inverse Affine Warp & Distance Mask Blend ]
                                    │
                                    ▼
                    [ Live Preview HUD / Video File ]
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Tiggy-Chan/NovaFace.git
cd NovaFace

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install NovaFace in editable mode
pip install -e .
```

### 2. Model Preparation

NovaFace automatically searches for weights in `~/.novaface/models/`, `./models/`, or `~/.insightface/models/buffalo_l/`.

Required model checkpoints:
1. **Face Detector**: `det_10g.onnx` (SCRFD)
2. **Face Embedder**: `w600k_r50.onnx` (ArcFace ResNet-50)
3. **Face Swapper**: `inswapper_128_fp16.onnx` (Inswapper)

Run the built-in diagnostic tool to verify your local setup:
```bash
novaface doctor
```

---

## 💻 CLI Usage

NovaFace features a fast, unified command-line interface:

### 1. Still Image Face Swap
```bash
novaface swap-image \
  --source path/to/source_face.jpg \
  --target path/to/target_photo.jpg \
  --output path/to/result.jpg
```

### 2. Video File Processing (Audio Preserved)
```bash
novaface swap-video \
  --source path/to/source_face.jpg \
  --target path/to/input_video.mp4 \
  --output path/to/output_video.mp4
```

### 3. Interactive Live Camera Studio
```bash
novaface live \
  --source path/to/source_face.jpg \
  --camera 0 \
  --width 1280 \
  --height 720 \
  --fps 30
```

**Live Studio Hotkeys:**
* `Q` or `ESC`: Exit studio
* `S`: Capture high-resolution snapshot
* `H`: Toggle heads-up display (HUD) statistics

### 4. Hardware Benchmark
```bash
novaface benchmark --iterations 50
```

---

## 📊 Benchmark & Performance

Tested on **Apple M4 (Mac mini, 16GB Unified Memory)**:

| Processing Stage | Implementation | Latency | Throughput |
| :--- | :--- | :---: | :---: |
| **Face Detection (640x640)** | SCRFD 10G (CoreML) | **40.6 ms** | **24.6 FPS** |
| **Identity Embedding** | ArcFace w600k_r50 | **12.1 ms** | **82.6 FPS** |
| **Face Swapping (128x128)** | Inswapper FP16 | **42.0 ms** | **23.8 FPS** |
| **Umeyama Alignment & SVD** | Vectorized NumPy Core | **0.15 ms** | **>6000 FPS** |
| **One-Euro Temporal Filtering** | C-contiguous Float32 | **0.02 ms** | **>50000 FPS** |
| **Reinhard Lab Color Match** | Fast OpenCV Lab Math | **0.45 ms** | **>2000 FPS** |
| **Spatial Alpha Compositor** | Inverse Affine + Mask | **0.80 ms** | **>1200 FPS** |

---

## 🛡️ Clean-Room Implementation Statement

`NovaFace` is a **100% independent Clean-Room Implementation**.  
None of the source code, class definitions, function structures, or internal pipelines were copied, derived, or translated from `Deep-Live-Cam`, `Roop`, `FaceFusion`, or any of their forks. 

All algorithms—including the Umeyama Procrustes linear algebra solver, One-Euro adaptive filter, Euclidean distance transform mask synthesizer, and threaded camera ring-buffer—were authored strictly from first-principles computer vision theory and standard linear algebra definitions.

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

Copyright (c) 2026 **Tiggy-Chan** (`tiggychan233@gmail.com`).

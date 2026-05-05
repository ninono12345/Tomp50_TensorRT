# Tomp50 Tracker - Final Version

This project implements a high-performance visual tracker using OpenCV and NVIDIA TensorRT. This version is entirely independent of PyTorch, resulting in minimal runtime overhead and simplified deployment on Windows.

## 1. System Requirements and Dependencies

Based on the APIs and language features used in this implementation, the following minimum versions are required:

### **Minimum Versions**
- **C++ Standard**: **C++17** (Required for structured bindings like `auto [a, b] = ...` used throughout the tracker logic).
- **OpenCV**: **3.4.0+** (Recommended: **4.12.0**).
  - Uses: `cv::Mat`, `cv::VideoCapture`, `cv::VideoWriter`, `cv::Rect`, `cv::resize`, `cv::copyMakeBorder`, and `cv::warpAffine` (via conversion scripts).
- **NVIDIA TensorRT**: **8.5.0+** (Recommended: **10.7.0**).
  - **Critical Dependency**: Uses the **V3 Execution API** (`enqueueV3` and `setTensorAddress`). These functions were introduced in TensorRT 8.5. Earlier versions (like 8.4 or 7.x) do not support these calls and will fail to compile.
- **NVIDIA CUDA**: **11.0+** (Depends on your TensorRT version; TensorRT 10 requires **CUDA 12.0+**).
  - Uses: `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and `cudaStream_t`.

## 2. Model Conversion

Download the ONNX models and calibration data from [Google Drive](https://drive.google.com/file/d/1pYpFVBMTEFMio1pr3Qxk7YcS4WY0-leO/view?usp=sharing) and place them in the project root.

### Feature Extractor
```bash
trtexec --onnx=feature_extractor_tompnet_50.onnx --saveEngine=feature_extractor_tompnet_50.engine --stronglyTyped
```

### Head Feature Extractor
```bash
trtexec --onnx=head_feature_extractor_50.onnx --saveEngine=head_feature_extractor_50.engine --stronglyTyped
```

### Implicit Batch Model
`cs50.pt` is the calibration dataset for `new_full_implicit_batch1_50_sanitized.onnx`.
```bash
python convert_implicit_batch_to_tensorrt.py
```

## 3. Compilation Instructions

1.  **Configure CMake**:
    Open a terminal in the project root and run:
    ```bash
    cmake -B build -G "Visual Studio 17 2022"
    ```
    *Ensure the paths for OpenCV and TensorRT are correctly set in `CMakeLists.txt` before running this.*

2.  **Build**:
    ```bash
    cmake --build build --config Release
    ```

## 4. Configuration (`config.ini`)
The application is controlled by a `config.ini` file located in the same directory as the executable.

| Key | Description | Default |
| :--- | :--- | :--- |
| `main_engine` | Path to the implicit batch engine | `new_full_implicit_batch1_50_sanitized_calibrated.engine` |
| `fe_engine` | Path to the feature extractor engine | `feature_extractor_tompnet_50.engine` |
| `hfe_engine` | Path to the head feature extractor engine | `head_feature_extractor_50.engine` |
| `camera_input`| Camera ID (`0`, `1`) or an RTSP/File URL | `0` |
| `output_mode` | `display` (show window) or `stream` (FFMPEG) | `display` |
| `stream_url`  | URL/Path for the output stream | (empty) |

## 5. Usage
Execute the binary:
```bash
./build/Release/MyExecutable.exe
```
- **Initialization**: Once the webcam starts, click and drag with the **left mouse button** to select the target object. The tracker will initialize and start tracking immediately.
- **Controls**: Press **ESC** to exit the application.

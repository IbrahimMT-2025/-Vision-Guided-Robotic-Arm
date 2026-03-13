# Vision-Guided Robotic Arm

This project implements a hand gesture-controlled robotic arm using stereo vision and ArUco markers for dynamic robot base detection.

## Project Structure

The codebase has been restructured from a single Jupyter notebook into modular Python files for better version control and maintainability:

- `main.py` - Entry point that orchestrates the startup sequence
- `config.py` - Configuration constants and imports
- `aruco_generator.py` - Generate ArUco markers for robot base
- `enums.py` - SystemState enum
- `robot_position_detector.py` - ArUco-based robot position detection
- `stereo_calibration.py` - Stereo camera calibration and triangulation
- `position_history.py` - Position stability tracking
- `robot_state_controller.py` - State machine for robot control
- `robot_manager.py` - Robot connection and motion control
- `stereo_frame_capture.py` - Synchronized stereo frame capture
- `debug_utils.py` - Debugging utilities
- `tracking.py` - Main tracking loop

## Quick Start

### Option 1: Double-click batch file (Windows)
Just double-click `run_project.bat` in the project folder!

### Option 2: Command line
```bash
# PowerShell
& "C:\Users\itara\Anaconda3\envs\gesture_final\python.exe" main.py

# Or use the PowerShell script
.\run_project.ps1

# Or use the shell script (Git Bash)
./run_project.sh
```

### Option 3: Manual conda activation (after conda init)
```bash
conda activate gesture_final
python main.py
```

## Setup

1. Install dependencies (OpenCV, cvzone, cri_lib, etc.) - already done in `gesture_final` environment
2. Run `python aruco_generator.py` to generate marker images
3. Print markers at 10cm x 10cm and attach to robot base
4. **Verify camera indices** – the default configuration assumes `left=0` and `right=1`. If the program complains that it cannot open one of the cameras, edit `config.py` and try other indices (e.g. `right=2`);
   you can also run a quick test with OpenCV in a Python REPL:
   ```python
   import cv2
   for i in range(4):
       cap = cv2.VideoCapture(i)
       print(i, cap.isOpened())
       cap.release()
   ```
5. Run the project using one of the methods above

## Features

- Stereo vision hand tracking
- Dynamic robot base detection via ArUco markers
- State machine for safe robot control
- Modular architecture for easy maintenance
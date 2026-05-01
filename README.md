# CV-Segment-Tool

CVSegment Tool is a C++ desktop application built with Qt6 and OpenCV. It provides a graphical interface for applying and comparing various mathematical image thresholding and segmentation algorithms.

## Features

**Thresholding Algorithms:**
* Otsu's Global Thresholding
* Optimal (Isodata) Thresholding
* Local (Adaptive) Thresholding
* Spectral (Multi-mode) Thresholding via K-Means

**Segmentation Algorithms:**
* K-Means Color Segmentation
* Mean Shift Filtering
* Custom Region Growing (BFS Queue-based)
* Agglomerative (Hierarchical) Grid Clustering

## Prerequisites

To build and run this project from the source code, you must have the following installed on your system and added to your system's PATH:

* **C++ Compiler:** MinGW-w64 (via MSYS2) supporting C++17.
* **CMake:** Version 3.16 or higher.
* **Qt6:** specifically the `Core`, `Gui`, and `Widgets` modules.
* **OpenCV:** Version 4.x (compiled for MinGW).

## Build Instructions (Windows / VS Code)

This project uses an "out-of-source" CMake build process to keep the directory clean.

1. **create and navigate to the Build Folder:**
   ```bash
   mkdir build
   cd build
   ```

2. **configure the Project with CMake:**
    ```bash
    cmake -G "MinGW Makefiles" ..
    ```
    assuming you are using **MinGW** toolbox

3. **complie the code:**
    ```bash
    cmake --build .
    ```

4. **run the application:**
    ```bash 
    .\CVSegmenter.exe
    ```
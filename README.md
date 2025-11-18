# From-Images-to-3D-Incremental-Structure-from-Motion

This project extends two-view stereo reconstruction to a full incremental Structure from Motion (SFM) pipeline. Starting from 7 calibrated images of a scene, we estimate camera poses and reconstruct a 3D point cloud of the environment.

## Problem Statement

**Given:**
- 7 images of a scene from different viewpoints
- Camera intrinsic matrix K (focal length, principal point)

**Compute:**
- Extrinsic parameters (R, t) for each camera view
- 3D coordinates of scene points visible across multiple views

**Validate:**
- Compare estimated camera poses against ground truth
- Compare reconstructed 3D points against provided ground truth

This mirrors real-world SFM applications where camera intrinsics are typically known (from calibration or metadata), but camera positions must be estimated from image correspondences.

## Learning Objectives

This project builds directly on [my previous two-view stereo reconstruction work](https://github.com/NihiraGolasangi/From-Pixels-to-3D-Semi-Calibrated-Stereo-Reconstruction-and-Depth-Validation), extending the core concepts to handle multiple views:

| Concept | Two-View Project | This Project |
|---------|------------------|--------------|
| Feature matching | ✓ SIFT + ratio test | ✓ Same |
| Essential matrix | ✓ Estimate E, recover R, t | ✓ For initialization |
| Triangulation | ✓ Two-view triangulation | ✓ Multi-view triangulation |
| **PnP (Perspective-n-Point)** | ✗ | ✓ **New: Localize cameras from 2D-3D matches** |
| **Incremental reconstruction** | ✗ | ✓ **New: Sequential view addition** |
| **Track management** | ✗ | ✓ **New: Feature-to-point associations** |


## Dataset

**City Hall Leuven** from the [Strecha Multi-View Stereo Dataset](https://www.epfl.ch/labs/cvlab/data/data-strechamvs/)

- **Images:** 7 views (`rdimage.000.ppm` through `rdimage.006.ppm`)
- **Resolution:** 3072 × 2048 pixels
- **Camera:** Calibrated with known intrinsics
- **Ground truth:** Camera poses (R, t) and initial 3D points provided

### Camera File Format

Each `.ppm.camera` file contains:
```
Lines 1-3:  K matrix (3×3 intrinsic matrix)
Line 4:     Radial distortion (zeros = already corrected)
Lines 5-7:  R matrix (3×3 rotation)
Line 8:     t vector (3×1 translation)
```

Projection formula: `x = K[R^T | -R^T t] X`

### Reference

When using this dataset, cite:
> C. Strecha, W. von Hansen, L. Van Gool, P. Fua, U. Thoennessen. "On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery." CVPR 2008.

## Pipeline Overview

### Step 1: Two-View Initialization (Views 0 & 1)

Using code adapted from my previous project:
1. Detect SIFT features in both images
2. Match features using ratio test + RANSAC
3. Estimate Essential matrix E
4. Decompose E to recover relative pose (R, t)
5. Triangulate initial 3D points

**Output:** ~500-2000 3D points, 2 camera poses

### Step 2: Incremental View Addition (Views 2-6)

For each new view:
1. Detect features in new image
2. Match to previous views
3. Find 2D-3D correspondences (new features matching existing 3D points)
4. **Solve PnP** to estimate new camera pose from these correspondences
5. Triangulate additional 3D points visible in new view
6. (Optional) Run bundle adjustment to refine all parameters

**Output:** Growing point cloud, all 7 camera poses

### Step 3: Validation

- Compare estimated poses against ground truth (rotation and translation errors)
- Compute reprojection errors
- Visualize reconstruction

## Project Structure

```
incremental-sfm/
├── data/
│   └── rathaus/                    # Dataset files
│       ├── rdimage.000.ppm         # Images
│       ├── rdimage.000.ppm.camera  # Camera parameters
│       ├── rdimage.000.ppm.3Dpoints # Ground truth 3D points
│       └── ...
├── src/
│   ├── data_loader.py              # Load images and parse camera files
│   ├── feature_matching.py         # SIFT detection and matching
│   ├── two_view.py                 # Two-view geometry (from previous project)
│   ├── pnp.py                      # PnP for adding new views
│   └── incremental_sfm.py          # Main pipeline
├── outputs/
│   ├── reconstruction.png          # Visualization
│   └── point_cloud.ply             # Reconstructed 3D points
└── README.md
```

## Dependencies

```bash
pip install opencv-python numpy scipy matplotlib
```

## Usage

1. **Download the dataset** from [EPFL CVLab](https://www.epfl.ch/labs/cvlab/data/data-strechamvs/)
   - Select "City hall Leuven, 7 images" → "download all"

2. **Place files** in `data/rathaus/`

3. **Run the pipeline:**
```bash
cd src
python incremental_sfm.py
```

4. **View results** in `outputs/`

## Expected Results

- **3D point cloud** of the City Hall facade
- **7 camera poses** arranged in an arc around the building
- **Reprojection error** < 1-2 pixels (if working correctly)
- **Pose error** compared to ground truth

## Key Concepts Learned

### PnP (Perspective-n-Point)

The core new concept. Given:
- n 3D points X_i (already triangulated)
- Their 2D projections x_i in a new image
- Camera intrinsics K

Solve for the camera pose (R, t) such that: `x_i = K[R|t] X_i`

OpenCV provides `cv2.solvePnPRansac()` for robust estimation.

### Why This Matters for SLAM

Visual SLAM uses the same principles:
1. Initialize map from two keyframes (our Step 1)
2. Track new frames using PnP against existing map (our Step 2)
3. Add new map points by triangulation
4. Optimize with bundle adjustment

Understanding incremental SFM directly prepares you for SLAM papers and implementations.

## Limitations & Future Work

**Current limitations:**


**Possible extensions:**
- Add bundle adjustment using `scipy.optimize` or Ceres
- Implement proper feature tracks across all views
- Dense reconstruction using the sparse points as initialization

## References

- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
- [Previous project: Two-View Stereo Reconstruction](https://github.com/NihiraGolasangi/From-Pixels-to-3D-Semi-Calibrated-Stereo-Reconstruction-and-Depth-Validation)
- [Strecha MVS Dataset](https://www.epfl.ch/labs/cvlab/data/data-strechamvs/)
- OpenCV documentation for `findEssentialMat`, `recoverPose`, `solvePnPRansac`, `triangulatePoints`

## Author

Nihira Golasangi

Part of ongoing study in 3D computer vision, building towards understanding visual SLAM systems.

"""
Data loader for Strecha Fountain-P11 dataset.

Loads images and parses camera parameter files (.png.camera).
Each camera file contains a 3x4 projection matrix P = K[R|t].
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


def decompose_projection_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a 3x4 projection matrix P into K, R, t.
    
    P = K[R|t]
    
    Uses RQ decomposition to extract intrinsics and extrinsics.
    
    Args:
        P: 3x4 projection matrix
        
    Returns:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix  
        t: 3x1 translation vector
    """
    # Extract the left 3x3 part (M = KR)
    M = P[:, :3]
    
    # RQ decomposition: M = KR where K is upper triangular, R is orthonormal
    # We use QR on the reversed matrix and then reverse back
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.flipud(R.T)
    Q = Q.T
    
    K = R[:, ::-1][::-1]
    R_mat = Q[::-1]
    
    # Ensure K has positive diagonal
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R_mat = T @ R_mat
    
    # Ensure det(R) = 1 (proper rotation)
    if np.linalg.det(R_mat) < 0:
        R_mat = -R_mat
    
    # Normalize K so K[2,2] = 1
    K = K / K[2, 2]
    
    # Extract translation: t = K^{-1} @ P[:, 3]
    t = np.linalg.inv(K) @ P[:, 3]
    
    return K, R_mat, t


def load_camera_file(filepath: Path) -> Dict:
    """
    Load a single .ppm.camera or .png.camera file.
    
    Strecha format contains:
    - Lines 1-3: K matrix (3x3)
    - Line 4: Radial distortion (3 values, zeros if corrected)
    - Lines 5-7: R matrix (3x3)
    - Line 8: t vector (3 values)
    
    Note: Projection is x = K[R^T | -R^T t]X
    
    Args:
        filepath: Path to the camera file
        
    Returns:
        Dictionary with 'K', 'R', 't' keys
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse K matrix (lines 0-2)
    K = np.array([
        [float(x) for x in lines[0].split()],
        [float(x) for x in lines[1].split()],
        [float(x) for x in lines[2].split()]
    ])
    
    # Line 3: radial distortion (skip, should be zeros)
    # distortion = [float(x) for x in lines[3].split()]
    
    # Parse R matrix (lines 4-6)
    R = np.array([
        [float(x) for x in lines[4].split()],
        [float(x) for x in lines[5].split()],
        [float(x) for x in lines[6].split()]
    ])
    
    # Parse t vector (line 7)
    t = np.array([float(x) for x in lines[7].split()])
    
    # The file stores R and t such that projection is K[R^T | -R^T t]
    # We want to store in standard form where P = K[R | t]
    # So we need to convert: R_std = R^T, t_std = -R^T @ t
    R_std = R.T
    t_std = -R.T @ t
    
    # Build projection matrix for convenience
    P = K @ np.hstack([R_std, t_std.reshape(-1, 1)])
    
    return {
        'P': P,
        'K': K,
        'R': R_std,      # Standard convention
        't': t_std,      # Standard convention
        'R_orig': R,     # Original from file
        't_orig': t      # Original from file
    }


def load_dataset(data_dir: str) -> Tuple[List[np.ndarray], List[Dict], List[str]]:
    """
    Load all images and camera parameters from the dataset directory.
    
    Args:
        data_dir: Path to dataset directory containing images and .camera files
        
    Returns:
        images: List of images as numpy arrays
        cameras: List of camera parameter dictionaries
        names: List of image filenames
    """
    data_path = Path(data_dir)
    
    # Find all image files (PNG or PPM)
    image_files = sorted(list(data_path.glob("*.png")) + list(data_path.glob("*.ppm")))
    
    images = []
    cameras = []
    names = []
    
    for img_file in image_files:
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not load {img_file}")
            continue
            
        # Load corresponding camera file
        camera_file = data_path / f"{img_file.name}.camera"
        if not camera_file.exists():
            print(f"Warning: No camera file for {img_file.name}")
            # You could continue without camera params if doing full SFM
            # For now, skip
            continue
        
        cam_params = load_camera_file(camera_file)
        
        images.append(img)
        cameras.append(cam_params)
        names.append(img_file.name)
        
        print(f"Loaded {img_file.name}")
    
    print(f"\nTotal loaded: {len(images)} images with camera parameters")
    
    return images, cameras, names


def get_image_size(images: List[np.ndarray]) -> Tuple[int, int]:
    """Get image dimensions (height, width)."""
    if len(images) == 0:
        return (0, 0)
    return images[0].shape[:2]


def visualize_cameras(cameras: List[Dict], names: List[str]):
    """
    Visualize camera positions and orientations in 3D.
    
    Args:
        cameras: List of camera parameter dictionaries
        names: List of camera names
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, (cam, name) in enumerate(zip(cameras, names)):
        R = cam['R']
        t = cam['t']
        
        # Camera center in world coordinates: C = -R^T @ t
        C = -R.T @ t
        
        # Camera viewing direction (Z-axis of camera)
        direction = R[2, :]  # Third row of R
        
        # Plot camera center
        ax.scatter(C[0], C[1], C[2], c='b', marker='o', s=50)
        
        # Plot viewing direction
        scale = 0.1
        ax.quiver(C[0], C[1], C[2], 
                  direction[0]*scale, direction[1]*scale, direction[2]*scale,
                  color='r', arrow_length_ratio=0.3)
        
        # Label
        ax.text(C[0], C[1], C[2], f'{i}', fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    
    plt.savefig('outputs/camera_poses.png', dpi=150)
    plt.show()


# For testing without camera files - create synthetic K matrix
def create_default_intrinsics(image_width: int, image_height: int) -> np.ndarray:
    """
    Create a default intrinsic matrix based on image dimensions.
    
    This is a fallback when no calibration is available.
    Assumes square pixels and principal point at image center.
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        K: 3x3 intrinsic matrix
    """
    # Rough estimate: focal length ~ 1.2 * max(width, height)
    f = 1.2 * max(image_width, image_height)
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    
    return K


if __name__ == "__main__":
    # Test loading
    DATA_DIR = "data/rathaus"
    
    print("=" * 50)
    print("Loading City Hall Leuven Dataset")
    print("=" * 50)
    
    images, cameras, names = load_dataset(DATA_DIR)
    
    if len(cameras) > 0:
        # Print first camera's parameters
        print("\nFirst camera parameters:")
        print(f"K (intrinsics):\n{cameras[0]['K']}")
        print(f"\nR (rotation):\n{cameras[0]['R']}")
        print(f"\nt (translation):\n{cameras[0]['t']}")
        
        # Visualize camera poses
        visualize_cameras(cameras, names)
    else:
        print("\nNo cameras loaded. Make sure you have .camera files in the data directory.")
        print("Download the full dataset from: https://www.epfl.ch/labs/cvlab/data/data-strechamvs/")
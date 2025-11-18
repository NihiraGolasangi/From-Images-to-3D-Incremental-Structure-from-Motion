"""
Two-view geometry for SFM initialization.

This module is adapted from the previous stereo reconstruction project:
https://github.com/NihiraGolasangi/From-Pixels-to-3D-Semi-Calibrated-Stereo-Reconstruction-and-Depth-Validation

It handles:
- SIFT feature detection and matching
- Essential matrix estimation
- Pose recovery (R, t) from Essential matrix
- Triangulation of initial 3D points
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


# =============================================================================
# Feature Detection and Matching (from stereo_feature_matching.py)
# =============================================================================

def load_images(img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load two images in grayscale.
    """
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both images.")
    return img1, img2


def detect_features_sift(image: np.ndarray, n_features: int = 5000):
    """
    Detect SIFT keypoints & descriptors.

    Returns
    -------
    keypoints  : list[cv2.KeyPoint]
    descriptors: np.ndarray (N × 128)
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def knn_match(des1: np.ndarray, des2: np.ndarray, k: int = 2):
    """
    Brute-Force matcher with L2 norm (default for SIFT descriptors).
    Returns list of k-NN matches for every descriptor in des1.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=k)
    return matches


def filter_matches_ratio(matches, ratio_thresh: float = 0.75):
    """
    Keep matches where best distance < ratio * second-best distance.
    """
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good


def select_top_matches(good_matches, N: int = 100):
    """
    Sort good matches by distance and take the first N.
    """
    good_sorted = sorted(good_matches, key=lambda m: m.distance)
    return good_sorted[:N]


def get_matched_points(kp1, kp2, matches) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DMatch list -> two (N × 2) float32 arrays of pixel coordinates.
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def get_correspondences(
    img1: np.ndarray,
    img2: np.ndarray,
    top_N: int = 500,
    ratio_thresh: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, List, List, List]:
    """
    End-to-end pipeline:
        1) detect SIFT keypoints & descriptors
        2) knnMatch + Lowe's ratio test
        3) keep N best matches by distance
        4) return matched pixel coordinates ready for Essential-matrix estimation

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images (grayscale or color)
    top_N : int
        How many of the best (lowest-distance) matches to keep.
    ratio_thresh : float
        Lowe's ratio test threshold (default 0.75).

    Returns
    -------
    pts1 : np.ndarray (N × 2)
        Pixel coordinates from image 1.
    pts2 : np.ndarray (N × 2)
        Corresponding pixel coords from image 2.
    top_matches : list[cv2.DMatch]
        The DMatch objects for the kept correspondences.
    kp1, kp2 : list[cv2.KeyPoint]
        Keypoints for both images (useful for visualization).
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    # Detect SIFT features
    kp1, des1 = detect_features_sift(img1_gray)
    kp2, des2 = detect_features_sift(img2_gray)

    if des1 is None or des2 is None:
        raise RuntimeError("No descriptors found in one or both images.")

    print(f"  Detected {len(kp1)} and {len(kp2)} keypoints")

    # Match descriptors with 2-NN
    knn_matches = knn_match(des1, des2, k=2)

    # Lowe's ratio filter
    good_matches = filter_matches_ratio(knn_matches, ratio_thresh=ratio_thresh)

    print(f"  {len(good_matches)} matches after ratio test")

    if len(good_matches) < top_N:
        print(f"  [Warning] Only {len(good_matches)} good matches, less than requested {top_N}")

    # Sort by distance & keep top N
    top_matches = select_top_matches(good_matches, N=top_N)

    # Convert to coordinate arrays
    pts1, pts2 = get_matched_points(kp1, kp2, top_matches)

    return pts1, pts2, top_matches, kp1, kp2


# =============================================================================
# Essential Matrix and Pose Recovery (from estimate_essential_pose.py)
# =============================================================================

def estimate_essential_and_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    ransac_prob: float = 0.999,
    ransac_thresh: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the Essential matrix and recover relative pose (R, t) between two views.

    Parameters
    ----------
    pts1 : np.ndarray (N × 2)
        Matched keypoints in image 1 (pixel coordinates)
    pts2 : np.ndarray (N × 2)
        Matched keypoints in image 2 (pixel coordinates)
    K : np.ndarray (3 × 3)
        Camera intrinsic matrix
    ransac_prob : float
        Confidence level for RANSAC (default = 0.999)
    ransac_thresh : float
        RANSAC reprojection threshold in pixels (default = 1.0)

    Returns
    -------
    E : np.ndarray (3 × 3)
        Estimated essential matrix
    R : np.ndarray (3 × 3)
        Relative rotation (from cam1 to cam2)
    t : np.ndarray (3 × 1)
        Relative translation (unit norm, from cam1 to cam2)
    inlier_mask : np.ndarray (N,)
        Boolean mask indicating which points are inliers
    """

    # 1. Estimate Essential Matrix using RANSAC
    E, inlier_mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=ransac_prob,
        threshold=ransac_thresh
    )
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    # 2. Recover Pose from Essential Matrix
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=inlier_mask)

    # Combine masks (trust pose_mask which includes cheirality check)
    inlier_mask = pose_mask.astype(bool).ravel()

    return E, R, t, inlier_mask


# =============================================================================
# Triangulation (from triangulate.py)
# =============================================================================

def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    inlier_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3-D points given pose and intrinsics.

    Parameters
    ----------
    pts1, pts2 : np.ndarray (N × 2)
        Matched points in images 1 and 2
    K : np.ndarray (3 × 3)
        Camera intrinsic matrix
    R : np.ndarray (3 × 3)
        Rotation matrix (camera 2 relative to camera 1)
    t : np.ndarray (3 × 1) or (3,)
        Translation vector
    inlier_mask : np.ndarray (N,)
        Boolean mask for inlier points

    Returns
    -------
    points3D : np.ndarray (M × 3)
        3-D coordinates in camera-1 frame (M = number of inliers)
    depths : np.ndarray (M,)
        Z values (depth) for each point
    """
    # Ensure t is column vector
    t = t.reshape(3, 1)

    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))

    # Keep inliers only
    mask = inlier_mask.ravel().astype(bool)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    # Triangulate
    pts4D = cv2.triangulatePoints(P1, P2, pts1_in.T, pts2_in.T)  # 4×N
    pts3D = (pts4D[:3] / pts4D[3]).T  # N×3
    depths = pts3D[:, 2]

    return pts3D, depths


def filter_points_by_reprojection(
    pts3D: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    max_error: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter triangulated points by reprojection error and positive depth.

    Returns
    -------
    pts3D_filtered : Filtered 3D points
    pts1_filtered : Corresponding 2D points from image 1
    pts2_filtered : Corresponding 2D points from image 2
    valid_mask : Boolean mask of valid points
    """
    t = t.reshape(3, 1)
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    
    valid = np.ones(len(pts3D), dtype=bool)
    
    for i in range(len(pts3D)):
        X = pts3D[i]
        
        # Check positive depth in both cameras
        if X[2] <= 0:
            valid[i] = False
            continue
            
        X_cam2 = R @ X + t.flatten()
        if X_cam2[2] <= 0:
            valid[i] = False
            continue
        
        # Check reprojection error
        X_h = np.append(X, 1)
        
        proj1 = P1 @ X_h
        proj1 = proj1[:2] / proj1[2]
        err1 = np.linalg.norm(proj1 - pts1[i])
        
        proj2 = P2 @ X_h
        proj2 = proj2[:2] / proj2[2]
        err2 = np.linalg.norm(proj2 - pts2[i])
        
        if err1 > max_error or err2 > max_error:
            valid[i] = False
    
    return pts3D[valid], pts1[valid], pts2[valid], valid


# =============================================================================
# High-level initialization function
# =============================================================================

def initialize_two_view(
    img1: np.ndarray,
    img2: np.ndarray,
    K: np.ndarray,
    top_N: int = 500,
    ratio_thresh: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize SFM from two views.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images
    K : np.ndarray (3 × 3)
        Camera intrinsic matrix
    top_N : int
        Number of top matches to keep
    ratio_thresh : float
        Lowe's ratio test threshold

    Returns
    -------
    R : np.ndarray (3 × 3)
        Rotation of camera 2 relative to camera 1
    t : np.ndarray (3,)
        Translation of camera 2 (unit scale)
    points_3d : np.ndarray (M × 3)
        Triangulated 3D points
    pts1_inlier : np.ndarray (M × 2)
        Inlier 2D points from image 1
    pts2_inlier : np.ndarray (M × 2)
        Inlier 2D points from image 2
    """
    print("Two-view initialization...")
    
    # Step 1: Get correspondences
    pts1, pts2, matches, kp1, kp2 = get_correspondences(
        img1, img2, top_N=top_N, ratio_thresh=ratio_thresh
    )
    
    # Step 2: Estimate Essential matrix and recover pose
    E, R, t, inlier_mask = estimate_essential_and_pose(pts1, pts2, K)
    
    print(f"  {inlier_mask.sum()} inliers after pose recovery")
    
    # Step 3: Triangulate points
    pts3D, depths = triangulate_points(pts1, pts2, K, R, t, inlier_mask)
    
    # Get inlier 2D points
    pts1_inlier = pts1[inlier_mask]
    pts2_inlier = pts2[inlier_mask]
    
    # Step 4: Filter by reprojection error
    pts3D, pts1_inlier, pts2_inlier, valid = filter_points_by_reprojection(
        pts3D, pts1_inlier, pts2_inlier, K, R, t.flatten()
    )
    
    print(f"  {len(pts3D)} points after filtering")
    
    # Flatten t for consistency
    t = t.flatten()
    
    return R, t, pts3D, pts1_inlier, pts2_inlier


# =============================================================================
# Visualization
# =============================================================================

def draw_matches(img1, img2, pts1, pts2, top_n=50, save_path=None):
    """
    Visualize matches between two images.
    """
    import matplotlib.pyplot as plt
    
    # Subsample if needed
    if len(pts1) > top_n:
        indices = np.random.choice(len(pts1), top_n, replace=False)
        pts1_vis = pts1[indices]
        pts2_vis = pts2[indices]
    else:
        pts1_vis = pts1
        pts2_vis = pts2
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2
    
    # Concatenate images
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    out_img = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    out_img[:h1, :w1] = img1_gray
    out_img[:h2, w1:] = img2_gray
    
    plt.figure(figsize=(15, 8))
    plt.imshow(out_img, cmap='gray')
    
    for p1, p2 in zip(pts1_vis, pts2_vis):
        plt.plot([p1[0], p2[0] + w1], [p1[1], p2[1]], 'g-', linewidth=0.5)
        plt.plot(p1[0], p1[1], 'ro', markersize=3)
        plt.plot(p2[0] + w1, p2[1], 'ro', markersize=3)
    
    plt.title(f'{len(pts1_vis)} matches visualized')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def compute_reprojection_error(
    pts3D: np.ndarray,
    pts2D: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> float:
    """
    Compute mean reprojection error.
    """
    t = t.reshape(3, 1)
    P = K @ np.hstack([R, t])
    
    errors = []
    for X, x in zip(pts3D, pts2D):
        X_h = np.append(X, 1)
        proj = P @ X_h
        proj = proj[:2] / proj[2]
        errors.append(np.linalg.norm(proj - x))
    
    return np.mean(errors)


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    from data_loader import load_dataset
    
    DATA_DIR = "data/rathaus"
    
    print("=" * 50)
    print("Two-View Initialization Test")
    print("=" * 50)
    
    # Load dataset
    images, cameras, names = load_dataset(DATA_DIR)
    
    if len(images) < 2:
        print("Need at least 2 images!")
        exit(1)
    
    # Get intrinsics from first camera
    K = cameras[0]['K']
    print(f"\nIntrinsic matrix K:\n{K}")
    
    # Initialize from first two views
    R, t, pts3D, pts1, pts2 = initialize_two_view(images[0], images[1], K)
    
    print(f"\nResults:")
    print(f"  R:\n{R}")
    print(f"  t: {t}")
    print(f"  Triangulated {len(pts3D)} points")
    
    # Compute reprojection errors
    R1 = np.eye(3)
    t1 = np.zeros(3)
    err1 = compute_reprojection_error(pts3D, pts1, K, R1, t1)
    err2 = compute_reprojection_error(pts3D, pts2, K, R, t)
    print(f"  Reprojection error: {err1:.2f}px (img1), {err2:.2f}px (img2)")
    
    # Compare with ground truth if available
    if len(cameras) >= 2:
        print(f"\nGround truth pose (camera 1):")
        print(f"  R:\n{cameras[1]['R']}")
        print(f"  t: {cameras[1]['t']}")
    
    # Visualize matches
    draw_matches(images[0], images[1], pts1, pts2, 
                 save_path="outputs/two_view_matches.png")
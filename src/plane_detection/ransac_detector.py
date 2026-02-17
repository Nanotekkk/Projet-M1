"""Enhanced RANSAC Plane Detector

Author: Matheo LANCEA
Description: RANSAC algorithm for robust multi-plane detection in point clouds
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PlaneResult:
    """Result of plane detection"""
    plane_id: int
    normal: np.ndarray  # Normal vector (a, b, c)
    distance: float     # Distance to origin (d in ax+by+cz+d=0)
    inlier_indices: np.ndarray  # Indices of inlier points
    inlier_count: int
    color: Tuple[float, float, float]  # RGB color for visualization


class RANSACPlaneDetector:
    """
    RANSAC-based multi-plane detector for point clouds.
    
    Detects multiple planes iteratively, assigning each a unique color
    for visualization in Open3D.
    """

    # Default colors for planes (RGB)
    PLANE_COLORS = [
        (1.0, 0.0, 0.0),      # Red
        (0.0, 1.0, 0.0),      # Green
        (0.0, 0.0, 1.0),      # Blue
        (1.0, 1.0, 0.0),      # Yellow
        (1.0, 0.0, 1.0),      # Magenta
        (0.0, 1.0, 1.0),      # Cyan
        (1.0, 0.5, 0.0),      # Orange
        (0.5, 0.0, 1.0),      # Purple
        (0.0, 0.5, 1.0),      # Light Blue
        (1.0, 0.0, 0.5),      # Pink
    ]

    def __init__(
        self,
        distance_threshold: float = 0.1,
        iterations: int = 1000,
        min_points_per_plane: int = 10,
        max_planes: int = 10,
        inlier_ratio_threshold: float = 0.05,
    ):
        """
        Initialize RANSAC plane detector.

        Args:
            distance_threshold: Max distance from point to plane to be inlier
            iterations: Number of RANSAC iterations per plane
            min_points_per_plane: Minimum points to define a plane
            max_planes: Maximum number of planes to detect
            inlier_ratio_threshold: Min ratio of inliers to consider a plane valid
        """
        self.distance_threshold = distance_threshold
        self.iterations = iterations
        self.min_points_per_plane = min_points_per_plane
        self.max_planes = max_planes
        self.inlier_ratio_threshold = inlier_ratio_threshold

    def detect_planes(self, points: np.ndarray) -> List[PlaneResult]:
        """
        Detect multiple planes in point cloud.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            List of PlaneResult objects for each detected plane
        """
        remaining_points = np.arange(len(points))
        planes = []
        plane_id = 0

        while len(remaining_points) > self.min_points_per_plane and \
              plane_id < self.max_planes:
            
            # Get 3D coordinates of remaining points
            remaining_coords = points[remaining_points]

            # Detect single best plane
            best_plane, best_inliers_local = self._fit_best_plane(remaining_coords)

            if best_plane is None:
                break

            # Calculate inlier ratio
            inlier_ratio = len(best_inliers_local) / len(remaining_coords)
            if inlier_ratio < self.inlier_ratio_threshold:
                break

            # Convert local indices to global
            best_inliers_global = remaining_points[best_inliers_local]

            # Create plane result with color
            color = self.PLANE_COLORS[plane_id % len(self.PLANE_COLORS)]
            plane_result = PlaneResult(
                plane_id=plane_id,
                normal=best_plane[:3],
                distance=best_plane[3],
                inlier_indices=best_inliers_global,
                inlier_count=len(best_inliers_global),
                color=color,
            )
            planes.append(plane_result)

            # Remove inliers from remaining points
            outlier_mask = np.ones(len(remaining_points), dtype=bool)
            outlier_mask[best_inliers_local] = False
            remaining_points = remaining_points[outlier_mask]

            plane_id += 1

        return planes

    def _fit_best_plane(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Find best fitting plane using RANSAC.

        Args:
            points: Point cloud subset (N, 3)

        Returns:
            Tuple of (plane_coefficients, inlier_indices) or (None, []) if no plane found
        """
        if len(points) < self.min_points_per_plane:
            return None, np.array([], dtype=int)

        best_inliers = np.array([], dtype=int)
        best_plane = None

        for _ in range(self.iterations):
            # Random sample of 3 points
            indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[indices]

            # Fit plane
            plane = self._fit_plane(sample_points)
            if plane is None:
                continue

            # Calculate distances
            distances = self._point_to_plane_distance(points, plane)

            # Find inliers
            inliers = np.where(distances < self.distance_threshold)[0]

            # Update best plane
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = plane

        return best_plane, best_inliers

    @staticmethod
    def _fit_plane(points: np.ndarray) -> Optional[np.ndarray]:
        """
        Fit plane to 3 points using SVD.
        
        Plane equation: ax + by + cz + d = 0
        
        Args:
            points: 3 points (3, 3)

        Returns:
            Plane coefficients [a, b, c, d] or None if degenerate
        """
        # Center points at origin
        centroid = points.mean(axis=0)
        centered = points - centroid

        # SVD to find normal
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]

        # Normalize
        normal = normal / np.linalg.norm(normal)

        # Distance to origin
        d = -np.dot(normal, centroid)

        return np.array([normal[0], normal[1], normal[2], d])

    @staticmethod
    def _point_to_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
        """
        Calculate distance from points to plane.

        Args:
            points: Point cloud (N, 3)
            plane: Plane coefficients [a, b, c, d]

        Returns:
            Distances (N,)
        """
        a, b, c, d = plane
        numerator = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        denominator = np.sqrt(a**2 + b**2 + c**2)
        return numerator / denominator

    def detect_single_plane(self, points: np.ndarray) -> PlaneResult:
        """
        Detect a single plane in point cloud.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            PlaneResult for the best fitting plane
        """
        best_plane, best_inliers = self._fit_best_plane(points)

        if best_plane is None:
            raise ValueError("Could not fit a plane to the point cloud")

        return PlaneResult(
            plane_id=0,
            normal=best_plane[:3],
            distance=best_plane[3],
            inlier_indices=best_inliers,
            inlier_count=len(best_inliers),
            color=(1.0, 0.0, 0.0),
        )

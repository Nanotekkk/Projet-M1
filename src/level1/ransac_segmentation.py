"""RANSAC Segmentation for Ground Detection - Level 1"""

import numpy as np
from typing import Tuple
from src.core.segmentation_strategy import SegmentationStrategy, SegmentationResult


class RANSACSegmentation(SegmentationStrategy):
    """
    RANSAC-based ground plane detection.

    RANSAC (Random Sample Consensus) iteratively selects random samples
    to fit a plane and counts inliers. The plane with the most inliers
    is considered the ground plane.
    """

    def __init__(
        self,
        distance_threshold: float = 0.2,
        iterations: int = 1000,
        min_points_for_plane: int = 3,
    ):
        """
        Initialize RANSAC segmentation.

        Args:
            distance_threshold: Maximum distance from point to plane to be inlier
            iterations: Number of RANSAC iterations
            min_points_for_plane: Minimum points needed to define a plane
        """
        self.distance_threshold = distance_threshold
        self.iterations = iterations
        self.min_points_for_plane = min_points_for_plane

    def segment(self, points: np.ndarray) -> SegmentationResult:
        """
        Segment ground points using RANSAC.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            SegmentationResult with ground point indices
        """
        if len(points) < self.min_points_for_plane:
            raise ValueError("Not enough points for plane fitting")

        best_inliers = np.array([], dtype=int)
        best_plane = None

        for _ in range(self.iterations):
            # Random sample of 3 points
            sample_indices = np.random.choice(
                len(points), self.min_points_for_plane, replace=False
            )
            sample_points = points[sample_indices]

            # Fit plane using least squares
            plane = self._fit_plane(sample_points)
            if plane is None:
                continue

            # Calculate distances from all points to plane
            distances = self._point_to_plane_distance(points, plane)

            # Count inliers
            inliers = np.where(distances < self.distance_threshold)[0]

            # Update best plane if this has more inliers
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = plane

        labels = np.zeros(len(points), dtype=int)
        labels[best_inliers] = 1

        return SegmentationResult(
            labels=labels,
            ground_indices=best_inliers,
            metadata={
                "method": "RANSAC",
                "plane_normal": best_plane[:3] if best_plane is not None else None,
                "plane_distance": best_plane[3] if best_plane is not None else None,
                "num_inliers": len(best_inliers),
                "num_outliers": len(points) - len(best_inliers),
            },
        )

    @staticmethod
    def _fit_plane(points: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Fit a plane to points using least squares.

        Plane equation: ax + by + cz + d = 0

        Args:
            points: At least 3 points (N, 3)

        Returns:
            Plane parameters (a, b, c, d) normalized
        """
        if len(points) < 3:
            return None

        # Compute centroid
        centroid = points.mean(axis=0)

        # Center points
        centered = points - centroid

        # SVD to find plane normal
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1, :]

        # Normalize
        normal = normal / np.linalg.norm(normal)

        # Plane distance
        d = -np.dot(normal, centroid)

        return np.append(normal, d)

    @staticmethod
    def _point_to_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
        """
        Calculate perpendicular distance from points to plane.

        Args:
            points: Point cloud (N, 3)
            plane: Plane parameters (a, b, c, d)

        Returns:
            Distances (N,)
        """
        a, b, c, d = plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        distances /= np.sqrt(a**2 + b**2 + c**2)
        return distances

    def get_name(self) -> str:
        return "RANSAC Ground Detection"

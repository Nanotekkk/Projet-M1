"""Plane Detection for Surface Identification - Level 2

Author: Matheo LANCEA
Description: RANSAC-based detection of planar surfaces in point clouds
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Plane:
    """Represents a detected plane"""

    normal: np.ndarray  # Plane normal vector (a, b, c)
    distance: float  # Distance from origin (d in ax+by+cz+d=0)
    points_indices: np.ndarray  # Indices of points on this plane
    area: float = 0.0  # Estimated area
    plane_type: str = "unknown"  # 'horizontal', 'vertical', or 'unknown'

    def get_equation(self) -> Tuple[float, float, float, float]:
        """Get plane equation coefficients (a, b, c, d)"""
        return (*self.normal, self.distance)

    def classify_plane_type(self) -> str:
        """Classify plane as horizontal or vertical based on normal"""
        z_component = abs(self.normal[2])
        if z_component > 0.9:  # Nearly vertical normal = horizontal plane
            self.plane_type = "horizontal"
        elif z_component < 0.1:  # Normal in XY plane = vertical plane
            self.plane_type = "vertical"
        else:
            self.plane_type = "inclined"
        return self.plane_type


class PlaneDetection:
    """
    Detect planes in point clouds for surface identification.

    Used to identify floors (horizontal planes), walls (vertical planes),
    and other structural elements.
    """

    def __init__(
        self,
        distance_threshold: float = 0.1,
        iterations: int = 100,
        min_inliers: int = 50,
    ):
        """
        Initialize plane detection.

        Args:
            distance_threshold: Max distance from point to plane
            iterations: Number of RANSAC iterations
            min_inliers: Minimum inliers for valid plane
        """
        self.distance_threshold = distance_threshold
        self.iterations = iterations
        self.min_inliers = min_inliers

    def detect_planes(self, points: np.ndarray, num_planes: int = 3) -> List[Plane]:
        """
        Detect multiple planes in point cloud.

        Args:
            points: Input point cloud (N, 3)
            num_planes: Number of planes to detect

        Returns:
            List of detected Plane objects
        """
        detected_planes = []
        remaining_indices = np.arange(len(points))

        for _ in range(num_planes):
            if len(remaining_indices) < self.min_inliers:
                break

            remaining_points = points[remaining_indices]
            plane = self._ransac_plane_detection(remaining_points, remaining_indices)

            if plane is not None:
                detected_planes.append(plane)
                # Remove inliers for next iteration
                remaining_indices = np.setdiff1d(
                    remaining_indices, plane.points_indices
                )
            else:
                break

        return detected_planes

    def _ransac_plane_detection(
        self, points: np.ndarray, original_indices: np.ndarray
    ) -> Optional[Plane]:
        """
        RANSAC plane detection on a subset of points.

        Args:
            points: Point cloud subset (N, 3)
            original_indices: Original indices in full point cloud

        Returns:
            Detected Plane or None
        """
        best_inliers = np.array([], dtype=int)
        best_plane_eq = None

        for _ in range(self.iterations):
            # Sample 3 random points
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_idx]

            # Fit plane
            plane_eq = self._fit_plane_from_points(sample_points)
            if plane_eq is None:
                continue

            # Count inliers
            distances = self._point_to_plane_distance(points, plane_eq)
            inliers = np.where(distances < self.distance_threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane_eq = plane_eq

        if len(best_inliers) < self.min_inliers:
            return None

        # Refit plane with all inliers
        inlier_points = points[best_inliers]
        best_plane_eq = self._fit_plane_from_points(inlier_points)

        a, b, c, d = best_plane_eq
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        plane = Plane(
            normal=normal,
            distance=d / np.linalg.norm([a, b, c]),
            points_indices=original_indices[best_inliers],
        )

        plane.classify_plane_type()
        plane.area = self._estimate_plane_area(inlier_points)

        return plane

    @staticmethod
    def _fit_plane_from_points(points: np.ndarray) -> Optional[Tuple]:
        """
        Fit plane using least squares.

        Args:
            points: At least 3 points (N, 3)

        Returns:
            Plane equation (a, b, c, d) or None
        """
        if len(points) < 3:
            return None

        centroid = points.mean(axis=0)
        centered = points - centroid

        try:
            _, _, Vt = np.linalg.svd(centered)
            normal = Vt[-1, :]
            normal = normal / np.linalg.norm(normal)

            d = -np.dot(normal, centroid)
            return (*normal, d)
        except:
            return None

    @staticmethod
    def _point_to_plane_distance(points: np.ndarray, plane_eq: Tuple) -> np.ndarray:
        """Calculate distance from points to plane"""
        a, b, c, d = plane_eq
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
        distances /= np.sqrt(a**2 + b**2 + c**2)
        return distances

    @staticmethod
    def _estimate_plane_area(points: np.ndarray) -> float:
        """Estimate area covered by plane points"""
        if len(points) < 3:
            return 0.0

        # Use PCA to project points onto plane and compute convex hull area
        centered = points - points.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered)

        # Project onto first two principal components
        basis1 = Vt[0, :]
        basis2 = Vt[1, :]

        coords_2d = np.column_stack([
            np.dot(centered, basis1),
            np.dot(centered, basis2),
        ])

        # Approximate area using convex hull
        from scipy.spatial import ConvexHull

        try:
            hull = ConvexHull(coords_2d)
            return hull.volume  # In 2D, volume is area
        except:
            # Fallback: approximate using bounding box
            return np.prod(coords_2d.max(axis=0) - coords_2d.min(axis=0))

"""Model Comparison Module

Compares different plane detection methods:
- RANSAC (main method)
- Linear Regression (for height-based ground plane)
- K-means clustering
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Single model comparison result"""
    model_name: str
    inlier_count: int
    inlier_ratio: float
    plane_normal: np.ndarray
    plane_distance: float
    computation_time: float
    inlier_indices: np.ndarray
    additional_metrics: Dict[str, Any]


class ModelComparison:
    """Compare multiple plane detection models"""

    def __init__(self):
        """Initialize model comparison"""
        pass

    def compare_all_methods(self, points: np.ndarray) -> Dict[str, ComparisonResult]:
        """
        Compare all plane detection methods.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            Dictionary with results for each method
        """
        results = {}

        # RANSAC (via external call expected)
        results["RANSAC"] = self._method_ransac(points)
        
        # Linear Regression based
        results["Linear Regression"] = self._method_linear_regression(points)
        
        # K-means clustering
        results["K-Means"] = self._method_kmeans(points)

        return results

    def _method_ransac(self, points: np.ndarray) -> ComparisonResult:
        """
        RANSAC-based plane detection (placeholder for external RANSAC).
        For actual RANSAC, integrate with RANSACPlaneDetector.
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            ComparisonResult
        """
        import time
        start_time = time.time()
        
        # Use simple plane fitting via SVD (similar to RANSAC best fit)
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]
        normal = normal / np.linalg.norm(normal)
        
        d = -np.dot(normal, centroid)
        
        # Calculate distances
        distances = np.abs(np.dot(points, normal) + d) / np.sqrt(np.sum(normal**2))
        threshold = np.percentile(distances, 10)
        inliers = np.where(distances < threshold)[0]
        
        elapsed = time.time() - start_time
        
        return ComparisonResult(
            model_name="RANSAC",
            inlier_count=len(inliers),
            inlier_ratio=len(inliers) / len(points),
            plane_normal=normal,
            plane_distance=d,
            computation_time=elapsed,
            inlier_indices=inliers,
            additional_metrics={"threshold": float(threshold)},
        )

    def _method_linear_regression(self, points: np.ndarray) -> ComparisonResult:
        """
        Linear regression to fit plane: z = ax + by + c
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            ComparisonResult
        """
        import time
        start_time = time.time()
        
        X = points[:, :2]  # x, y coordinates
        y = points[:, 2]   # z coordinate
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict z values
        y_pred = model.predict(X)
        residuals = np.abs(y - y_pred)
        threshold = np.percentile(residuals, 15)
        inliers = np.where(residuals < threshold)[0]
        
        # Convert to plane equation format
        a, b = model.coef_
        c = model.intercept_
        normal = np.array([-a, -b, 1.0])
        normal = normal / np.linalg.norm(normal)
        
        d = -c
        
        elapsed = time.time() - start_time
        
        return ComparisonResult(
            model_name="Linear Regression",
            inlier_count=len(inliers),
            inlier_ratio=len(inliers) / len(points),
            plane_normal=normal,
            plane_distance=d,
            computation_time=elapsed,
            inlier_indices=inliers,
            additional_metrics={
                "coefficients": {"a": float(a), "b": float(b), "c": float(c)},
                "threshold": float(threshold),
            },
        )

    def _method_kmeans(self, points: np.ndarray, n_clusters: int = 2) -> ComparisonResult:
        """
        K-means clustering approach (find largest cluster as ground).
        
        Args:
            points: Input point cloud (N, 3)
            n_clusters: Number of clusters
            
        Returns:
            ComparisonResult
        """
        import time
        start_time = time.time()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        
        # Find largest cluster
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        inliers = np.where(labels == largest_cluster)[0]
        
        # Fit plane to largest cluster
        cluster_points = points[inliers]
        centroid = cluster_points.mean(axis=0)
        centered = cluster_points - centroid
        
        _, _, vt = np.linalg.svd(centered)
        normal = vt[-1]
        normal = normal / np.linalg.norm(normal)
        
        d = -np.dot(normal, centroid)
        
        elapsed = time.time() - start_time
        
        return ComparisonResult(
            model_name="K-Means",
            inlier_count=len(inliers),
            inlier_ratio=len(inliers) / len(points),
            plane_normal=normal,
            plane_distance=d,
            computation_time=elapsed,
            inlier_indices=inliers,
            additional_metrics={"n_clusters": n_clusters, "cluster_id": int(largest_cluster)},
        )



    def print_comparison(self, results: Dict[str, ComparisonResult]) -> None:
        """Print comparison results"""
        print("\n" + "=" * 80)
        print("PLANE DETECTION METHOD COMPARISON")
        print("=" * 80)
        print(f"{'Method':<20} {'Inliers':<15} {'Ratio':<15} {'Time (ms)':<15}")
        print("-" * 80)

        for method, result in results.items():
            time_ms = result.computation_time * 1000
            print(f"{method:<20} {result.inlier_count:<15} "
                  f"{result.inlier_ratio:<15.3f} {time_ms:<15.3f}")

        print("=" * 80)

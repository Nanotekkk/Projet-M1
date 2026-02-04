"""Euclidean Clustering for Point Cloud Segmentation - Level 2

Author: Matheo LANCEA
Description: DBSCAN-based clustering for multi-element segmentation
"""

import numpy as np
from typing import List, Tuple
from sklearn.cluster import DBSCAN


class ClusterResult:
    """Result of clustering operation"""

    def __init__(self, labels: np.ndarray, cluster_centers: np.ndarray):
        self.labels = labels
        self.cluster_centers = cluster_centers
        self.n_clusters = len(cluster_centers)

    def get_cluster_points(self, points: np.ndarray, cluster_id: int) -> np.ndarray:
        """Get points belonging to a specific cluster"""
        return points[self.labels == cluster_id]


class EuclideanClustering:
    """
    Euclidean distance-based clustering for segmenting point clouds.

    Uses DBSCAN algorithm to group nearby points, useful for identifying
    ground, walls, obstacles, and other geometric elements.
    """

    def __init__(self, eps: float = 0.5, min_points: int = 5):
        """
        Initialize clustering.

        Args:
            eps: Maximum distance between points in a cluster
            min_points: Minimum points to form a cluster
        """
        self.eps = eps
        self.min_points = min_points

    def cluster(self, points: np.ndarray) -> ClusterResult:
        """
        Perform euclidean clustering on point cloud.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            ClusterResult with cluster labels and centers
        """
        # Apply DBSCAN
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_points)
        labels = clusterer.fit_predict(points)

        # Calculate cluster centers
        cluster_centers = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise points
                continue
            cluster_points = points[labels == cluster_id]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)

        cluster_centers = np.array(cluster_centers) if cluster_centers else np.empty((0, 3))

        return ClusterResult(labels, cluster_centers)

    def cluster_with_normals(
        self, points: np.ndarray, normals: np.ndarray
    ) -> ClusterResult:
        """
        Perform clustering considering both position and normals.

        Args:
            points: Input point cloud (N, 3)
            normals: Surface normals (N, 3)

        Returns:
            ClusterResult with cluster labels and centers
        """
        # Combine spatial and normal information
        combined = np.hstack([points, normals * 0.5])  # Weight normals less
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_points)
        labels = clusterer.fit_predict(combined)

        # Calculate cluster centers (only from positions)
        cluster_centers = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_points = points[labels == cluster_id]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)

        cluster_centers = np.array(cluster_centers) if cluster_centers else np.empty((0, 3))

        return ClusterResult(labels, cluster_centers)

    def adaptive_clustering(self, points: np.ndarray) -> ClusterResult:
        """
        Adaptive clustering with automatic eps parameter estimation.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            ClusterResult with cluster labels and centers
        """
        # Estimate eps from k-distance graph
        from sklearn.neighbors import NearestNeighbors

        k = max(self.min_points, int(np.sqrt(len(points))))
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(points)
        distances, _ = neighbors.kneighbors(points)
        distances = np.sort(distances[:, k - 1], axis=0)

        # Use elbow method to find optimal eps
        optimal_eps = np.percentile(distances, 90)

        clusterer = DBSCAN(eps=optimal_eps, min_samples=self.min_points)
        labels = clusterer.fit_predict(points)

        # Calculate cluster centers
        cluster_centers = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster_points = points[labels == cluster_id]
            center = cluster_points.mean(axis=0)
            cluster_centers.append(center)

        cluster_centers = np.array(cluster_centers) if cluster_centers else np.empty((0, 3))

        return ClusterResult(labels, cluster_centers)

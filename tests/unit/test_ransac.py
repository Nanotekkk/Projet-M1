"""Tests for RANSAC Plane Detector

Author: Matheo LANCEA
Description: Unit tests for the RANSAC plane detection module
"""

import pytest
import numpy as np
from src.plane_detection.ransac_detector import RANSACPlaneDetector, PlaneResult


class TestRANSACPlaneDetector:
    """Test suite for RANSAC plane detector"""

    @pytest.fixture
    def detector(self):
        """Create a RANSAC detector instance"""
        return RANSACPlaneDetector(
            distance_threshold=0.1,
            iterations=100,
            min_points_per_plane=10,
            max_planes=5,
        )

    @pytest.fixture
    def synthetic_plane(self):
        """Create a synthetic plane point cloud"""
        # Create points on a plane z = 0
        np.random.seed(42)
        x = np.random.uniform(-5, 5, 1000)
        y = np.random.uniform(-5, 5, 1000)
        z = np.random.normal(0, 0.1, 1000)  # Small noise
        return np.column_stack([x, y, z])

    @pytest.fixture
    def synthetic_multi_plane(self):
        """Create multiple planes"""
        np.random.seed(42)
        planes = []

        # Plane 1: z = 0
        p1 = np.random.uniform(-5, 5, (1000, 2))
        p1 = np.column_stack([p1, np.random.normal(0, 0.1, 1000)])
        planes.append(p1)

        # Plane 2: z = 5
        p2 = np.random.uniform(-5, 5, (800, 2))
        p2 = np.column_stack([p2, np.random.normal(5, 0.1, 800)])
        planes.append(p2)

        # Plane 3: z = 10
        p3 = np.random.uniform(-5, 5, (600, 2))
        p3 = np.column_stack([p3, np.random.normal(10, 0.1, 600)])
        planes.append(p3)

        return np.vstack(planes)

    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.distance_threshold == 0.1
        assert detector.iterations == 100
        assert detector.min_points_per_plane == 10
        assert detector.max_planes == 5

    def test_single_plane_detection(self, detector, synthetic_plane):
        """Test detection of a single plane"""
        result = detector.detect_single_plane(synthetic_plane)

        assert isinstance(result, PlaneResult)
        assert result.plane_id == 0
        assert result.inlier_count > 800  # Should detect most points
        assert len(result.normal) == 3
        assert len(result.inlier_indices) > 0

    def test_single_plane_normal(self, detector, synthetic_plane):
        """Test that detected plane normal is approximately [0, 0, 1]"""
        result = detector.detect_single_plane(synthetic_plane)

        # Normal should be approximately [0, 0, ±1]
        expected_normal = np.array([0, 0, 1])
        dot_product = abs(np.dot(result.normal, expected_normal))

        # Allow some tolerance due to sampling
        assert dot_product > 0.8

    def test_multi_plane_detection(self, detector, synthetic_multi_plane):
        """Test detection of multiple planes"""
        planes = detector.detect_planes(synthetic_multi_plane)

        assert len(planes) >= 2
        assert len(planes) <= 5

        for plane in planes:
            assert isinstance(plane, PlaneResult)
            assert plane.inlier_count > 0
            assert len(plane.inlier_indices) == plane.inlier_count

    def test_plane_color_assignment(self, detector, synthetic_multi_plane):
        """Test that planes get different colors"""
        planes = detector.detect_planes(synthetic_multi_plane)

        colors = [plane.color for plane in planes]

        # Check all colors are in valid RGB range
        for color in colors:
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 1

        # Check at least some colors are different
        if len(colors) > 1:
            assert not all(c == colors[0] for c in colors)

    def test_plane_fitting(self, detector):
        """Test internal plane fitting"""
        # Create 3 coplanar points
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

        plane = detector._fit_plane(points)

        assert plane is not None
        assert len(plane) == 4
        assert abs(np.linalg.norm(plane[:3]) - 1) < 1e-10  # Should be normalized

    def test_point_to_plane_distance(self, detector):
        """Test distance calculation from point to plane"""
        # Plane: z = 1 (normal = [0, 0, 1], distance = -1)
        plane = np.array([0, 0, 1, -1])

        points = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 0]], dtype=np.float64)

        distances = detector._point_to_plane_distance(points, plane)

        assert len(distances) == 3
        assert distances[0] < 1e-9  # Point on plane
        assert distances[1] > 0.9  # Point above plane
        assert distances[2] > 0.9  # Point below plane

    def test_min_points_error(self, detector):
        """Test error when too few points provided"""
        points = np.array([[0, 0, 0], [1, 1, 1]])

        with pytest.raises(ValueError):
            detector.detect_single_plane(points)

    def test_empty_point_cloud(self, detector):
        """Test handling of empty point cloud"""
        points = np.array([]).reshape(0, 3)

        with pytest.raises(ValueError):
            detector.detect_single_plane(points)

    def test_plane_result_structure(self, detector, synthetic_plane):
        """Test PlaneResult data structure"""
        result = detector.detect_single_plane(synthetic_plane)

        # Check all required attributes
        assert hasattr(result, 'plane_id')
        assert hasattr(result, 'normal')
        assert hasattr(result, 'distance')
        assert hasattr(result, 'inlier_indices')
        assert hasattr(result, 'inlier_count')
        assert hasattr(result, 'color')

        # Check attribute types
        assert isinstance(result.plane_id, int)
        assert isinstance(result.normal, np.ndarray)
        assert isinstance(result.distance, (int, float, np.number))
        assert isinstance(result.inlier_indices, np.ndarray)
        assert isinstance(result.inlier_count, (int, np.integer))
        assert isinstance(result.color, tuple)

    def test_normal_normalization(self, detector, synthetic_plane):
        """Test that plane normals are normalized"""
        result = detector.detect_single_plane(synthetic_plane)

        norm = np.linalg.norm(result.normal)
        assert abs(norm - 1.0) < 1e-10

    def test_reproducibility(self, synthetic_plane):
        """Test that results are reproducible with fixed seed"""
        detector1 = RANSACPlaneDetector(iterations=50)
        detector2 = RANSACPlaneDetector(iterations=50)

        np.random.seed(42)
        result1 = detector1.detect_single_plane(synthetic_plane)

        np.random.seed(42)
        result2 = detector2.detect_single_plane(synthetic_plane)

        assert result1.inlier_count == result2.inlier_count

    def test_plane_colors_uniqueness(self):
        """Test that RANSAC has enough unique colors"""
        colors = RANSACPlaneDetector.PLANE_COLORS

        assert len(colors) >= 10
        
        # Check all are valid RGB tuples
        for color in colors:
            assert len(color) == 3
            for c in color:
                assert 0 <= c <= 1

    def test_best_plane_improves_with_iterations(self, synthetic_plane):
        """Test that more iterations generally find better planes"""
        detector_low = RANSACPlaneDetector(iterations=10)
        detector_high = RANSACPlaneDetector(iterations=500)

        np.random.seed(42)
        result_low = detector_low.detect_single_plane(synthetic_plane)

        np.random.seed(42)
        result_high = detector_high.detect_single_plane(synthetic_plane)

        # Higher iterations should find more or equal inliers
        assert result_high.inlier_count >= result_low.inlier_count * 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

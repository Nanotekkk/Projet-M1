"""Unit tests for Point Cloud Loading"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from src.core.point_cloud_loader import PointCloudLoader, PointCloud


class TestPointCloud:
    """Tests for PointCloud class"""

    def test_point_cloud_creation(self):
        """Test basic point cloud creation"""
        points = np.random.rand(100, 3)
        pc = PointCloud(points)

        assert len(pc) == 100
        assert pc.points.shape == (100, 3)

    def test_point_cloud_with_colors(self):
        """Test point cloud with colors"""
        points = np.random.rand(100, 3)
        colors = np.random.rand(100, 3)
        pc = PointCloud(points, colors)

        assert pc.colors is not None
        assert pc.colors.shape == (100, 3)

    def test_get_bounds(self):
        """Test bounds calculation"""
        points = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5]])
        pc = PointCloud(points)

        min_b, max_b = pc.get_bounds()
        np.testing.assert_array_equal(min_b, [0, 0, 0])
        np.testing.assert_array_equal(max_b, [10, 10, 10])

    def test_get_center(self):
        """Test center calculation"""
        points = np.array([[0, 0, 0], [10, 10, 10]])
        pc = PointCloud(points)

        center = pc.get_center()
        np.testing.assert_array_almost_equal(center, [5, 5, 5])

    def test_downsample(self):
        """Test downsampling"""
        points = np.random.rand(1000, 3)
        pc = PointCloud(points)

        downsampled = pc.downsample(0.1)
        assert len(downsampled) < len(pc)


class TestPointCloudLoader:
    """Tests for PointCloudLoader factory"""

    def test_create_synthetic_plane(self):
        """Test synthetic plane generation"""
        pc = PointCloudLoader.create_synthetic_plane(
            width=100, depth=100, num_points=1000
        )

        assert len(pc) == 1000
        assert pc.points.shape == (1000, 3)
        # Check that z coordinates are close to 0 (flat plane)
        assert np.abs(pc.points[:, 2]).max() < 1.0

    def test_create_synthetic_scene(self):
        """Test synthetic scene generation"""
        pc = PointCloudLoader.create_synthetic_scene(num_ground=1000, num_obstacles=500)

        # Allow for rounding in point distribution
        assert 1490 <= len(pc) <= 1510
        assert pc.points.shape[0] >= 1490


@pytest.mark.unit
class TestPointCloudIntegration:
    """Integration tests for point cloud operations"""

    def test_load_and_process_workflow(self):
        """Test complete workflow of loading and processing"""
        # Create synthetic data
        pc = PointCloudLoader.create_synthetic_plane(num_points=500)

        # Test operations
        assert len(pc) > 0
        min_b, max_b = pc.get_bounds()
        center = pc.get_center()

        assert np.all(min_b <= center) and np.all(max_b >= center)

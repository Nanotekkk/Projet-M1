"""Unit tests for Level 1 Segmentation"""

import pytest
import numpy as np
from src.level1.ransac_segmentation import RANSACSegmentation
from src.level1.dl_segmentation import DLSegmentation
from src.core.point_cloud_loader import PointCloudLoader


class TestRANSACSegmentation:
    """Tests for RANSAC ground detection"""

    @pytest.fixture
    def ground_plane(self):
        """Create synthetic ground plane"""
        return PointCloudLoader.create_synthetic_plane(num_points=500)

    def test_ransac_initialization(self):
        """Test RANSAC initialization"""
        ransac = RANSACSegmentation()
        assert ransac.distance_threshold > 0
        assert ransac.iterations > 0

    def test_ransac_plane_detection(self, ground_plane):
        """Test plane detection on synthetic plane"""
        ransac = RANSACSegmentation(distance_threshold=0.5, iterations=100)
        result = ransac.segment(ground_plane.points)

        assert result is not None
        assert len(result.ground_indices) > ground_plane.points.shape[0] * 0.7
        assert result.metadata['method'] == 'RANSAC'

    def test_ransac_with_obstacles(self):
        """Test RANSAC on scene with obstacles"""
        scene = PointCloudLoader.create_synthetic_scene()
        ransac = RANSACSegmentation(iterations=200)
        result = ransac.segment(scene.points)

        # Should detect most ground points
        assert len(result.ground_indices) > scene.points.shape[0] * 0.3

    def test_ransac_get_ground_points(self, ground_plane):
        """Test extracting ground points"""
        ransac = RANSACSegmentation()
        result = ransac.segment(ground_plane.points)

        ground_pts = result.get_ground_points(ground_plane.points)
        non_ground_pts = result.get_non_ground_points(ground_plane.points)

        assert len(ground_pts) + len(non_ground_pts) == len(ground_plane.points)

    def test_ransac_plane_fitting(self):
        """Test plane fitting function"""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        plane = RANSACSegmentation._fit_plane(points)

        assert plane is not None
        assert len(plane) == 4
        # Normal should be normalized
        norm = np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
        assert np.isclose(norm, 1.0)


@pytest.mark.unit
class TestDLSegmentation:
    """Tests for deep learning segmentation"""

    def test_dl_initialization(self):
        """Test DL model initialization"""
        dl = DLSegmentation()
        assert dl.model is not None
        assert dl.confidence_threshold > 0

    def test_dl_segment(self):
        """Test DL segmentation inference"""
        pc = PointCloudLoader.create_synthetic_plane(num_points=200)
        dl = DLSegmentation(confidence_threshold=0.5)
        result = dl.segment(pc.points)

        assert result is not None
        assert len(result.labels) == len(pc.points)

    def test_dl_training(self):
        """Test DL model training"""
        pc = PointCloudLoader.create_synthetic_plane(num_points=100)
        labels = np.ones(100, dtype=int)  # All ground

        dl = DLSegmentation()
        dl.train_model(pc.points, labels, epochs=2)

        # After training, inference should work
        result = dl.segment(pc.points)
        assert result is not None

    def test_dl_normalize_points(self):
        """Test point normalization via internal scaler"""
        points = np.array([[0, 0, 0], [10, 10, 10], [5, 5, 5], [2, 2, 2], [8, 8, 8], [6, 6, 6]], dtype=float)
        dl = DLSegmentation()
        
        # Train on points to initialize scaler
        labels = np.array([0, 1, 0, 1, 0, 1])
        dl.train_model(points, labels)
        
        # Normalized points should be scaled
        normalized = dl.scaler.transform(points)
        assert normalized.shape == points.shape


class TestGroundClassifier:
    """Tests for neural network model"""

    def test_model_creation(self):
        """Test model instantiation"""
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(64, 64))
        assert model is not None

    def test_model_forward(self):
        """Test model training and prediction"""
        from sklearn.neural_network import MLPClassifier
        import numpy as np
        
        model = MLPClassifier()
        X = np.random.randn(10, 3)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (10,)


@pytest.mark.unit
@pytest.mark.level1
class TestLevel1Integration:
    """Integration tests for Level 1 components"""

    def test_compare_ransac_vs_dl(self):
        """Compare RANSAC and DL segmentation methods"""
        scene = PointCloudLoader.create_synthetic_scene()

        ransac = RANSACSegmentation()
        dl = DLSegmentation()

        ransac_result = ransac.segment(scene.points)
        dl_result = dl.segment(scene.points)

        # Both should detect some ground points
        assert len(ransac_result.ground_indices) > 0
        assert len(dl_result.ground_indices) > 0
